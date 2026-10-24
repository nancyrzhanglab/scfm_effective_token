#!/usr/bin/env python
"""Embed the AIFI atlas with scGPT, in resumable blocks.
"""
import argparse, json, os, time, glob
import numpy as np, pandas as pd, torch, anndata as ad
from scipy.sparse import load_npz

ROOT = "/"
DEF  = os.path.join(ROOT, "data/scGPT_related/mat/Gong_immuneAtlas")

ap = argparse.ArgumentParser()
ap.add_argument("--prep-dir", default=os.path.join(DEF, "prep"))
ap.add_argument("--out-dir",  default=os.path.join(DEF, "emb"))
ap.add_argument("--model-dir", default=os.path.join(ROOT, "data/scGPT_related/models/full"))
ap.add_argument("--n-hvg", type=int, default=0)
ap.add_argument("--counts-npz", required=True)
ap.add_argument("--genes-txt", required=True)
ap.add_argument("--shard", type=int, required=True)
ap.add_argument("--n-shards", type=int, required=True)
ap.add_argument("--block", type=int, default=25_000, help="cells per checkpoint file")
ap.add_argument("--batch-size", type=int, default=0, help="0 = 128 on cuda, 64 on cpu")
ap.add_argument("--max-length", type=int, default=1200)
ap.add_argument("--device", default="cuda")
ap.add_argument("--require-cuda", action="store_true",
                help="fail instead of silently falling back to CPU (which would "
                     "not finish inside the 4 h gpu.q cap)")
ap.add_argument("--block-lo", type=int, default=0,
                help="first global block index this array is responsible for")
ap.add_argument("--block-hi", type=int, default=-1,
                help="one past the last global block index; -1 = through the end")
ap.add_argument("--seed", type=int, default=0)
A = ap.parse_args()
t0 = time.time(); log = lambda m: print(f"[{time.time()-t0:8.1f}s] s{A.shard}: {m}", flush=True)
os.makedirs(A.out_dir, exist_ok=True)
torch.manual_seed(A.seed); np.random.seed(A.seed)

# ---------------------------------------------------------------- device
dev = A.device
if dev == "cuda":
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        log(f"GPU: {p.name}  sm{p.major}{p.minor}  {p.total_memory/2**30:.1f} GiB  "
            f"{p.multi_processor_count} SMs  |  flash-attn capable (sm75+): "
            f"{(p.major, p.minor) >= (7, 5)}")
    else:
        msg = "cuda requested but torch.cuda.is_available() is False"
        if A.require_cuda:
            raise SystemExit(f"FATAL: {msg}")
        log(f"WARNING: {msg}; falling back to cpu"); dev = "cpu"
bs = A.batch_size or (128 if dev == "cuda" else 64)
log(f"device={dev} batch_size={bs} threads={torch.get_num_threads()}")

# ---------------------------------------------------------------- data
K = None
Cnt = load_npz(os.path.join(A.prep_dir, A.counts_npz)).tocsr()
genes = [l.strip() for l in open(os.path.join(A.prep_dir, A.genes_txt))]
N = Cnt.shape[0]
K = len(genes); assert Cnt.shape[1] == K

blocks = [(i, min(i + A.block, N)) for i in range(0, N, A.block)]
hi_b = len(blocks) if A.block_hi < 0 else A.block_hi
assert 0 <= A.block_lo < hi_b <= len(blocks), \
    f"bad block range [{A.block_lo},{hi_b}) over {len(blocks)} blocks"
scope = blocks[A.block_lo:hi_b]
mine = [b for j, b in enumerate(scope) if j % A.n_shards == A.shard]
log(f"{len(blocks)} global blocks of {A.block}; this array owns blocks "
    f"[{A.block_lo},{hi_b}) = {len(scope)} blocks covering "
    f"{sum(h-l for l,h in scope):,} cells; this shard owns {len(mine)}")

def path_of(lo, hi):
    return os.path.join(A.out_dir, f"emb_{lo:08d}_{hi:08d}.npy")

todo = [(lo, hi) for lo, hi in mine if not os.path.exists(path_of(lo, hi))]
log(f"{len(mine) - len(todo)} blocks already done, {len(todo)} to do")
if not todo:
    log("nothing to do"); raise SystemExit(0)

# ---------------------------------------------------------------- model
from scgpt.model import TransformerModel
from scgpt.tokenizer import GeneVocab
from scgpt.utils import load_pretrained
from scgpt.tasks.cell_emb import get_batch_cell_embeddings

vocab = GeneVocab.from_file(os.path.join(A.model_dir, "vocab.json"))
for s in ["<pad>", "<cls>", "<eoc>"]:
    if s not in vocab: vocab.append_token(s)
vocab.set_default_index(vocab["<pad>"])
cfg = json.load(open(os.path.join(A.model_dir, "args.json")))
gene_ids = np.array(vocab(genes), dtype=int)
assert (gene_ids >= 0).all() and (gene_ids != vocab["<pad>"]).all(), \
    "a selected HVG is not in the scGPT vocabulary"

model = TransformerModel(
    ntoken=len(vocab), d_model=cfg["embsize"], nhead=cfg["nheads"],
    d_hid=cfg["d_hid"], nlayers=cfg["nlayers"], nlayers_cls=cfg["n_layers_cls"],
    n_cls=1, vocab=vocab, dropout=cfg["dropout"], pad_token=cfg["pad_token"],
    pad_value=cfg["pad_value"], do_mvc=True, do_dab=False, use_batch_labels=False,
    domain_spec_batchnorm=False, explicit_zero_prob=False,
    use_fast_transformer=False, fast_transformer_backend="flash", pre_norm=False)
sd = torch.load(os.path.join(A.model_dir, "best_model.pt"), map_location="cpu")
n_wqkv = sum(1 for k in sd if "Wqkv." in k)
load_pretrained(model, sd, verbose=False)
w = model.transformer_encoder.layers[0].self_attn.in_proj_weight
ref = sd["transformer_encoder.layers.0.self_attn.Wqkv.weight"]
assert torch.equal(w.detach().cpu(), ref.cpu()), \
    "fused Wqkv -> in_proj remap did not land; attention would be RANDOM"
log(f"model loaded; {n_wqkv} fused attention tensors remapped and verified")
model.to(dev); model.eval()

# ---------------------------------------------------------------- blocks
done_cells = 0; tstart = time.time()
for bi, (lo, hi) in enumerate(todo):
    Xb = Cnt[lo:hi]
    nz = np.diff(Xb.indptr)
    assert nz.max() < A.max_length, (
        f"block {lo}:{hi} has cells at max_length -- the collator would take a "
        f"RANDOM gene subset and the embedding would not be reproducible")
    assert nz.min() > 0, f"block {lo}:{hi} has a cell with no expressed HVG"
    adata = ad.AnnData(X=Xb, var=pd.DataFrame({"feature_name": genes},
                                              index=np.arange(K).astype(str)))
    adata.var["id_in_vocab"] = gene_ids
    t1 = time.time()
    E = get_batch_cell_embeddings(
        adata, cell_embedding_mode="cls", model=model, vocab=vocab,
        max_length=A.max_length, batch_size=bs, model_configs=cfg,
        gene_ids=gene_ids, use_batch_labels=False)
    E = np.asarray(E, dtype=np.float32)
    assert E.shape == (hi - lo, cfg["embsize"]), f"bad shape {E.shape}"
    assert np.isfinite(E).all(), f"non-finite embedding in block {lo}:{hi}"
    nrm = np.linalg.norm(E, axis=1)
    assert np.allclose(nrm, 1.0, atol=1e-3), f"not unit-norm: {nrm.min()}..{nrm.max()}"
    tmp = path_of(lo, hi) + ".tmp.npy"
    np.save(tmp, E); os.replace(tmp, path_of(lo, hi))   # atomic: no half-written block
    dt = time.time() - t1; done_cells += hi - lo
    rate = done_cells / (time.time() - tstart)
    log(f"block {bi+1}/{len(todo)} [{lo:,},{hi:,}) {dt:.1f}s "
        f"({(hi-lo)/dt:.0f} cells/s); cumulative {rate:.0f} cells/s; "
        f"eta {(sum(h-l for l,h in todo)-done_cells)/max(rate,1e-9)/60:.1f} min")

json.dump(dict(shard=A.shard, n_shards=A.n_shards, device=dev, batch_size=bs,
               blocks_done=len(todo), cells=done_cells,
               cells_per_s=float(done_cells/(time.time()-tstart)), seed=A.seed),
          open(os.path.join(A.out_dir, f"shard{A.shard:03d}_run.json"), "w"), indent=2)
log("SHARD OK")
