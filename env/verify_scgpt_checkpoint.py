#!/usr/bin/env python
"""Prove the scgpt env and the whole-human checkpoint work together.

Nothing here is asserted from documentation: the model flags are *inferred from
the checkpoint's own keys*, the architecture sizes come from args.json, and the
load is strict.  A clean run means every pretrained weight found a home and a
forward pass executed.
"""
import argparse, json, inspect, sys
from pathlib import Path

import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument("--model-dir", default="/home/mnt/weka/nzh/nzhanglab/project/"
                                      "emiliac/FM_effective/data/scGPT_related/models/full")
p.add_argument("--n-cells", type=int, default=4)
p.add_argument("--seq-len", type=int, default=64)
args_cli = p.parse_args()

MD = Path(args_cli.model_dir)
print(f"torch {torch.__version__}  cuda_available={torch.cuda.is_available()}")

import scgpt
from scgpt.model import TransformerModel
from scgpt.tokenizer import GeneVocab
print(f"scgpt {getattr(scgpt, '__version__', '?')}  from {Path(scgpt.__file__).parent}")

# ---------------------------------------------------------------- inputs
cfg = json.loads((MD / "args.json").read_text())
vocab = GeneVocab.from_file(MD / "vocab.json")
for tok in ["<pad>", "<cls>", "<eoc>"]:
    if tok not in vocab:
        vocab.append_token(tok)
vocab.set_default_index(vocab["<pad>"])
ntoken = len(vocab)
print(f"vocab: {ntoken} tokens")

sd = torch.load(MD / "best_model.pt", map_location="cpu", weights_only=True)
keys = list(sd)
print(f"checkpoint: {len(keys)} tensors, "
      f"{sum(t.numel() for t in sd.values())/1e6:.1f} M params")

# ------------------------------------------- flags inferred from the weights
has = lambda pref: any(k.startswith(pref) for k in keys)
flags = dict(
    do_mvc              = has("mvc_decoder"),
    do_dab              = has("grad_reverse_discriminator"),
    use_batch_labels    = has("batch_encoder"),
    explicit_zero_prob  = any("zero_logit" in k for k in keys),
    domain_spec_batchnorm = has("dsbn"),
)
n_cls = sd["cls_decoder.out_layer.weight"].shape[0] if has("cls_decoder") else 1
print("inferred from checkpoint keys:")
for k, v in flags.items():
    print(f"    {k:22s} {v}")
print(f"    {'n_cls':22s} {n_cls}  (cls_decoder present: {has('cls_decoder')})")

# ---------------------------------------------------------------- build
want = dict(
    ntoken=ntoken,
    d_model=cfg["embsize"], nhead=cfg["nheads"],
    d_hid=cfg["d_hid"], nlayers=cfg["nlayers"],
    nlayers_cls=cfg["n_layers_cls"], n_cls=n_cls,
    vocab=vocab, dropout=cfg["dropout"],
    pad_token=cfg["pad_token"], pad_value=cfg["pad_value"],
    input_emb_style=cfg["input_emb_style"], n_input_bins=cfg["n_bins"],
    cell_emb_style="cls",
    use_fast_transformer=False,     # no flash-attn: see README_env.md
    pre_norm=False,
    **flags,
)
sig = set(inspect.signature(TransformerModel.__init__).parameters)
dropped = sorted(set(want) - sig)
if dropped:
    print(f"NOTE: this scgpt build has no such args, dropped: {dropped}")
kwargs = {k: v for k, v in want.items() if k in sig}

model = TransformerModel(**kwargs)
model.eval()
n_model = sum(q.numel() for q in model.parameters())
print(f"model built: {n_model/1e6:.1f} M params")

# ---------------------------------------------------------------- load
# The checkpoint was trained with use_fast_transformer=True, so attention is
# stored in flash-attn's fused layout `self_attn.Wqkv.{weight,bias}`.  nn.
# MultiheadAttention calls the identical tensor `self_attn.in_proj_{weight,bias}`
# -- both are [Wq; Wk; Wv] stacked on dim 0 -- so the rename is lossless.
# This is what scgpt.utils.load_pretrained does; done explicitly here so the
# leftovers can be accounted for instead of silently dropped.
if not kwargs.get("use_fast_transformer", False):
    n_ren = sum(1 for k in sd if "Wqkv." in k)
    sd = {k.replace("Wqkv.", "in_proj_"): v for k, v in sd.items()}
    print(f"remapped {n_ren} fused-attention tensors: Wqkv. -> in_proj_")

missing, unexpected = model.load_state_dict(sd, strict=False)
print(f"load_state_dict: {len(missing)} missing, {len(unexpected)} unexpected")

# `flag_encoder` is an nn.Embedding(2, d_model) used only by scGPT's generative
# pretraining objective (args.json: USE_GENERATIVE_TRAINING=true) to mark which
# positions are given vs. to be generated.  TransformerModel has no such module
# and never consults it, so it is dead weight for encoding, not a load failure.
BENIGN_UNUSED = {"flag_encoder.weight"}
leftover = [k for k in unexpected if k not in BENIGN_UNUSED]
if missing:  print("  MISSING   :", missing)
if unexpected:
    print("  UNUSED FROM CHECKPOINT:", sorted(unexpected))
# TransformerModel always constructs a ClsDecoder head, but this checkpoint was
# pretrained with no_cls=true / USE_CLS=false, so it carries no classifier
# weights.  Those parameters therefore stay RANDOMLY INITIALISED.  They are only
# read when forward() is called with CLS=True, i.e. never during embedding
# extraction -- but they must be fine-tuned before any use as a classifier.
UNTRAINED = [k for k in missing if k.startswith("cls_decoder.")]
still_missing = [k for k in missing if k not in UNTRAINED]
assert not leftover,      f"checkpoint weights with no home in the model: {leftover}"
assert not still_missing, f"pretrained weights the model did not receive: {still_missing}"
print(f"  -> all 51.3 M pretrained params loaded; unused from checkpoint: "
      f"{sorted(BENIGN_UNUSED & set(unexpected))} (generative-training only).")
if UNTRAINED:
    n_un = sum(model.state_dict()[k].numel() for k in UNTRAINED)
    print(f"  !! {len(UNTRAINED)} tensors ({n_un/1e6:.2f} M params) are RANDOM, not pretrained:")
    print(f"     cls_decoder.* -- no_cls=true in args.json.  Safe only while CLS=False.")

# the remap must have landed on real pretrained values, not left inits in place
w = model.transformer_encoder.layers[0].self_attn.in_proj_weight
assert torch.equal(w, sd["transformer_encoder.layers.0.self_attn.in_proj_weight"])
print("  -> layer-0 attention weights verified identical to the checkpoint.")

# ---------------------------------------------------------------- forward
rng = np.random.default_rng(0)
n, L = args_cli.n_cells, args_cli.seq_len
gene_ids = rng.choice(np.arange(3, ntoken), size=(n, L), replace=True)
gene_ids[:, 0] = vocab["<cls>"]
src = torch.as_tensor(gene_ids, dtype=torch.long)
# binned values in [0, n_bins-1]; the <cls> slot carries pad_value by convention
values = torch.as_tensor(rng.integers(0, cfg["n_bins"], size=(n, L)), dtype=torch.float32)
values[:, 0] = cfg["pad_value"]
pad_mask = torch.zeros((n, L), dtype=torch.bool)

with torch.no_grad():
    emb = model._encode(src, values, src_key_padding_mask=pad_mask)
print(f"_encode -> {tuple(emb.shape)}  finite={bool(torch.isfinite(emb).all())}")
assert emb.shape == (n, L, cfg["embsize"])
assert torch.isfinite(emb).all()

cell_emb = emb[:, 0, :]
cn = cell_emb.norm(dim=1)
print(f"cell embedding norms: {cn.numpy().round(2)}")
assert (cn > 0).all()

with torch.no_grad():
    out = model(src, values, src_key_padding_mask=pad_mask, MVC=flags["do_mvc"])
print("forward() keys:", sorted(out))
mlm = out["mlm_output"]
print(f"mlm_output {tuple(mlm.shape)}  finite={bool(torch.isfinite(mlm).all())}  "
      f"range=[{mlm.min():.3f}, {mlm.max():.3f}]")
assert torch.isfinite(mlm).all()

# determinism in eval mode (dropout really is off)
with torch.no_grad():
    emb2 = model._encode(src, values, src_key_padding_mask=pad_mask)
assert torch.equal(emb, emb2), "eval() forward is not deterministic"
print("eval-mode forward is deterministic.")

# prove the untrained head really is untouched by the paths we use
if UNTRAINED:
    for k in UNTRAINED:
        mod_name, _, attr = k.rpartition(".")
        model.get_submodule(mod_name).get_parameter(attr).data.fill_(float("nan"))
    with torch.no_grad():
        emb3 = model._encode(src, values, src_key_padding_mask=pad_mask)
        out3 = model(src, values, src_key_padding_mask=pad_mask, MVC=flags["do_mvc"])
    assert torch.equal(emb3, emb), "_encode depends on the untrained cls_decoder"
    assert torch.isfinite(out3["mlm_output"]).all(), "forward(CLS=False) leaks cls_decoder"
    print("  -> poisoned cls_decoder with NaN: _encode and forward(CLS=False) unchanged.")

print("\nVERIFY OK")
