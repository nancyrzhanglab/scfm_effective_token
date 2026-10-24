#!/usr/bin/env python
"""Build the scGPT input for the CD8 set.

Cells left with NO expressed usable gene cannot be tokenised.  They are dropped
and their positions recorded in *_scgpt_rowidx.npy, so the embedding rows can
always be mapped back to h5ad row order rather than silently misaligning.
"""
import argparse, json, os, time
import h5py, numpy as np, pandas as pd
from scipy.sparse import csr_matrix, save_npz

ap = argparse.ArgumentParser()
ap.add_argument("--h5ad", required=True)
ap.add_argument("--var-csv", required=True)
ap.add_argument("--vocab", required=True)
ap.add_argument("--out-dir", required=True)
ap.add_argument("--prefix", default="allcd8")
ap.add_argument("--block", type=int, default=250_000)
A = ap.parse_args()
t0 = time.time(); log = lambda m: print(f"[{time.time()-t0:8.1f}s] {m}", flush=True)

# read data
vocab = json.load(open(A.vocab))
var = pd.read_csv(A.var_csv)

# map gene symbol and token
sym = var.feature_name.astype(str).to_numpy()
inv = np.array([s in vocab for s in sym])
first = ~pd.Series(sym).duplicated().to_numpy()      # a symbol maps to one token
usable = inv & first
cols = np.flatnonzero(usable)
syms = sym[usable]
assert len(set(syms)) == len(syms)
log(f"{len(cols)} of {len(var)} genes usable by scGPT ({usable.mean()*100:.1f}%); "
    f"{int((~inv).sum())} absent from vocab, {int((inv & ~first).sum())} duplicate symbols")
pd.DataFrame(dict(ensembl=var.feature_id, symbol=sym, in_vocab=inv,
                  usable=usable)).to_csv(
    os.path.join(A.out_dir, f"{A.prefix}_gene_symbol_map.csv"), index=False)

keep = np.zeros(var.shape[0], bool); keep[cols] = True
newcol = np.full(var.shape[0], -1, np.int32); newcol[cols] = np.arange(len(cols))

f = h5py.File(A.h5ad, "r")
Xp, Xi, Xd = f["X/indptr"], f["X/indices"], f["X/data"]
N = Xp.shape[0] - 1; G = int(f["X"].attrs["shape"][1])
assert G == var.shape[0], f"var.csv has {var.shape[0]} rows, X has {G} columns"
indptr = Xp[:]
log(f"{N:,} cells x {G:,} genes; scanning")

# ---- pass 1: nonzeros per cell within the usable columns ----
cnt = np.zeros(N, np.int32)
for lo in range(0, N, A.block):
    hi = min(lo + A.block, N); a, b = int(indptr[lo]), int(indptr[hi])
    j = Xi[a:b]
    ok = keep[j]
    cnt[lo:hi] = np.add.reduceat(ok.astype(np.int32),
                                 (indptr[lo:hi] - indptr[lo]).astype(np.int64)) \
                 if hi > lo else 0
    # reduceat mis-handles empty rows: fix them explicitly
    empty = np.flatnonzero(np.diff(indptr[lo:hi + 1]) == 0)
    if len(empty): cnt[lo + empty] = 0
    if (lo // A.block) % 5 == 0: log(f"  pass1 {hi:,}/{N:,}")
good = cnt > 0
ndrop = int((~good).sum())
log(f"cells with no expressed usable gene: {ndrop:,} ({ndrop/N*100:.3f}%) -> dropped")
rows = np.flatnonzero(good)
nnz_tot = int(cnt[good].sum())
log(f"keeping {len(rows):,} cells, {nnz_tot:,} nonzeros")

# ---- pass 2: fill preallocated CSR ----
data = np.empty(nnz_tot, np.float32); ind = np.empty(nnz_tot, np.int32)
newp = np.zeros(len(rows) + 1, np.int64); np.cumsum(cnt[good], out=newp[1:])
pos = 0; r = 0
for lo in range(0, N, A.block):
    hi = min(lo + A.block, N); a, b = int(indptr[lo]), int(indptr[hi])
    j = Xi[a:b]; d = Xd[a:b]; ok = keep[j]
    ip = (indptr[lo:hi + 1] - indptr[lo]).astype(np.int64)
    for i in range(hi - lo):
        if cnt[lo + i] == 0: continue
        sl = slice(ip[i], ip[i + 1]); m = ok[sl]
        k = int(m.sum())
        ind[pos:pos + k] = newcol[j[sl][m]]
        data[pos:pos + k] = d[sl][m]
        pos += k; r += 1
    if (lo // A.block) % 5 == 0: log(f"  pass2 {hi:,}/{N:,}")
f.close()
assert pos == nnz_tot and r == len(rows), (pos, nnz_tot, r, len(rows))
X = csr_matrix((data, ind, newp), shape=(len(rows), len(cols)))
nz = np.diff(X.indptr)
assert nz.min() > 0
log(f"counts {X.shape}, {X.nnz:,} nonzeros; usable genes/cell "
    f"median {np.median(nz):.0f} p95 {np.percentile(nz,95):.0f} max {nz.max()}")

save_npz(os.path.join(A.out_dir, f"{A.prefix}_counts_scgpt.npz"), X)
np.save(os.path.join(A.out_dir, f"{A.prefix}_scgpt_rowidx.npy"), rows)
with open(os.path.join(A.out_dir, f"{A.prefix}_genes_scgpt.txt"), "w") as fh:
    fh.write("\n".join(syms) + "\n")
json.dump(dict(n_cells_h5ad=int(N), n_cells_kept=int(len(rows)), n_dropped=ndrop,
               n_genes_in=int(G), n_genes_used=int(len(cols)),
               frac_used=float(usable.mean()), nnz=int(X.nnz),
               median_expressed=float(np.median(nz)), max_expressed=int(nz.max())),
          open(os.path.join(A.out_dir, f"{A.prefix}_scgpt_input.json"), "w"), indent=2)
log("PREP SCGPT OK")
