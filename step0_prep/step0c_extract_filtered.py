#!/usr/bin/env python
"""Write a row subset of an h5ad given a precomputed row index.

Two-pass preallocation rather than vstack(blocks): at 5.35M cells the block list
plus the final matrix would hold the data twice (~5 GB each).  Pass 1 counts
nonzeros per kept row, pass 2 fills preallocated arrays, so peak is one copy.
"""
import argparse, os, time
import h5py, numpy as np, pandas as pd, anndata as ad
from scipy.sparse import csr_matrix

ap = argparse.ArgumentParser()
ap.add_argument("--h5ad", required=True)
ap.add_argument("--rowidx", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--block", type=int, default=500_000)
a = ap.parse_args()
t0 = time.time(); log = lambda m: print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)

rows = np.sort(np.load(a.rowidx).astype(np.int64))
with h5py.File(a.h5ad, "r") as f:
    N, G = (int(x) for x in f["X"].attrs["shape"])
    ip = f["X"]["indptr"][:]
    assert rows.max() < N and rows.min() >= 0
    log(f"{N:,} cells -> keeping {len(rows):,} ({len(rows)/N*100:.2f}%)")
    keep = np.zeros(N, bool); keep[rows] = True
    cnt = np.diff(ip)[rows]                       # nnz per kept row
    nnz = int(cnt.sum())
    data = np.empty(nnz, np.float32); ind = np.empty(nnz, np.int32)
    newp = np.zeros(len(rows) + 1, np.int64); np.cumsum(cnt, out=newp[1:])
    pos = 0; r = 0
    for lo in range(0, N, a.block):
        hi = min(lo + a.block, N)
        sel = np.flatnonzero(keep[lo:hi])
        if not len(sel):
            continue
        lo_b, hi_b = int(ip[lo]), int(ip[hi])
        d = f["X"]["data"][lo_b:hi_b]; j = f["X"]["indices"][lo_b:hi_b]
        off = (ip[lo:hi+1] - ip[lo]).astype(np.int64)
        for i in sel:
            s = slice(off[i], off[i+1]); k = s.stop - s.start
            data[pos:pos+k] = d[s]; ind[pos:pos+k] = j[s]
            pos += k; r += 1
        log(f"  {hi:,}/{N:,}")
    assert pos == nnz and r == len(rows), (pos, nnz, r, len(rows))
X = csr_matrix((data, ind, newp), shape=(len(rows), G))
assert np.all(X.data == np.floor(X.data)), "not integer counts"

A = ad.read_h5ad(a.h5ad, backed="r")
obs = A.obs.iloc[rows].copy(); var = A.var.copy(); A.file.close()
out = ad.AnnData(X=X, obs=obs, var=var)
out.write_h5ad(a.out)
log(f"wrote {a.out}  {out.shape}  nnz={X.nnz:,}  {os.path.getsize(a.out)/1e9:.2f} GB")
log(f"datasets {obs.dataset_id.nunique()}  assays {obs.assay.nunique()}  "
    f"cell types {obs.cell_type.nunique()}  donors {obs.donor_id.nunique()}")
log("EXTRACT OK")
