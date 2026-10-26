#!/usr/bin/env python
"""Per-gene totals of the observed counts, over the whole cohort.

step3's --gene-obs: the denominator of its calibration line, which reports
sum(reconstructed)/sum(observed) per gene. 

    out[g] = sum over all cells of X[c, g]      X = raw counts, the h5ad's X

The result is float64 and its total equals sum(observed_hvg_libsize.npy)
exactly, which --check asserts: both are the same sum taken along different
axes, so any mismatch means one of them is stale.
"""
import argparse, json, os, time
import h5py
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--h5ad", required=True, help="CD8T_all_hvg.h5ad (raw counts in X)")
ap.add_argument("--out", required=True, help="observed_hvg_genesum.npy")
ap.add_argument("--check", default="",
                help="observed_hvg_libsize.npy; its total must match")
ap.add_argument("--block", type=int, default=50_000_000,
                help="nonzeros per read (50M ~ 400 MB of float32 + int32)")
A = ap.parse_args()
t0 = time.time()
log = lambda m: print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)

with h5py.File(A.h5ad, "r") as f:
    X = f["X"]
    enc = X.attrs.get("encoding-type", "")
    assert enc == "csr_matrix", f"X is {enc!r}, expected csr_matrix"
    n_cells, n_genes = (int(v) for v in X.attrs["shape"])
    nnz = X["data"].shape[0]
    log(f"{n_cells:,} cells x {n_genes:,} genes, {nnz:,} nonzeros")

    out = np.zeros(n_genes, np.float64)
    for lo in range(0, nnz, A.block):
        hi = min(lo + A.block, nnz)
        d = np.asarray(X["data"][lo:hi], np.float64)
        i = np.asarray(X["indices"][lo:hi], np.int64)
        out += np.bincount(i, weights=d, minlength=n_genes) # sum each gene's total count within the block
        log(f"  {hi:,}/{nnz:,}")

np.save(A.out, out)
log(f"wrote {A.out}  total {out.sum():.10e}  "
    f"min {out.min():.0f}  max {out.max():.4g}  zeros {(out == 0).sum()}")

if A.check:
    L = np.load(A.check)
    assert L.size == n_cells, f"{A.check} has {L.size} rows, X has {n_cells}"
    rel = abs(out.sum() - L.sum()) / L.sum()
    log(f"check vs {os.path.basename(A.check)}: "
        f"{out.sum():.10e} vs {L.sum():.10e}  rel {rel:.3e}")
    assert rel < 1e-12, "gene totals and cell totals disagree"
    log("CHECK OK")

json.dump(dict(h5ad=A.h5ad, n_cells=n_cells, n_genes=n_genes, nnz=int(nnz),
               total=float(out.sum()), n_zero_genes=int((out == 0).sum())),
          open(os.path.splitext(A.out)[0] + ".json", "w"), indent=1)
