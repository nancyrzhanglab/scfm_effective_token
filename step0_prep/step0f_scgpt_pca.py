#!/usr/bin/env python
"""PCA of the scGPT embedding over ALL 5,381,464 CD8 T cells.

Two streaming passes over the 216 shards, so the 11 GB embedding is never held
whole:
  pass 1  accumulate sum and X^T X  ->  mean and covariance (512 x 512)
  pass 2  centre and project onto the top d eigenvectors

Only the (N x d) output is materialised: 5,381,464 x 50 float32 = 1.08 GB.

Covariance PCA, centred, unscaled -- the same basis scgpt_pca_scree.py used to
pick d, and what a euclidean-metric consumer downstream sees.

WHY THIS FEEDS CellANOVA.  fit_M solves, per integration level x,
    B_x = pinv(C_x^T C_x) @ (C_x^T S_x),   Mmat = mean_x B_x
so C enters only through its column space.  Over the FULL 512 components PCA is
an orthogonal reparametrisation and Mmat would be unchanged; truncating to d < 512
is a projection and genuinely changes the fit -- that truncation is the point of
the exercise, not an incidental detail.

Eigenvector signs are fixed (largest-|value| entry made positive) so reruns are
bit-comparable.
"""
import argparse, glob, json, os, re, time

import numpy as np

t0 = time.time()
def log(m):
    print(f"[{time.time()-t0:8.1f}s] {m}", flush=True)

ap = argparse.ArgumentParser()
ap.add_argument("--emb-dir", required=True)
ap.add_argument("--out-dir", required=True)
ap.add_argument("--prefix", default="allcd8_scgpt")
ap.add_argument("--n-pcs", type=int, default=50)
ap.add_argument("--chunk", type=int, default=50_000)
ap.add_argument("--expect-n", type=int, default=0)
A = ap.parse_args()

fs = sorted(glob.glob(os.path.join(A.emb_dir, "emb_*.npy")))
assert fs, f"no shards in {A.emb_dir}"
spans, exp = [], 0
for f in fs:
    lo, hi = map(int, re.match(r"emb_(\d+)_(\d+)\.npy$", os.path.basename(f)).groups())
    assert lo == exp, f"shards not contiguous at {os.path.basename(f)}"
    with open(f, "rb") as fh:
        v = np.lib.format.read_magic(fh)
        sh, _, dt = np.lib.format._read_array_header(fh, v)
    assert sh[0] == hi - lo, f"{os.path.basename(f)}: header {sh[0]} vs name {hi-lo}"
    spans.append((lo, hi, f))
    exp = hi
N, D = exp, sh[1]
if A.expect_n:
    assert N == A.expect_n, f"shards cover {N:,}, expected {A.expect_n:,}"
log(f"{len(fs)} shards tile [0,{N:,}), d={D}, dtype {dt}")

# --- pass 1: mean and covariance ------------------------------------------------
S = np.zeros(D, np.float64)
G = np.zeros((D, D), np.float64)
seen = 0
for lo, hi, f in spans:
    M = np.load(f, mmap_mode="r")
    for c in range(0, hi - lo, A.chunk):
        B = np.asarray(M[c:c + A.chunk], dtype=np.float64)
        S += B.sum(0)
        G += B.T @ B
        seen += len(B)
    if (lo // 400_000) != ((lo - 1) // 400_000):
        log(f"  pass1 {hi:,}/{N:,}")
assert seen == N, f"pass 1 saw {seen:,} of {N:,}"
mu = S / N
COV = (G - N * np.outer(mu, mu)) / (N - 1)
COV = (COV + COV.T) / 2.0
log("pass 1 done; covariance accumulated over all cells")

w, V = np.linalg.eigh(COV)
w = w[::-1]
V = V[:, ::-1]
w = np.clip(w, 0, None)
for j in range(V.shape[1]):                      # deterministic sign
    k = np.argmax(np.abs(V[:, j]))
    if V[k, j] < 0:
        V[:, j] *= -1
d = A.n_pcs
W = V[:, :d].copy()
evr = w / w.sum()
log(f"PC1 {evr[0]*100:.2f}%  |  {d} PCs retain {evr[:d].sum()*100:.2f}%  |  "
    f"90% at {int(np.searchsorted(np.cumsum(evr),0.90)+1)} PCs")

# --- pass 2: project ------------------------------------------------------------
P = np.empty((N, d), np.float32)
for lo, hi, f in spans:
    M = np.load(f, mmap_mode="r")
    for c in range(0, hi - lo, A.chunk):
        B = np.asarray(M[c:c + A.chunk], dtype=np.float64)
        P[lo + c: lo + c + len(B)] = ((B - mu) @ W).astype(np.float32)
    if (lo // 400_000) != ((lo - 1) // 400_000):
        log(f"  pass2 {hi:,}/{N:,}")
log(f"projected -> {P.shape} ({P.nbytes/1e9:.2f} GB)")

# empirical check: projected variances must match the leading eigenvalues
emp = P.var(axis=0, ddof=1, dtype=np.float64)
rel = np.abs(emp - w[:d]) / np.maximum(w[:d], 1e-300)
log(f"variance check: max rel dev {rel.max():.3e} (expect ~1e-6)")
assert rel.max() < 1e-4, "projected variances disagree with the eigenvalues"

os.makedirs(A.out_dir, exist_ok=True)
O = os.path.join(A.out_dir, A.prefix)
np.save(f"{O}_pca{d}.npy", P)
np.save(f"{O}_pca{d}_loadings.npy", W.astype(np.float32))
np.save(f"{O}_pca_mean.npy", mu.astype(np.float32))
np.save(f"{O}_pca_eigenvalues.npy", w)
json.dump({"n_cells": int(N), "d_in": int(D), "n_pcs": int(d),
           "var_retained": float(evr[:d].sum()), "pc1_frac": float(evr[0]),
           "k90": int(np.searchsorted(np.cumsum(evr), 0.90) + 1),
           "k95": int(np.searchsorted(np.cumsum(evr), 0.95) + 1),
           "emb_dir": os.path.abspath(A.emb_dir),
           "centred": True, "scaled": False, "sign_convention": "max-|v| positive"},
          open(f"{O}_pca{d}.json", "w"), indent=2)
log(f"wrote {O}_pca{d}.npy and companions")
log("PCA OK")
