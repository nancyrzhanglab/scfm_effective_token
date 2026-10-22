#!/usr/bin/env python
"""Combine per-pile MetaCell-1 partitions into cohort-wide labels.

Concatenation only -- no merging across piles.  MetaCell is a FLAT method: there
is no tree to cut, so unlike task10's Ward step this produces exactly one
resolution, the one the per-pile runs were given.  Pile boundaries are therefore
permanent here, which is the price of using piles for tractability.

Emits labels on BOTH indexings: the filtered cohort the embedding uses, and the
full 5,381,464-row cohort the reconstruction uses (-1 outside), so cv_ltv_vs_k_task10.py
can consume it unchanged.
"""
import argparse, glob, json, os, re

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--pile-dir", required=True)
ap.add_argument("--piles", required=True)
ap.add_argument("--rowidx", required=True)
ap.add_argument("--n-full", type=int, default=5_381_464)
ap.add_argument("--K", type=int, default=100)
ap.add_argument("--out-prefix", required=True)
ap.add_argument("--tag", default="metacell1")
A = ap.parse_args()

fs = sorted(glob.glob(os.path.join(A.pile_dir, f"mc1_pile_*_K{A.K}.npz")))
n_exp = int(np.load(A.piles).max()) + 1
assert len(fs) == n_exp, f"{len(fs)} outputs but {n_exp} piles -- array incomplete"

rows_l, lab_l, cent_l, size_l, pile_l, secs = [], [], [], [], [], []
off = 0
for f in fs:
    z = np.load(f, allow_pickle=False)
    m = json.loads(str(z["meta"]))
    rows_l.append(z["rows"]); lab_l.append(z["local"].astype(np.int64) + off)
    cent_l.append(z["centroids"]); size_l.append(z["sizes"])
    pile_l.append(np.full(len(z["sizes"]), m["pile"], np.int32))
    secs.append(m["seconds"]); off += len(z["sizes"])

rows = np.concatenate(rows_l); lab = np.concatenate(lab_l)
CENT = np.concatenate(cent_l); SIZE = np.concatenate(size_l)
PILE = np.concatenate(pile_l)
o = np.argsort(rows, kind="stable"); rows, lab = rows[o], lab[o]
assert np.all(np.diff(rows) > 0), "pile row sets overlap"
N, M = len(rows), len(SIZE)
assert SIZE.sum() == N

print(f"{M:,} metacells over {N:,} cells (mean {N/M:.1f}); "
      f"sizes min {SIZE.min()} p10 {np.percentile(SIZE,10):.0f} "
      f"median {np.median(SIZE):.0f} p90 {np.percentile(SIZE,90):.0f} max {SIZE.max()}")
print(f"per-pile runtime: median {np.median(secs)/60:.1f} min, "
      f"total {np.sum(secs)/3600:.1f} CPU-h")

idx = np.load(A.rowidx)
assert len(idx) == N and np.all(np.diff(idx) > 0)
full = np.full(A.n_full, -1, np.int64); full[idx] = lab
assert int((full >= 0).sum()) == N

key = f"lab_mean{A.K}_{A.tag}"
os.makedirs(os.path.dirname(A.out_prefix), exist_ok=True)
np.savez_compressed(A.out_prefix + f"_{A.tag}.npz", **{key: lab.astype(np.int32)})
np.savez_compressed(A.out_prefix + f"_{A.tag}_fullcohort.npz",
                    **{key: full.astype(np.int32)})
np.save(A.out_prefix + f"_{A.tag}_centroids.npy", CENT)
np.save(A.out_prefix + f"_{A.tag}_sizes.npy", SIZE.astype(np.int32))
np.save(A.out_prefix + f"_{A.tag}_pile.npy", PILE)
json.dump({"method": "MetaCell-1 (Baran 2019) on Harmony embedding, per pile",
           "K": A.K, "n_piles": len(fs), "n_metacells": int(M), "n_cells": int(N),
           "mean_size": float(N/M), "size_min": int(SIZE.min()),
           "size_median": float(np.median(SIZE)), "size_max": int(SIZE.max()),
           "cpu_hours": float(np.sum(secs)/3600), "key": key,
           "nested": False,
           "note": "flat partition; pile boundaries are permanent"},
          open(A.out_prefix + f"_{A.tag}.json", "w"), indent=2)
print(f"wrote {A.out_prefix}_{A.tag}.npz and _fullcohort.npz  (key {key})")
