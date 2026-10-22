#!/usr/bin/env python
"""Lift the task10 SEACells+Ward labels from the filtered cohort to the full one.

task10 was built on allcd8f_harmony_dataset_id.npy -- the FILTERED cohort of
5,353,450 cells (94 datasets) -- while the reconstruction and the cv/ltv machinery
are indexed by the full 5,381,464-row h5ad.  allcd8_filtered_rowidx.npy is the map
between them.

Cells outside the filtered set get label -1.  cv_ltv_vs_k_task10.py drops those
from BOTH the pooled terms and the totals, so the LTV closure still holds; the
unpatched task8 script would not (its Wtot runs over all N cells) and would trip
its own closure assertion.
"""
import argparse, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--labels", required=True)
ap.add_argument("--rowidx", required=True, help="filtered -> full row map")
ap.add_argument("--n-full", type=int, required=True)
ap.add_argument("--out", required=True)
A = ap.parse_args()

idx = np.load(A.rowidx)
Z = np.load(A.labels)
assert idx.ndim == 1 and np.all(np.diff(idx) > 0), "row map must be sorted, unique"
assert idx.max() < A.n_full

out = {}
for key in sorted(Z.files):
    lab = Z[key]
    assert lab.shape == idx.shape, f"{key}: {lab.shape} vs row map {idx.shape}"
    full = np.full(A.n_full, -1, np.int64)
    full[idx] = lab
    assert (full[idx] == lab).all()
    assert int((full >= 0).sum()) == len(idx)
    out[key] = full.astype(np.int32)
    print(f"{key}: {lab.max()+1:,} pools, {len(idx):,} of {A.n_full:,} cells labelled",
          flush=True)

os.makedirs(os.path.dirname(A.out), exist_ok=True)
np.savez_compressed(A.out, **out)
print(f"wrote {A.out}", flush=True)
