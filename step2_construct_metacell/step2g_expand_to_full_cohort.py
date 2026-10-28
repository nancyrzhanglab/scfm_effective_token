#!/usr/bin/env python
"""Index SEACells+Ward and MetaCell-1+Ward labels from the 5.35M cohort to the 5.38M cohort since that stores the full data downloaded

step2f writes labels only for the cells it was given, in the filtered-row order
saved as <prefix>_rowidx_filtered.npy.  For SEACells that is every filtered cell;
for MetaCell-1 it leaves out the outliers MetaCell does not cover.  Two maps
compose:

    ladder position -> filtered row  (--ladder-rows, <prefix>_rowidx_filtered.npy)
    filtered row    -> full row      (--rowidx, allcd8_filtered_rowidx.npy)

Cells with no label -- the 28,014 non-UMI cells and any MetaCell outliers -- get
-1.  step3a_cv_ltv_vs_k.py drops those from BOTH the pooled terms and the
totals, so the LTV closure still holds, and the support is the same at every
rung, as it requires.
"""
import argparse, json, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--labels", required=True, help="<prefix>_<tag>.npz from step2f")
ap.add_argument("--ladder-rows", required=True, help="<prefix>_rowidx_filtered.npy from step2f")
ap.add_argument("--rowidx", required=True, help="filtered -> full row map")
ap.add_argument("--n-full", type=int, required=True)
ap.add_argument("--out", required=True)
A = ap.parse_args()

idx = np.load(A.rowidx)
pos = np.load(A.ladder_rows)
assert idx.ndim == 1 and np.all(np.diff(idx) > 0), "row map must be sorted, unique"
assert idx.max() < A.n_full
assert np.all(np.diff(pos) > 0) and pos.max() < len(idx), "ladder rows must be sorted filtered rows"
dest = idx[pos]
Z = np.load(A.labels)

out, rungs = {}, []
for key in sorted(Z.files):
    lab = Z[key]
    assert lab.shape == pos.shape, f"{key}: {lab.shape} vs ladder rows {pos.shape}"
    full = np.full(A.n_full, -1, np.int64)
    full[dest] = lab
    assert (full[dest] == lab).all()
    assert int((full >= 0).sum()) == len(pos)
    out[key] = full.astype(np.int32)
    rungs.append(dict(key=key, n_metacells=int(lab.max()) + 1))
    print(f"{key}: {lab.max()+1:,} pools, {len(pos):,} of {A.n_full:,} cells labelled",
          flush=True)
print(f"unlabelled: {A.n_full - len(idx):,} outside the filtered cohort + "
      f"{len(idx) - len(pos):,} filtered cells without a metacell", flush=True)

os.makedirs(os.path.dirname(A.out), exist_ok=True)
np.savez_compressed(A.out, **out)
json.dump(dict(labels=A.labels, n_full=A.n_full, n_filtered=int(len(idx)),
               n_labelled=int(len(pos)), n_unlabelled_filtered=int(len(idx) - len(pos)),
               rungs=rungs),
          open(os.path.splitext(A.out)[0] + ".json", "w"), indent=2)
print(f"wrote {A.out}", flush=True)
