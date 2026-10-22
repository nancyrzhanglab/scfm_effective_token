#!/usr/bin/env python
"""Build the per-cell clip-group code vector for CellANOVA count conversion.

The ceiling table is computed per (group, gene); groups here are cell_type
levels, so a cell's reconstructed log-rate is clipped at the max observed for
its own cell type rather than a global maximum.  Idempotent: if the output
already exists with the right length it is left alone.
"""
import argparse, os
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--obs", required=True, help="obs csv.gz")
ap.add_argument("--col", default="cell_type")
ap.add_argument("--out", required=True)
ap.add_argument("--expect-n", type=int, default=0, help="assert this many rows (0=skip)")
a = ap.parse_args()

if os.path.exists(a.out):
    prev = np.load(a.out)
    if a.expect_n and prev.shape[0] != a.expect_n:
        raise SystemExit(f"existing {a.out} has {prev.shape[0]:,} rows, expected {a.expect_n:,}")
    print(f"exists: {a.out} ({prev.shape[0]:,} rows, {prev.max()+1} levels) — keeping", flush=True)
    raise SystemExit(0)

o = pd.read_csv(a.obs, usecols=[a.col], low_memory=False)
c = pd.Categorical(o[a.col])
if a.expect_n and len(c) != a.expect_n:
    raise SystemExit(f"{a.obs} has {len(c):,} rows, expected {a.expect_n:,}")
assert (c.codes >= 0).all(), f"{(c.codes < 0).sum()} NaN values in {a.col}"

tmp = a.out + f".tmp{os.getpid()}.npy"
np.save(tmp, c.codes.astype(np.int32))
os.replace(tmp, a.out)

b = np.bincount(c.codes)
print(f"clip groups from {a.col}: {len(c.categories)} levels, "
      f"sizes {b.min():,}-{b.max():,}, total {b.sum():,}", flush=True)
print("largest:", ", ".join(f"{c.categories[i]}={b[i]:,}"
                            for i in np.argsort(b)[::-1][:5]), flush=True)
