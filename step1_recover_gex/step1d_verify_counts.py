#!/usr/bin/env python
"""Independent whole-matrix check on the CellANOVA count artefact.

Deliberately NOT the per-shard self-check: this runs in a separate process
after every task has exited, streams the file with ordinary reads, and covers
all N rows -- the failure mode it exists to catch (job 1423187) passed the
per-shard check on two tasks while the file on disk was empty.
"""
import argparse, json, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--counts", required=True)
ap.add_argument("--obs-tot", required=True)
ap.add_argument("--chunk", type=int, default=20_000)
ap.add_argument("--tol", type=float, default=1e-4)
a = ap.parse_args()
t0 = time.time(); log = lambda m: print(f"[{time.time()-t0:8.1f}s] {m}", flush=True)

obs = np.load(a.obs_tot).astype(np.float64)
R = np.load(a.counts, mmap_mode="r")
N, G = R.shape
assert len(obs) == N, f"obs_tot {len(obs):,} != {N:,} rows"
log(f"{N:,} x {G} ({N*G*4/1e9:.1f} GB)")

nzero = 0; worst = 0.0; worst_row = -1; nbad = 0; nneg = 0
for lo in range(0, N, a.chunk):
    hi = min(lo + a.chunk, N)
    blk = np.asarray(R[lo:hi], np.float64)
    tot = blk.sum(1)
    nneg += int((blk < 0).sum())
    nzero += int((tot <= 0).sum())
    rel = np.abs(tot - obs[lo:hi]) / np.maximum(obs[lo:hi], 1e-12)
    nbad += int((rel >= a.tol).sum())
    i = int(np.nanargmax(rel))
    if rel[i] > worst: worst, worst_row = float(rel[i]), lo + i
    del blk, tot, rel
    if (lo // a.chunk) % 50 == 0: log(f"  {hi:,}/{N:,}")

res = dict(n=N, n_genes=G, all_zero_rows=nzero, rows_over_tol=nbad,
           negative_entries=nneg, max_rel_err=worst, max_rel_err_row=worst_row,
           tol=a.tol, ok=bool(nzero == 0 and nbad == 0 and nneg == 0))
log(json.dumps(res, indent=1))
json.dump(res, open(a.counts + ".verify.json", "w"), indent=1)
if not res["ok"]:
    raise SystemExit(f"VERIFY FAILED: {nzero:,} all-zero rows, {nbad:,} rows over tol, "
                     f"{nneg:,} negative entries, max rel err {worst:.3g}")
log("VERIFY OK")
