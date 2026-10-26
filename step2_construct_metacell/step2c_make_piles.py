#!/usr/bin/env python
"""Step 1 of 3 -- cut the cohort into locality-based piles for SEACells.

SEACells is kernel archetypal analysis: it holds an (n_cells x n_archetypes)
weight matrix and iterates Franke-Wolfe over it.  At 5.35 M cells that matrix is
not representable, so the cohort is partitioned first and SEACells is run once
per pile.  This is the Metacell-2 "divide and conquer" device, used here purely
for tractability -- the piles are scaffolding, not an output.

Piles must be LOCALITY-based, not random.  A random pile would scatter the cells
of any real metacell across all piles, so no pile could recover it.  Piles built
from k-means on the Harmony embedding keep neighbourhoods intact; the only cost
is that metacells cannot straddle a pile boundary, which the later Ward merge
partly repairs by re-joining base metacells across piles.

BALANCING.  Raw k-means piles are wildly uneven (dense T-cell states swallow
100k+ cells).  Uneven piles are a problem in both directions: an oversized pile
blows the archetype matrix, an undersized one cannot support enough archetypes.
So piles are post-processed -- oversized ones split by recursive 2-means,
undersized ones absorbed into the nearest surviving centroid that has room.

Output: piles.npy, int32 pile id per FILTERED-cohort row (0..n_piles-1).
"""
import argparse, json, os, time

import numpy as np
from sklearn.cluster import MiniBatchKMeans, KMeans


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


ap = argparse.ArgumentParser()
ap.add_argument("--emb", required=True)
ap.add_argument("--out-dir", required=True)
ap.add_argument("--target", type=int, default=15_000, help="cells per pile")
ap.add_argument("--max-frac", type=float, default=1.6, help="cap = max_frac*target")
ap.add_argument("--min-frac", type=float, default=0.25, help="floor = min_frac*target")
ap.add_argument("--seed", type=int, default=0)
A = ap.parse_args()

E = np.load(A.emb, mmap_mode="r")
N, D = E.shape
log(f"embedding {N:,} x {D} from {os.path.basename(A.emb)}")
E = np.ascontiguousarray(E, dtype=np.float32)

n_piles = int(np.ceil(N / A.target))
cap = int(A.max_frac * A.target)
floor_ = int(A.min_frac * A.target)
log(f"target {A.target:,}/pile -> {n_piles} piles | cap {cap:,} | floor {floor_:,}")

# run k-means to get a first set of piles
km = MiniBatchKMeans(n_clusters=n_piles, random_state=A.seed, batch_size=8192,
                     n_init=3, max_iter=200, reassignment_ratio=0.01, verbose=0)
lab = km.fit_predict(E).astype(np.int32)
cnt = np.bincount(lab, minlength=n_piles)
log(f"k-means: sizes min {cnt.min():,} median {int(np.median(cnt)):,} max {cnt.max():,}")

# split oversized piles by recursive 2-means as dense regions produce large piles
nxt = n_piles
work = [p for p in range(n_piles) if cnt[p] > cap]
rounds = 0
while work:
    rounds += 1
    new = []
    for p in work:
        rows = np.flatnonzero(lab == p)
        if len(rows) <= cap:
            continue
        k2 = KMeans(n_clusters=2, random_state=A.seed, n_init=3).fit_predict(E[rows])
        # the "1" half becomes a fresh pile; the "0" half keeps the id
        moved = rows[k2 == 1]
        if len(moved) == 0 or len(moved) == len(rows):      # degenerate split
            moved = rows[len(rows) // 2:]                   # fall back to a halving
        lab[moved] = nxt
        for q in (p, nxt):
            sz = int((lab == q).sum())
            if sz > cap:
                new.append(q)
        nxt += 1
    work = new
    if rounds > 40:
        raise RuntimeError("splitting did not converge")
log(f"after {rounds} split rounds: {nxt} piles, max {np.bincount(lab).max():,}")

# absorb undersized piles
lab = np.unique(lab, return_inverse=True)[1].astype(np.int32)
P = int(lab.max()) + 1
cnt = np.bincount(lab, minlength=P)
cent = np.zeros((P, D), np.float32)
for p in range(P):
    cent[p] = E[lab == p].mean(axis=0)

small = np.flatnonzero(cnt < floor_)
log(f"{len(small)} piles under the floor, absorbing")
for p in small[np.argsort(cnt[small])]:
    if cnt[p] == 0 or cnt[p] >= floor_:
        continue
    d = ((cent - cent[p]) ** 2).sum(axis=1)
    d[p] = np.inf
    d[cnt == 0] = np.inf
    d[cnt + cnt[p] > cap] = np.inf                 # do not create an oversized pile
    if not np.isfinite(d).any():
        continue                                   # nowhere to put it; leave as is
    q = int(np.argmin(d))
    rows = lab == p
    cent[q] = (cent[q] * cnt[q] + cent[p] * cnt[p]) / (cnt[q] + cnt[p])
    lab[rows] = q
    cnt[q] += cnt[p]
    cnt[p] = 0

lab = np.unique(lab, return_inverse=True)[1].astype(np.int32)
P = int(lab.max()) + 1
cnt = np.bincount(lab, minlength=P)
assert cnt.min() > 0 and cnt.sum() == N

log(f"FINAL {P} piles | min {cnt.min():,} median {int(np.median(cnt)):,} "
    f"mean {cnt.mean():.0f} max {cnt.max():,}")

os.makedirs(A.out_dir, exist_ok=True)
np.save(os.path.join(A.out_dir, "piles.npy"), lab)
np.save(os.path.join(A.out_dir, "pile_sizes.npy"), cnt.astype(np.int64))
json.dump({"n_cells": int(N), "n_dims": int(D), "n_piles": int(P),
           "target": A.target, "cap": cap, "floor": floor_, "seed": A.seed,
           "emb": os.path.abspath(A.emb),
           "size_min": int(cnt.min()), "size_max": int(cnt.max()),
           "size_median": int(np.median(cnt)), "size_mean": float(cnt.mean())},
          open(os.path.join(A.out_dir, "piles.json"), "w"), indent=2)
log("DONE")
