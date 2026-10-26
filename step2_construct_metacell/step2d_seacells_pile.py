#!/usr/bin/env python
"""Step 2 of 3 -- run SEACells on one pile to build the BASE rung (k=64).

n_SEACells = round(n_pile / target_size), so the MEAN metacell size is the target
by construction; the realised size distribution is recorded and is not uniform.

NO COUNT MATRIX IS LOADED.  The archetypal fit reads only ad.obsm; ad.X is
touched solely by summarize_by_SEACell(), which is not called.  X is therefore a
zero placeholder and a pile costs a few hundred MB, not tens of GB.
"""
import argparse, json, os, sys, time, types

import numpy as np

# SEACells/__init__ eagerly imports its ATAC modules (genescores, accessibility,
# tfactivity), which pull pyranges and friends.  None of that is on the RNA path.
# Stubbing the names is cheaper and less fragile than installing a genomics stack
# we never call into.
for _m in ("pyranges", "pyBigWig", "genomepy", "tangermeme", "pyjaspar", "cython"):
    sys.modules.setdefault(_m, types.ModuleType(_m))

import anndata as ad_mod
import SEACells


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


ap = argparse.ArgumentParser()
ap.add_argument("--emb", required=True)
ap.add_argument("--piles", required=True)
ap.add_argument("--pile", type=int, required=True)
ap.add_argument("--out-dir", required=True)
ap.add_argument("--target-size", type=int, default=16)
ap.add_argument("--n-neighbors", type=int, default=15)
ap.add_argument("--n-waypoint-eigs", type=int, default=10)
ap.add_argument("--max-iter", type=int, default=100)
ap.add_argument("--min-iter", type=int, default=10)
ap.add_argument("--epsilon", type=float, default=1e-3)
ap.add_argument("--seed", type=int, default=0,
                help="seeds numpy's global RNG, which SEACells draws from unseeded")
ap.add_argument("--force", action="store_true")
A = ap.parse_args()

os.makedirs(A.out_dir, exist_ok=True)
out = os.path.join(A.out_dir, f"pile_{A.pile:05d}.npz")
if os.path.exists(out) and not A.force:
    log(f"{os.path.basename(out)} exists, skipping (--force to redo)")
    sys.exit(0)

# read data
piles = np.load(A.piles)
rows = np.flatnonzero(piles == A.pile)
n = len(rows)
assert n > 0, f"pile {A.pile} is empty"

E = np.load(A.emb, mmap_mode="r")
X = np.ascontiguousarray(E[rows], dtype=np.float32)
D = X.shape[1]

n_mc = max(2, int(round(n / A.target_size)))
assert n_mc < n, f"pile {A.pile}: {n_mc} archetypes for {n} cells"
log(f"pile {A.pile}: {n:,} cells x {D} dims -> {n_mc:,} SEACells "
    f"(target {A.target_size} cells each)")

adata = ad_mod.AnnData(X=np.zeros((n, 1), dtype=np.float32))
adata.obs_names = rows.astype(str)
adata.obsm["X_harmony"] = X # here "X_harmony" is just a label name, not necessarily only holds Harmony embedding, and can be others

t0 = time.time()
model = SEACells.core.SEACells(
    adata,
    build_kernel_on="X_harmony",
    n_SEACells=n_mc,
    n_neighbors=A.n_neighbors,
    n_waypoint_eigs=A.n_waypoint_eigs,
    convergence_epsilon=A.epsilon,
    verbose=False,
)
model.construct_kernel_matrix()
log(f"kernel built ({time.time()-t0:.0f}s)")
model.initialize_archetypes()
log(f"archetypes initialised ({time.time()-t0:.0f}s)")
model.fit(min_iter=A.min_iter, max_iter=A.max_iter)
log(f"fit done ({time.time()-t0:.0f}s), {len(model.RSS_iters)} iterations")

# hard assignment: each cell to the archetype carrying its largest weight
assign = adata.obs["SEACell"].astype(str).to_numpy()
uniq, loc = np.unique(assign, return_inverse=True)
loc = loc.astype(np.int32)
K = len(uniq)
sz = np.bincount(loc, minlength=K)
assert sz.min() > 0

# compute metacell centroid and use in step2g Ward merge to repeatly merge base metacells
cent = np.zeros((K, D), np.float64)
np.add.at(cent, loc, X.astype(np.float64))
cent /= sz[:, None]

log(f"{K:,} non-empty metacells | size min {sz.min()} median "
    f"{int(np.median(sz))} mean {sz.mean():.1f} max {sz.max()}")

np.savez_compressed(
    out,
    rows=rows.astype(np.int64),          # FILTERED-cohort row indices
    local=loc,                           # 0..K-1, index into cent/sizes
    centroids=cent.astype(np.float32),
    sizes=sz.astype(np.int32),
    meta=np.array(json.dumps({
        "pile": A.pile, "n_cells": int(n), "n_requested": int(n_mc),
        "n_realised": int(K), "target_size": A.target_size,
        "n_neighbors": A.n_neighbors, "seed": A.seed + A.pile,
        "iters": int(len(model.RSS_iters)),
        "seconds": round(time.time() - t0, 1),
    })),
)
log(f"wrote {out}")
log("DONE")
