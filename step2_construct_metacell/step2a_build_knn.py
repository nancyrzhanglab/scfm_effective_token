#!/usr/bin/env python
"""Approximate kNN on the Harmony embedding, for the metacell ladder.

Building the exact kNN is O(N^2) and impossible at 5.35M cells.
pynndescent gives the same graph to within a small recall loss at a
fraction of the cost.  Recall is measured against brute force on a random
sample.
"""
import argparse, json, os, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--emb", required=True)
ap.add_argument("--out-dir", required=True)
ap.add_argument("--prefix", required=True)
ap.add_argument("--k", type=int, default=30, help="neighbours excluding self")
ap.add_argument("--recall-cells", type=int, default=2000)
ap.add_argument("--seed", type=int, default=0)
A = ap.parse_args()
t0=time.time(); log=lambda m: print(f"[{time.time()-t0:8.1f}s] {m}", flush=True)

Z = np.load(A.emb).astype(np.float32)
N, d = Z.shape
log(f"{N:,} cells x {d} dims from {os.path.basename(A.emb)}")

from pynndescent import NNDescent
t1=time.time()
index = NNDescent(Z, n_neighbors=A.k+1, metric="euclidean", random_state=A.seed,
                  low_memory=True, verbose=True)
idx, dist = index.neighbor_graph
log(f"NNDescent in {(time.time()-t1)/60:.1f} min -> {idx.shape}")
assert idx.shape == (N, A.k+1)

# self should be column 0; enforce it so downstream can assume knn[:,1:] are neighbours
selfcol = (idx[:, 0] == np.arange(N))
log(f"self is column 0 for {selfcol.mean()*100:.2f}% of cells; repairing the rest")
if not selfcol.all():
    bad = np.flatnonzero(~selfcol)
    for i in bad:
        row = idx[i]; pos = np.flatnonzero(row == i)
        if len(pos):
            j = pos[0]; row[[0, j]] = row[[j, 0]]
            dist[i][[0, j]] = dist[i][[j, 0]]
        else:
            idx[i, 1:] = idx[i, :-1]; idx[i, 0] = i
            dist[i, 1:] = dist[i, :-1]; dist[i, 0] = 0.0
assert (idx[:, 0] == np.arange(N)).all(), "self-column repair failed"

# ---- gate: recall against brute force on a random sample --------------------
rng = np.random.default_rng(A.seed)
sub = rng.choice(N, size=min(A.recall_cells, N), replace=False)
hits = 0; tot = 0
for c in sub:
    dd = np.linalg.norm(Z - Z[c], axis=1)
    true = np.argpartition(dd, A.k+1)[:A.k+1]
    hits += len(np.intersect1d(true, idx[c])); tot += A.k+1
rec = hits/tot
log(f"gate: kNN recall vs brute force on {len(sub)} cells = {rec*100:.2f}%")
assert rec > 0.85, f"approximate kNN recall too low ({rec:.3f})"

np.save(os.path.join(A.out_dir, f"knn_idx_{A.prefix}.npy"), idx.astype(np.int32))
np.save(os.path.join(A.out_dir, f"knn_dist_{A.prefix}.npy"), dist.astype(np.float32))
json.dump(dict(n_cells=int(N), d=int(d), k=A.k, recall=float(rec), seed=A.seed,
               emb=os.path.basename(A.emb)),
          open(os.path.join(A.out_dir, f"knn_{A.prefix}.json"), "w"), indent=2)
log("KNN OK")
