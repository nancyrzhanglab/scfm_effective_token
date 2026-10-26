#!/usr/bin/env python
"""Nested metacells by recursive Leiden, target mean sizes 512 -> 2.

A max-heap of pools, pop the largest, Leiden-split it, push the children, stop 
when the pool count reaches N/mean. Because the recursion only ever splits, the 
ladder is strictly nested by construction and coverage is 1.0 at every rung.

Outputs (--out-dir):
    metacell_labels_<prefix>_leiden.npz    lab_mean{k}_leiden_s0 : int32 (N,)
    metacell_partition_<prefix>_leiden.csv
    metacell_<prefix>_leiden.json
"""
import argparse, heapq, json, os, time
import numpy as np, pandas as pd
import igraph as ig, leidenalg as la
from scipy.sparse import coo_matrix

ap = argparse.ArgumentParser()
ap.add_argument("--knn", required=True)
ap.add_argument("--out-dir", required=True)
ap.add_argument("--prefix", required=True)
ap.add_argument("--adj-k", type=int, default=30)
ap.add_argument("--target-means", default="512,256,128,64,32,16,8,4,2")
ap.add_argument("--resolution", type=float, default=1.0)
ap.add_argument("--retries", type=int, default=4)
ap.add_argument("--seed", type=int, default=0)
A = ap.parse_args()
t0=time.time(); log=lambda m: print(f"[{time.time()-t0:8.1f}s] {m}", flush=True)
TM=[int(x) for x in A.target_means.split(",")]
assert TM==sorted(TM, reverse=True), "--target-means must be descending"
# k=1 is the identity partition (every cell its own metacell).  Leiden cannot
# split a single-vertex subgraph, so it is emitted directly.

WANT_ONE = 1 in TM
TM = [t for t in TM if t > 1]
log(f"args: {vars(A)}")

knn = np.load(A.knn)
N = knn.shape[0]
assert (knn[:,0]==np.arange(N)).all(), "column 0 must be the focal cell"
NB = knn[:, 1:1+A.adj_k].astype(np.int64)
src = np.repeat(np.arange(N, dtype=np.int64), A.adj_k)
Aadj = coo_matrix((np.ones(src.size, np.float32), (src, NB.ravel())), shape=(N,N)).tocsr()
Aadj = Aadj.maximum(Aadj.T)                      # symmetrise: mutual OR one-sided
Aadj.setdiag(0); Aadj.eliminate_zeros()
g = ig.Graph.Weighted_Adjacency(Aadj, mode="undirected", attr="weight", loops=False)
log(f"graph: {g.vcount():,} vertices, {g.ecount():,} undirected edges (adj_k {A.adj_k})")
assert g.vcount()==N

def split(vids, res, seed):
    sub = g.subgraph(vids)
    part = la.find_partition(sub, la.RBConfigurationVertexPartition,
                             weights="weight", resolution_parameter=res,
                             n_iterations=2, seed=seed)
    out=[]
    for comm in part:
        if comm: out.append([vids[i] for i in comm])
    return out

pools = {0: list(range(N))}
heap = [(-N, 0, pools[0])]
nxt_id = 1
store, rows = {}, []
prev_lab = None
os.makedirs(A.out_dir, exist_ok=True)
npz_path = os.path.join(A.out_dir, f"metacell_labels_{A.prefix}_leiden.npz")

for tm in TM:
    target = max(1, int(round(N/tm)))
    stalled = 0; t1=time.time()
    while len(pools) < target and heap:
        negsz, pid, vids = heapq.heappop(heap)
        if pid not in pools: continue
        res, parts = A.resolution, None
        for t in range(A.retries+1):
            parts = split(vids, res, A.seed+t)
            if len(parts) > 1: break
            res *= 2.0
        if parts is None or len(parts) <= 1:
            stalled += 1; continue           # unsplittable: leave the queue
        del pools[pid]
        for pr in parts:
            pools[nxt_id]=pr
            heapq.heappush(heap, (-len(pr), nxt_id, pr))
            nxt_id += 1
    lab = np.empty(N, np.int64)
    for i, vids in enumerate(pools.values()):
        lab[np.asarray(vids, np.int64)] = i
    sizes = np.bincount(lab)
    assert sizes.sum()==N and (sizes>0).all(), "partition does not cover every cell once"
    # strict nesting: every child at this rung sits inside one parent of the last
    if prev_lab is not None:
        order = np.argsort(lab, kind="stable")
        b = lab[order]; p = prev_lab[order]
        starts = np.flatnonzero(np.r_[True, b[1:]!=b[:-1]])
        ends = np.r_[starts[1:], len(b)]
        nested = all(len(np.unique(p[s:e]))==1 for s,e in zip(starts,ends))
        assert nested, f"rung mean={tm} is NOT nested inside the previous rung"
    prev_lab = lab.copy()
    store[f"lab_mean{tm}_leiden_s0"] = lab.astype(np.int32)
    np.savez_compressed(npz_path, **store)        # checkpoint every rung
    rows.append(dict(target_mean=tm, n_pools=len(pools), coverage=1.0,
                     size_min=int(sizes.min()), size_median=float(np.median(sizes)),
                     size_max=int(sizes.max()), size_mean=float(sizes.mean()),
                     n_stalled=stalled, minutes=float((time.time()-t1)/60)))
    pd.DataFrame(rows).to_csv(
        os.path.join(A.out_dir, f"metacell_partition_{A.prefix}_leiden.csv"), index=False)
    log(f"target mean {tm:4d}: {len(pools):8,} pools, achieved mean {sizes.mean():7.2f}, "
        f"sizes {sizes.min()}-{sizes.max()} (median {np.median(sizes):.0f}), "
        f"{stalled} unsplittable, {(time.time()-t1)/60:.1f} min  [nested OK]")

if WANT_ONE:
    store["lab_mean1_leiden_s0"] = np.arange(N, dtype=np.int32)
    rows.append(dict(target_mean=1, n_pools=int(N), coverage=1.0, size_min=1,
                     size_median=1.0, size_max=1, size_mean=1.0,
                     n_stalled=0, minutes=0.0))
    np.savez_compressed(npz_path, **store)
    pd.DataFrame(rows).to_csv(
        os.path.join(A.out_dir, f"metacell_partition_{A.prefix}_leiden.csv"), index=False)
    log("k=1 identity rung added (every cell its own metacell)")

json.dump(dict(prefix=A.prefix, n_cells=int(N), adj_k=A.adj_k,
               target_means=sorted(set(TM + ([1] if WANT_ONE else [])), reverse=True),
               resolution=A.resolution, seed=A.seed,
               rungs=rows, knn=os.path.basename(A.knn)),
          open(os.path.join(A.out_dir, f"metacell_{A.prefix}_leiden.json"), "w"), indent=2)
log("METACELLS OK")
