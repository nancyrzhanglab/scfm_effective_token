#!/usr/bin/env python
"""Step 3 of 3 -- agglomerative merging of the SEACells base into a NESTED ladder.

This is the step that buys nesting.  SEACells is a flat method: running it at
k=16 and again at k=64 gives two partitions with no containment relation, so a
"metacell at k=64" is not a union of "metacells at k=16" and quantities cannot be
tracked across k.  Ward merging fixes that structurally -- ONE dendrogram is
built over the base metacells and every rung is a cut of that same tree, so each
coarse metacell is exactly a union of finer ones by construction, not by
approximation.  Nesting is a property of the tree, so it holds for any cut and is
asserted at the end rather than hoped for.

CONNECTIVITY.  Unconstrained Ward on 335k points is O(n^2) in both time and
memory.  A k-NN connectivity graph over the base centroids restricts merges to
neighbours, which makes the tree tractable and also keeps merges local in
Harmony space -- a pair of base metacells at opposite ends of the manifold can
never be fused just because the variance arithmetic likes it.  Where the graph
has several connected components sklearn bridges them; those bridge merges are
the only non-local ones and they occur near the root, above the rungs of
interest.

BRIDGING PILES.  Base metacells from different piles are merged freely here --
the tree knows nothing about pile identity.  That is deliberate: it lets the
merge repair boundaries that step 1's partition imposed for tractability.

CELL-WEIGHTED WARD.  The linkage is implemented here rather than taken from
sklearn, because sklearn's ward_tree accepts no sample weights: it counts each
base centroid once regardless of how many cells that centroid stands for.  Base
metacell sizes are centred on the target but far from uniform (roughly 1 to 40
cells), so the unweighted variant both misstates the variance arithmetic and
produces badly skewed cuts -- on a 9k-cell rehearsal it gave a k=128 rung whose
largest metacell held 1050 cells against a median of 22.

The merge cost used here is the true Ward increase in CELL-level inertia,

    d(a,b) = n_a n_b / (n_a + n_b) * ||c_a - c_b||^2 ,   n = cells, not metacells

which is exactly Ward on the underlying cells, since a centroid weighted by its
cell count carries the same first two moments as its members.  The n_a n_b term
penalises fusing two already-large metacells, so the cut sizes stay far closer to
the target than the unweighted version -- the balancing is a consequence of using
the right cost, not a constraint bolted on afterwards.

Greedy heap agglomeration with lazy deletion, restricted to the connectivity
graph.  Merges are recorded as a union-find sequence, so a cut at T clusters is
"apply the first NB - T merges" and nesting across rungs is automatic.

Rungs are cell-count targets: T_k = round(n_base * base_k / k), i.e. the cut that
makes the MEAN metacell size equal k.

Output: metacell_labels_<tag>.npz with one int32 label vector per rung, keyed
lab_mean<k>_<tag> and indexed by FILTERED-cohort row, matching the layout of
task9's recursive-Leiden ladder so the downstream analysis scripts take it
unchanged.  base_rowidx.npy maps those rows to the full 5,381,464-row cohort.
"""
import argparse, glob, json, os, time

import numpy as np
from scipy import sparse
from sklearn.cluster import ward_tree


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


ap = argparse.ArgumentParser()
ap.add_argument("--pile-dir", required=True)
ap.add_argument("--pattern", default="pile_*.npz",
                help="glob for the per-pile outputs.  SEACells writes pile_*.npz, "
                     "MetaCell-1 writes mc1_pile_*_K<K>.npz; both carry the same "
                     "arrays (rows/local/centroids/sizes/meta) so the merge is "
                     "identical either way.")
ap.add_argument("--piles", required=True)
ap.add_argument("--out-prefix", required=True)
ap.add_argument("--tag", default="seacells_ward")
ap.add_argument("--base-k", type=int, default=16)
ap.add_argument("--rungs", type=int, nargs="+",
                default=[16, 32, 64, 128, 256, 512, 1024])
ap.add_argument("--n-neighbors", type=int, default=15)
ap.add_argument("--linkage", choices=["ward_weighted", "ward_sklearn"],
                default="ward_weighted",
                help="ward_weighted = cell-count-weighted Ward (correct); "
                     "ward_sklearn = sklearn ward_tree, each base metacell counts 1")
ap.add_argument("--seed", type=int, default=0)
A = ap.parse_args()

# --- gather the base rung ---------------------------------------------------------
fs = sorted(glob.glob(os.path.join(A.pile_dir, A.pattern)))
assert fs, f"no {A.pattern} in {A.pile_dir}"
n_piles_expected = int(np.load(A.piles).max()) + 1
assert len(fs) == n_piles_expected, \
    f"{len(fs)} pile outputs but {n_piles_expected} piles -- array job incomplete"
log(f"{len(fs)} pile outputs")

rows_l, base_l, cent_l, size_l, pile_l = [], [], [], [], []
off = 0
for f in fs:
    z = np.load(f, allow_pickle=False)
    m = json.loads(str(z["meta"]))
    rows_l.append(z["rows"])
    base_l.append(z["local"].astype(np.int64) + off)
    cent_l.append(z["centroids"])
    size_l.append(z["sizes"])
    pile_l.append(np.full(len(z["sizes"]), m["pile"], np.int32))
    off += len(z["sizes"])

rows = np.concatenate(rows_l)
base = np.concatenate(base_l)
CENT = np.concatenate(cent_l).astype(np.float32)
SIZE = np.concatenate(size_l).astype(np.int64)
PILE = np.concatenate(pile_l)
del rows_l, base_l, cent_l, size_l, pile_l

order = np.argsort(rows, kind="stable")
rows, base = rows[order], base[order]
assert np.all(np.diff(rows) > 0), "pile row sets overlap or repeat"
N = len(rows)
NB = len(SIZE)
assert SIZE.sum() == N
log(f"base rung: {NB:,} metacells over {N:,} cells "
    f"(mean {N/NB:.2f} cells/metacell, target {A.base_k})")
log(f"base sizes: min {SIZE.min()} p10 {np.percentile(SIZE,10):.0f} "
    f"median {np.median(SIZE):.0f} p90 {np.percentile(SIZE,90):.0f} max {SIZE.max()}")

# --- k-NN connectivity over base centroids ----------------------------------------
# exact k-NN in 50-d over 335k points is brute-force territory; NN-descent gives
# the same neighbourhood structure at a fraction of the cost, and Ward only needs
# the graph to be a reasonable locality constraint, not an exact k-NN.
from pynndescent import NNDescent

t0 = time.time()
index = NNDescent(CENT, n_neighbors=A.n_neighbors + 1, metric="euclidean",
                  random_state=A.seed, low_memory=True, verbose=True)
nbr, _ = index.neighbor_graph
log(f"k-NN graph in {time.time()-t0:.0f}s")

src = np.repeat(np.arange(NB, dtype=np.int64), A.n_neighbors)
dst = nbr[:, 1:A.n_neighbors + 1].astype(np.int64).ravel()
conn = sparse.coo_matrix((np.ones(len(src), np.int8), (src, dst)),
                         shape=(NB, NB)).tocsr()
conn = conn + conn.T                      # symmetrise; ward_tree wants undirected
conn.data[:] = 1
ncomp = sparse.csgraph.connected_components(conn, directed=False,
                                            return_labels=False)
log(f"connectivity: {conn.nnz:,} edges, {ncomp} connected components")

# --- one merge tree over the whole base --------------------------------------------
targets = {}
for k in A.rungs:
    T = max(1, int(round(NB * A.base_k / k)))
    targets[k] = min(T, NB)
log("rung -> n_metacells: " + ", ".join(f"k={k}:{t:,}" for k, t in targets.items()))
n_merges_needed = NB - min(targets.values())

parent = np.arange(NB, dtype=np.int64)


def find(x):
    r = x
    while parent[r] != r:
        r = parent[r]
    while parent[x] != r:
        parent[x], x = r, parent[x]
    return r


def weighted_ward_merges(cent, size, conn, n_needed):
    """Greedy cell-weighted Ward over the connectivity graph.

    Returns the merge order as (survivor, absorbed) pairs of base-metacell ids.
    Lazy deletion: a heap entry is stale once either endpoint has been merged
    since it was pushed, which `ver` detects in O(1) -- cheaper than keeping the
    heap consistent, and the standard device for this algorithm.
    """
    import heapq

    n = len(size)
    C = cent.astype(np.float64).copy()
    W = size.astype(np.float64).copy()
    alive = np.ones(n, bool)
    ver = np.zeros(n, np.int64)
    adj = [set() for _ in range(n)]
    coo = conn.tocoo()
    for a, b in zip(coo.row, coo.col):
        if a != b:
            adj[a].add(int(b))
            adj[b].add(int(a))

    def cost(a, b):
        d = C[a] - C[b]
        return (W[a] * W[b] / (W[a] + W[b])) * float(d @ d)

    heap = []
    for a in range(n):
        for b in adj[a]:
            if b > a:
                heap.append((cost(a, b), a, b, 0, 0))
    heapq.heapify(heap)
    log(f"  heap seeded with {len(heap):,} candidate merges")

    merges = []
    t0 = time.time()
    while len(merges) < n_needed and heap:
        d, a, b, va, vb = heapq.heappop(heap)
        if a == b or not (alive[a] and alive[b]) or ver[a] != va or ver[b] != vb:
            continue
        # keep the endpoint with the larger adjacency so the set union is
        # small-into-large; which id survives is irrelevant to the partition
        keep, drop = (a, b) if len(adj[a]) >= len(adj[b]) else (b, a)
        C[keep] = (C[keep] * W[keep] + C[drop] * W[drop]) / (W[keep] + W[drop])
        W[keep] += W[drop]
        alive[drop] = False
        parent[drop] = keep
        merges.append((keep, drop))

        adj[keep] |= adj[drop]
        adj[keep].discard(keep)
        adj[keep].discard(drop)
        for q in adj[drop]:
            if q != keep and alive[q]:
                adj[q].discard(drop)
                adj[q].add(keep)
        adj[drop] = set()
        # the rewrite above can hand `keep` back an edge to itself; a self-loop
        # costs zero and would pop straight off the heap as a merge of a cluster
        # with itself, silently destroying it
        adj[keep].discard(keep)
        adj[keep].discard(drop)
        ver[keep] += 1
        for q in adj[keep]:
            if alive[q]:
                heapq.heappush(heap, (cost(keep, q), keep, q, ver[keep], ver[q]))
        if len(merges) % 25_000 == 0:
            log(f"  {len(merges):,}/{n_needed:,} merges "
                f"({time.time()-t0:.0f}s, heap {len(heap):,})")

    # The kNN graph is connected in practice, but if it is not the heap empties
    # with several components still alive.  Finish by fusing whole components on
    # the same Ward cost -- few merges, so the quadratic scan is free.
    if len(merges) < n_needed:
        rest = np.flatnonzero(alive)
        log(f"  heap exhausted with {len(rest)} clusters; bridging components")
        while len(merges) < n_needed and len(rest) > 1:
            best, pair = np.inf, None
            for i in range(len(rest)):
                for j in range(i + 1, len(rest)):
                    c = cost(rest[i], rest[j])
                    if c < best:
                        best, pair = c, (int(rest[i]), int(rest[j]))
            keep, drop = pair
            C[keep] = (C[keep] * W[keep] + C[drop] * W[drop]) / (W[keep] + W[drop])
            W[keep] += W[drop]
            alive[drop] = False
            parent[drop] = keep
            merges.append((keep, drop))
            rest = np.flatnonzero(alive)
    return merges


t0 = time.time()
if A.linkage == "ward_weighted":
    merges = weighted_ward_merges(CENT, SIZE, conn, n_merges_needed)
    log(f"cell-weighted Ward: {len(merges):,} merges in {time.time()-t0:.0f}s")
else:
    children, _, n_leaves, _ = ward_tree(CENT.astype(np.float64),
                                         connectivity=conn, n_clusters=None)
    assert n_leaves == NB
    rep = np.empty(2 * NB - 1, dtype=np.int64)
    rep[:NB] = np.arange(NB)
    merges = []
    for i, (c1, c2) in enumerate(children):
        r1, r2 = rep[c1], rep[c2]
        merges.append((int(r1), int(r2)))
        rep[NB + i] = r1
    log(f"sklearn ward_tree: {len(merges):,} merges in {time.time()-t0:.0f}s")
    parent[:] = np.arange(NB)

assert len(merges) >= n_merges_needed, \
    f"only {len(merges):,} merges available, need {n_merges_needed:,}"

# --- cut the SAME merge order at each rung -------------------------------------------
# Replaying one prefix of one merge order is what makes the rungs exactly nested:
# a cut at T is a coarsening of every cut at T' > T.
snap_at = {NB - t: k for k, t in targets.items()}
assert len(snap_at) == len(targets), "two rungs map to the same cluster count"
base_labels = {}
parent[:] = np.arange(NB)
if 0 in snap_at:
    base_labels[snap_at[0]] = np.arange(NB, dtype=np.int64)
for m, (keep, drop) in enumerate(merges, start=1):
    parent[find(drop)] = find(keep)
    if m in snap_at:
        lab = np.array([find(j) for j in range(NB)], dtype=np.int64)
        base_labels[snap_at[m]] = lab
        log(f"  cut after {m:,} merges -> k={snap_at[m]} "
            f"({len(np.unique(lab)):,} clusters)")

assert set(base_labels) == set(A.rungs), "a rung was not snapshotted"

# --- relabel, verify nesting, expand to cells --------------------------------------
out = {}
stats = []
prev_k, prev_comp = None, None
for k in sorted(A.rungs):
    _, comp = np.unique(base_labels[k], return_inverse=True)
    T = int(comp.max()) + 1
    assert T == targets[k], f"k={k}: got {T} clusters, wanted {targets[k]}"
    cell = comp[base].astype(np.int32)
    out[f"lab_mean{k}_{A.tag}"] = cell
    sz = np.bincount(cell, minlength=T)
    stats.append({"k": k, "n_metacells": T, "mean_size": float(N / T),
                  "size_min": int(sz.min()), "size_median": float(np.median(sz)),
                  "size_p90": float(np.percentile(sz, 90)), "size_max": int(sz.max())})
    log(f"k={k:5d}: {T:>8,} metacells | size min {sz.min()} median "
        f"{np.median(sz):.0f} max {sz.max()}")
    if prev_comp is not None:
        # every finer metacell must sit inside exactly one coarser one.  Checked
        # over base metacells, not cells: a cell's label is a lookup through the
        # same `base` array at both rungs, so the two statements are equivalent
        # and this one is 16x smaller.
        pair = np.unique(np.stack([prev_comp, comp]), axis=1)
        assert len(np.unique(pair[0])) == pair.shape[1], \
            f"k={prev_k} is not nested inside k={k}"
    prev_k, prev_comp = k, comp
log("nesting verified for every consecutive rung pair")

os.makedirs(os.path.dirname(A.out_prefix), exist_ok=True)
np.savez_compressed(A.out_prefix + f"_{A.tag}.npz", **out)
np.save(A.out_prefix + "_base_metacell.npy", base.astype(np.int32))
np.save(A.out_prefix + "_base_sizes.npy", SIZE.astype(np.int32))
np.save(A.out_prefix + "_base_centroids.npy", CENT)
np.save(A.out_prefix + "_base_pile.npy", PILE)
np.save(A.out_prefix + "_rowidx_filtered.npy", rows)
json.dump({"n_cells": int(N), "n_base_metacells": int(NB), "base_k": A.base_k,
           "rungs": list(map(int, A.rungs)), "n_piles": len(fs),
           "n_neighbors": A.n_neighbors, "n_connected_components": int(ncomp),
           "linkage": A.linkage,
           "base_size_min": int(SIZE.min()), "base_size_max": int(SIZE.max()),
           "base_size_median": float(np.median(SIZE)),
           "base_size_p10": float(np.percentile(SIZE, 10)),
           "base_size_p90": float(np.percentile(SIZE, 90)),
           "rung_stats": stats},
          open(A.out_prefix + f"_{A.tag}.json", "w"), indent=2)
log(f"wrote {A.out_prefix}_{A.tag}.npz")
log("DONE")
