#!/usr/bin/env python
"""MetaCell (Baran et al. 2019, Genome Biology 20:206) on one pile.

Implements the algorithm of the linked paper directly -- the authors' own R
package is archived, and the maintained `metacells` PyPI package implements
MetaCell-2 (Ben-Kiki 2022), a different algorithm.

  1. BALANCED K-NN.  s_ij = rank of j among i's neighbours (1 = nearest).  The
     paper ranks Pearson correlations of log UMIs over feature genes; here the
     similarity is Euclidean distance in the Harmony embedding.  Only the RANKS
     enter, and any strictly monotone similarity induces the same ranks, so the
     substitution changes the input space, not the algorithm.  Edges are kept
     where the symmetrised rank product s_ij * s_ji <= alpha*K^2, then capped at
     beta*K incoming and K outgoing per node.

  2. RESAMPLING.  NB subgraphs, each sampling cells independently without
     replacement with probability rho.  Each is partitioned by (3).  C_ij counts
     co-assignment to the same metacell, O_ij counts co-sampling, and the
     co-occurrence weight is C/O.  Accumulated ONLY on the balanced-graph edge
     set: the paper partitions the co-occurrence graph, and a dense n x n pair
     matrix would be 750 MB per pile for pairs that can never become edges.

  3. GREEDY PARTITION.  Seeds are drawn from uncovered cells with weight
     proportional to f(i)^3, f(i) = number of uncovered out-neighbours; the seed
     and its uncovered out-neighbours form a metacell; seeding stops when no
     candidate has f(i) >= size_min.  Then cells are reassigned to
     argmax_k w_ik until no label moves, where w_ik is the MEAN edge weight into
     metacell k -- edges(i,k) / |k|.  Normalising by |k| is essential: with raw
     edge counts the argmax favours whichever metacell is already largest, and
     the partition collapses (on a 3,775-cell pile it gave 10 metacells of mean
     size 377 instead of ~38 of ~100).
     f is maintained INCREMENTALLY -- each covering decrements only the in-
     neighbours of the covered cell, so seeding is O(E) overall rather than
     O(E * n_seeds), which is what makes NB=500 affordable.

  4. OUTLIERS/SPLITTING.  The paper flags outliers by per-gene observed/expected
     fold change and splits metacells whose members form several DBSCAN clusters.
     Both are gene-expression tests; on an embedding the gene test has no
     analogue, so only the splitting check is done, in embedding space, and it is
     REPORTED not applied (n_multi_cluster in the metadata).  Deviation recorded
     rather than silently skipped.

NOTE ON NESTING.  MetaCell is a flat method.  Running it at two target sizes
gives two partitions with no containment relation -- unlike task10's Ward tree.
Rungs produced this way must not be treated as a nested ladder.
"""
import argparse, json, os, sys, time

import numpy as np
from scipy import sparse


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def balanced_knn(X, K, alpha, beta, seed):
    """Rank-symmetrised, degree-capped K-nn graph.  Returns CSR (unweighted)."""
    from pynndescent import NNDescent
    n = len(X)
    kq = min(n - 1, K)
    index = NNDescent(X, n_neighbors=kq + 1, metric="euclidean",
                      random_state=seed, low_memory=True, verbose=False)
    nbr, _ = index.neighbor_graph
    nbr = nbr[:, 1:kq + 1]                       # drop self
    src = np.repeat(np.arange(n), kq)
    dst = nbr.ravel().astype(np.int64)
    rank = np.tile(np.arange(1, kq + 1), n).astype(np.float64)
    R = sparse.csr_matrix((rank, (src, dst)), shape=(n, n))

    # s_ij * s_ji ; a pair present in only one direction gets the missing rank
    # imputed as the worst possible (kq), which is how a one-sided neighbour is
    # penalised rather than dropped outright
    Rt = R.T.tocsr()
    both = R.multiply(Rt)
    onlyR = R - R.multiply(Rt.astype(bool))
    onlyT = Rt - Rt.multiply(R.astype(bool))
    prod = (both + onlyR * kq + onlyT * kq).tocoo()
    keep = prod.data <= alpha * (kq ** 2)
    r, c, w = prod.row[keep], prod.col[keep], prod.data[keep]

    # cap beta*K incoming, then K outgoing; smaller rank product = stronger edge
    def cap(rr, cc, ww, by, lim):
        order = np.lexsort((ww, by))
        grp = by[order]
        first = np.r_[True, grp[1:] != grp[:-1]]
        pos = np.arange(len(order)) - np.maximum.accumulate(
            np.where(first, np.arange(len(order)), 0))
        sel = order[pos < lim]
        return rr[sel], cc[sel], ww[sel]

    r, c, w = cap(r, c, w, c, int(beta * kq))
    r, c, w = cap(r, c, w, r, kq)
    A = sparse.csr_matrix((np.ones(len(r)), (r, c)), shape=(n, n))
    A.setdiag(0); A.eliminate_zeros()
    return A


def greedy_partition(A, size_min, rng, max_reassign=20):
    """Seed-and-expand, then reassign to argmax edge weight.  Returns labels."""
    n = A.shape[0]
    At = A.T.tocsr()
    indptr, indices = A.indptr, A.indices
    tptr, tind = At.indptr, At.indices

    mc = np.full(n, -1, np.int64)
    f = np.diff(indptr).astype(np.int64)          # uncovered out-degree
    covered = np.zeros(n, bool)

    def cover(u, k):
        if covered[u]:
            return
        covered[u] = True
        mc[u] = k
        for t in range(tptr[u], tptr[u + 1]):     # in-neighbours lose one
            f[tind[t]] -= 1

    k = 0
    n_seed = 0
    while True:
        cand = np.flatnonzero((~covered) & (f >= size_min))
        if cand.size == 0:
            break
        wgt = f[cand].astype(np.float64) ** 3
        j = int(rng.choice(cand, p=wgt / wgt.sum()))
        members = [j] + [u for u in indices[indptr[j]:indptr[j + 1]] if not covered[u]]
        for u in members:
            cover(u, k)
        k += 1; n_seed += 1
    if k == 0:                                    # graph too sparse to seed
        return np.zeros(n, np.int64), 1

    # leftovers + reassignment
    for _ in range(max_reassign):
        M = sparse.csr_matrix(
            (np.ones((mc >= 0).sum()), (np.flatnonzero(mc >= 0), mc[mc >= 0])),
            shape=(n, k))
        ksz = np.asarray(M.sum(0)).ravel()
        W = (np.asarray((A @ M).todense()) + np.asarray((At @ M).todense()))
        W /= np.maximum(ksz, 1)[None, :]
        best = W.max(1)
        new = np.where(best > 0, W.argmax(1), mc)
        if np.array_equal(new, mc):
            break
        mc = new
    un = mc < 0
    if un.any():                                  # never touched by any edge
        mc[un] = rng.integers(0, k, un.sum())
    _, mc = np.unique(mc, return_inverse=True)
    return mc.astype(np.int64), int(mc.max()) + 1


def _seed_count(A, size_min, rng):
    return None


ap = argparse.ArgumentParser()
ap.add_argument("--emb", required=True)
ap.add_argument("--piles", required=True)
ap.add_argument("--pile", type=int, required=True)
ap.add_argument("--out-dir", required=True)
ap.add_argument("--K", type=int, default=100, help="kNN degree ~ target metacell size")
ap.add_argument("--alpha", type=float, default=1.0, help="rank-product threshold, alpha*K^2")
ap.add_argument("--beta", type=float, default=3.0, help="incoming-degree cap, beta*K")
ap.add_argument("--n-boot", type=int, default=500)
ap.add_argument("--rho", type=float, default=0.75)
ap.add_argument("--size-min", type=int, default=0, help="0 -> K//2, the paper's size_min < K")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--force", action="store_true")
A = ap.parse_args()

os.makedirs(A.out_dir, exist_ok=True)
out = os.path.join(A.out_dir, f"mc1_pile_{A.pile:05d}_K{A.K}.npz")
if os.path.exists(out) and not A.force:
    log(f"{os.path.basename(out)} exists, skipping")
    sys.exit(0)

piles = np.load(A.piles)
rows = np.flatnonzero(piles == A.pile)
n = len(rows)
E = np.load(A.emb, mmap_mode="r")
X = np.ascontiguousarray(E[rows], dtype=np.float32)
size_min = A.size_min or max(2, A.K // 2)
rng = np.random.default_rng(A.seed + A.pile)
t0 = time.time()
log(f"pile {A.pile}: {n:,} cells, K={A.K}, size_min={size_min}, "
    f"NB={A.n_boot}, rho={A.rho}")

Ab = balanced_knn(X, A.K, A.alpha, A.beta, A.seed)
log(f"balanced K-nn: {Ab.nnz:,} edges, mean out-degree {Ab.nnz/n:.1f} ({time.time()-t0:.0f}s)")

# co-occurrence accumulated on the balanced edge set only
Ac = sparse.triu(Ab + Ab.T, k=1).tocoo()
ei, ej = Ac.row, Ac.col
C = np.zeros(len(ei), np.int32)
O = np.zeros(len(ei), np.int32)
for b in range(A.n_boot):
    samp = rng.random(n) < A.rho
    both = samp[ei] & samp[ej]
    O += both
    idx = np.flatnonzero(samp)
    remap = np.full(n, -1, np.int64); remap[idx] = np.arange(len(idx))
    Asub = Ab[idx][:, idx]
    lab, _ = greedy_partition(Asub, size_min, rng)
    full = np.full(n, -1, np.int64); full[idx] = lab
    C += both & (full[ei] == full[ej])
    if (b + 1) % 100 == 0:
        log(f"  bootstrap {b+1}/{A.n_boot} ({time.time()-t0:.0f}s)")

w = np.where(O > 0, C / np.maximum(O, 1), 0.0)
Sb = sparse.csr_matrix((w, (ei, ej)), shape=(n, n))
Sb = Sb + Sb.T
log(f"co-occurrence graph: {Sb.nnz:,} edges, mean weight {w[w>0].mean():.3f}")

# final partition: keep the strongest K co-occurrence edges per node, then the
# same greedy algorithm on that unweighted graph
Sc = Sb.tocoo()
keep_mask = Sc.data > 0
r, c, v = Sc.row[keep_mask], Sc.col[keep_mask], Sc.data[keep_mask]
order = np.lexsort((-v, r))
grp = r[order]
first = np.r_[True, grp[1:] != grp[:-1]]
pos = np.arange(len(order)) - np.maximum.accumulate(
    np.where(first, np.arange(len(order)), 0))
sel = order[pos < A.K]
Af = sparse.csr_matrix((np.ones(len(sel)), (r[sel], c[sel])), shape=(n, n))
lab, K_final = greedy_partition(Af, size_min, rng)
sz = np.bincount(lab, minlength=K_final)
log(f"{K_final:,} metacells | size min {sz.min()} median {int(np.median(sz))} "
    f"mean {sz.mean():.1f} max {sz.max()}")

D = X.shape[1]
cent = np.zeros((K_final, D), np.float64)
np.add.at(cent, lab, X.astype(np.float64))
cent /= sz[:, None]

# reported, not applied: does any metacell split into several density clusters?
from sklearn.cluster import DBSCAN
n_multi = 0
for m in rng.choice(K_final, size=min(50, K_final), replace=False):
    mem = np.flatnonzero(lab == m)
    if len(mem) < 10:
        continue
    d = DBSCAN(eps=np.median(np.linalg.norm(X[mem] - cent[m], axis=1)),
               min_samples=5).fit_predict(X[mem])
    if len(set(d[d >= 0])) > 1:
        n_multi += 1

np.savez_compressed(
    out, rows=rows.astype(np.int64), local=lab.astype(np.int32),
    centroids=cent.astype(np.float32), sizes=sz.astype(np.int32),
    meta=np.array(json.dumps({
        "pile": A.pile, "n_cells": int(n), "K": A.K, "alpha": A.alpha,
        "beta": A.beta, "n_boot": A.n_boot, "rho": A.rho,
        "size_min": size_min, "n_metacells": int(K_final),
        "n_edges_balanced": int(Ab.nnz),
        "n_multi_cluster_of_50_sampled": int(n_multi),
        "seconds": round(time.time() - t0, 1),
        "method": "MetaCell-1 (Baran 2019) on Harmony embedding"})))
log(f"wrote {out}  ({time.time()-t0:.0f}s)")
log("DONE")
