#!/usr/bin/env python
"""MetaCell-1 (Baran et al. 2019, Genome Biology 20:206) on one pile, run
through the authors' engine (tgstat, the C++ behind tanaylab/metacell).

The algorithm is step2e_metacell1_tgstat.R: the MetaCell R package's own pipeline
(bknn graph -> 500 resampled graph covers -> balanced co-cluster cover) calling
tgstat.  It replaces an earlier Python re-implementation that departed from the
package in its balanced k-NN, its graph cover and its outlier handling; that
version is kept as scripts/poisson/step13_verify_MetaCell1/
step2e_metacell1_pile_python_original.py.bak.  Needs R with tgstat
(`module load R/4.5.2`; tgstat 2.4.0 in the user library).

This driver does only what MetaCell does not:
  1. the input k-NN.  Exact Euclidean k-NN in the embedding, K * k_expand
     neighbours per cell, self excluded, rank 1 = nearest -- the layout of
     tgs_cor_knn's output, which is what tgs_graph consumes.  Exact (brute
     force) rather than approximate: a pile is <= ~24k cells, and k_expand = 10
     asks for 640 neighbours, deep in the tail where NN-descent recall drops.
  2. the hand-off to R and back.
  3. centroids and the per-pile npz in the layout step2f_merge_ward.py reads
     (rows / local / centroids / sizes / meta).

OUTLIERS ARE KEPT OUT.  MetaCell leaves cells it cannot cover unassigned; they
are written with no metacell -- `rows` lists only the assigned cells, and the
outlier rows go to `outlier_rows` -- so the Ward merge never sees them and the
expansion step labels them -1, which cv_ltv drops from every term.
"""
import argparse, json, os, subprocess, sys, tempfile, time

import numpy as np


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--emb", required=True)
ap.add_argument("--piles", required=True)
ap.add_argument("--pile", type=int, required=True)
ap.add_argument("--out-dir", required=True)
ap.add_argument("--rscript", default="Rscript")
# MetaCell parameters.  Defaults are the package's: metacell_params.yaml for
# k_expand / k_beta / cooling / burn_in / mc_rseed, and the vignette
# (a-basic_pbmc8k.Rmd) for the resampling and balanced-cover arguments.  K is
# the one size-setting knob; 64 matches the K the task10 run was given.
ap.add_argument("--K", type=int, default=64, help="bknn K (vignette: 100)")
ap.add_argument("--k-expand", type=float, default=10)
ap.add_argument("--k-beta", type=float, default=3)
ap.add_argument("--resamp-min-mc-size", type=int, default=20)
ap.add_argument("--p-resamp", type=float, default=0.75)
ap.add_argument("--n-resamp", type=int, default=500)
ap.add_argument("--balanced-K", type=int, default=30, help="project runs: 20")
ap.add_argument("--balanced-alpha", type=float, default=2)
ap.add_argument("--min-mc-size", type=int, default=30, help="project runs: 25")
ap.add_argument("--cooling", type=float, default=1.05)
ap.add_argument("--burn-in", type=int, default=10)
ap.add_argument("--rseed", type=int, default=42)
ap.add_argument("--nproc", type=int, default=int(os.environ.get("NSLOTS", "1")))
ap.add_argument("--coc-cache", default="",
                help="RDS path: reuse/save the stage-2 co-cluster table (parameter sweeps on one pile)")
ap.add_argument("--force", action="store_true")
A = ap.parse_args()

os.makedirs(A.out_dir, exist_ok=True)
out = os.path.join(A.out_dir, f"mc1_pile_{A.pile:05d}_K{A.K}.npz")
if os.path.exists(out) and not A.force:
    log(f"{os.path.basename(out)} exists, skipping (--force to redo)")
    sys.exit(0)

t0 = time.time()
piles = np.load(A.piles)
rows = np.flatnonzero(piles == A.pile)
n = len(rows)
assert n > 0, f"pile {A.pile} is empty"
E = np.load(A.emb, mmap_mode="r")
X = np.ascontiguousarray(E[rows], dtype=np.float32)
D = X.shape[1]
kq = int(min(n - 1, round(A.K * A.k_expand)))
log(f"pile {A.pile}: {n:,} cells x {D} dims; exact k-NN with {kq} neighbours")

# ---- exact Euclidean k-NN, self excluded, in blocks --------------------------------
sq = (X.astype(np.float64) ** 2).sum(1)
nbr = np.empty((n, kq), np.int32)
dst = np.empty((n, kq), np.float32)
Xd = X.astype(np.float64)
for lo in range(0, n, 2048):
    hi = min(lo + 2048, n)
    d2 = sq[lo:hi, None] + sq[None, :] - 2.0 * (Xd[lo:hi] @ Xd.T)
    d2[np.arange(hi - lo), np.arange(lo, hi)] = np.inf          # self
    part = np.argpartition(d2, kq - 1, axis=1)[:, :kq]
    pd_ = np.take_along_axis(d2, part, 1)
    o = np.argsort(pd_, axis=1, kind="stable")
    nbr[lo:hi] = np.take_along_axis(part, o, 1)
    dst[lo:hi] = np.sqrt(np.maximum(np.take_along_axis(pd_, o, 1), 0))
del Xd
assert not (nbr == np.arange(n)[:, None]).any(), "self in its own neighbour list"
log(f"k-NN done ({time.time()-t0:.0f}s)")

# ---- hand off to R in tgs_cor_knn's layout (1-based) --------------------------------
with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
    f = {k: os.path.join(tmp, k + ".bin") for k in ("col1", "col2", "rank", "dist", "out")}
    np.repeat(np.arange(1, n + 1, dtype=np.int32), kq).astype("<i4").tofile(f["col1"])
    (nbr.ravel() + 1).astype("<i4").tofile(f["col2"])
    np.tile(np.arange(1, kq + 1, dtype=np.int32), n).astype("<i4").tofile(f["rank"])
    dst.ravel().astype("<f4").tofile(f["dist"])
    del nbr, dst
    params = {"n": n, "K": A.K, "k_expand": A.k_expand, "k_beta": A.k_beta,
              "resamp_min_mc_size": A.resamp_min_mc_size, "p_resamp": A.p_resamp,
              "n_resamp": A.n_resamp, "balanced_K": A.balanced_K,
              "balanced_alpha": A.balanced_alpha, "min_mc_size": A.min_mc_size,
              "cooling": A.cooling, "burn_in": A.burn_in, "rseed": A.rseed,
              "nproc": A.nproc}
    cmd = [A.rscript, os.path.join(HERE, "step2e_metacell1_tgstat.R")] + \
          [f"--{k}={v}" for k, v in params.items()] + \
          [f"--knn_{k}={f[k]}" for k in ("col1", "col2", "rank", "dist")] + \
          [f"--out={f['out']}"] + \
          ([f"--coc_cache={A.coc_cache}"] if A.coc_cache else [])
    subprocess.run(cmd, check=True)
    cl = np.fromfile(f["out"], dtype="<i4").astype(np.int64)
assert len(cl) == n, f"R returned {len(cl)} labels for {n} cells"

# ---- assigned cells -> base metacells; outliers kept out -----------------------------
keep = cl > 0
M = int(cl.max())
loc = cl[keep] - 1
sz = np.bincount(loc, minlength=M)
assert M > 0 and sz.min() > 0, "empty metacell id from R"
cent = np.zeros((M, D), np.float64)
np.add.at(cent, loc, X[keep].astype(np.float64))
cent /= sz[:, None]
n_out = int((~keep).sum())
log(f"{M:,} metacells | size min {sz.min()} median {int(np.median(sz))} "
    f"mean {sz.mean():.1f} max {sz.max()} | {n_out:,} outliers ({n_out/n*100:.2f}%)")

meta = dict(pile=A.pile, n_cells=int(n), n_assigned=int(keep.sum()), n_outliers=n_out,
            n_metacells=M, knn_neighbours=kq, seconds=round(time.time() - t0, 1),
            method="MetaCell-1 via tgstat (tanaylab/metacell R pipeline), "
                   "exact Euclidean k-NN in the embedding as the bknn input",
            **{k: v for k, v in params.items() if k != "n"})
np.savez_compressed(
    out,
    rows=rows[keep].astype(np.int64),        # FILTERED-cohort rows, assigned cells only
    local=loc.astype(np.int32),
    centroids=cent.astype(np.float32),
    sizes=sz.astype(np.int32),
    outlier_rows=rows[~keep].astype(np.int64),
    meta=np.array(json.dumps(meta)))
log(f"wrote {out}  ({time.time()-t0:.0f}s)")
log("DONE")
