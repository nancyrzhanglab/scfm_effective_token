#!/usr/bin/env python
"""Harmony on the CD8 PCA, to be compared against the scGPT embedding.

Run on the UMI-based CD8 cohort: 5,353,450 cells, all tissues (blood 76.7%).

Batch key defaults to dataset_id (94 levels here) -- the CELLxGENE census study
identifier, and the analogue of the sample_id used in the earlier three-way
integration.  donor_id (4,707 levels) is available but is biological as well as
technical, and Harmony's per-level correction becomes ill-posed when many levels
hold only a handful of cells.

WHAT IS MEASURED, not asserted: for every PC, the fraction of variance explained
by the batch factor (a one-way ANOVA eta^2) before and after correction.  That
is the quantity Harmony exists to shrink, so reporting it both ways is the
check that the run did something, and how much.  Cell-type eta^2 is reported on
the same axes as a guard against over-correction -- batch structure should fall
while biological structure should not.

Outputs (--out-dir):
    cd8_harmony_<key>.npy        float32 (N, K)  the corrected embedding
    cd8_harmony_<key>_eta2.csv   per PC: batch eta^2 and cell-type eta^2, pre/post
    cd8_harmony_<key>.json
"""
import argparse, json, os, time
import numpy as np, pandas as pd

ROOT = "/home/mnt/weka/nzh/nzhanglab/project/emiliac/FM_effective"
DEF  = os.path.join(ROOT, "results/poisson/task8_scGPTemb2gex")

ap = argparse.ArgumentParser()
ap.add_argument("--in-dir",  default=DEF)
ap.add_argument("--out-dir", default=DEF)
ap.add_argument("--n-comps", type=int, default=50)
ap.add_argument("--batch-key", default="dataset_id",
                help="one key, or several comma-separated -> harmonypy vars_use")
ap.add_argument("--label-key", default="cell_type")
ap.add_argument("--nclust", type=int, default=0, help="0 = harmonypy default min(100, N/30)")
ap.add_argument("--max-iter", type=int, default=10)
ap.add_argument("--theta", type=float, default=None)
ap.add_argument("--prefix", default="cd8", help="output/input filename prefix")
ap.add_argument("--seed", type=int, default=0)
A = ap.parse_args()
t0 = time.time(); log = lambda m: print(f"[{time.time()-t0:8.1f}s] {m}", flush=True)
os.makedirs(A.out_dir, exist_ok=True)
log(f"args: {vars(A)}")

K = A.n_comps
P = np.load(os.path.join(A.in_dir, f"{A.prefix}_pca{K}.npy"))
obs = pd.read_csv(os.path.join(A.in_dir, f"{A.prefix}_obs.csv.gz"), low_memory=False)
N = P.shape[0]
assert len(obs) == N, f"obs has {len(obs)} rows, PCA has {N}"
KEYS = [k.strip() for k in A.batch_key.split(",")]
for k in KEYS:
    assert k in obs.columns, f"'{k}' not in obs"
TAG = "_".join(KEYS)          # filename tag; single key keeps the old names
# eta2 is reported against the COMPOSITE of the keys, so the number means "any
# batch factor" rather than only the first one
obs["_batch"] = obs[KEYS].astype(str).agg("|".join, axis=1) if len(KEYS) > 1 else obs[KEYS[0]]
log(f"{N:,} cells x {K} PCs; batch {KEYS} "
    f"({obs['_batch'].nunique()} composite levels), label '{A.label_key}' "
    f"({obs[A.label_key].nunique()} levels)")
sz = obs["_batch"].value_counts()
log(f"batch sizes: min={sz.min():,} median={int(sz.median()):,} max={sz.max():,}")

def eta2(Y, g):
    """Fraction of each column's variance explained by the grouping g."""
    g = pd.Categorical(g).codes
    out = np.empty(Y.shape[1])
    gm = Y.mean(0)
    tot = ((Y - gm) ** 2).sum(0)
    bet = np.zeros(Y.shape[1])
    for x in np.unique(g):
        m = g == x
        n = int(m.sum())
        bet += n * (Y[m].mean(0) - gm) ** 2
    np.divide(bet, tot, out=out, where=tot > 0)
    return out

Yb = P.astype(np.float64)
pre_b = eta2(Yb, obs["_batch"].to_numpy())
pre_l = eta2(Yb, obs[A.label_key].to_numpy())
log(f"pre-Harmony  batch eta^2: mean={pre_b.mean():.4f} max={pre_b.max():.4f} | "
    f"{A.label_key} eta^2: mean={pre_l.mean():.4f}")
del Yb

import harmonypy
kw = dict(vars_use=KEYS, max_iter_harmony=A.max_iter,
          random_state=A.seed, verbose=True)
if A.nclust: kw["nclust"] = A.nclust
if A.theta is not None: kw["theta"] = A.theta
log(f"running harmonypy {harmonypy.__version__ if hasattr(harmonypy,'__version__') else '0.0.10'} ...")
t1 = time.time()
ho = harmonypy.run_harmony(P.astype(np.float64), obs[KEYS], **kw)
Z = np.ascontiguousarray(np.asarray(ho.Z_corr).T).astype(np.float32)
log(f"harmony finished in {(time.time()-t1)/60:.1f} min -> {Z.shape}")

assert Z.shape == (N, K), f"unexpected corrected shape {Z.shape}"
assert np.isfinite(Z).all(), "non-finite values in the corrected embedding"

Zd = Z.astype(np.float64)
post_b = eta2(Zd, obs["_batch"].to_numpy())
post_l = eta2(Zd, obs[A.label_key].to_numpy())
log(f"post-Harmony batch eta^2: mean={post_b.mean():.4f} max={post_b.max():.4f} | "
    f"{A.label_key} eta^2: mean={post_l.mean():.4f}")
log(f"  batch variance removed: {(1 - post_b.mean()/max(pre_b.mean(),1e-12))*100:.1f}% "
    f"| {A.label_key} signal retained: {post_l.mean()/max(pre_l.mean(),1e-12)*100:.1f}%")
assert post_b.mean() < pre_b.mean(), \
    "Harmony did not reduce batch variance -- the run is not doing its job"

pd.DataFrame(dict(pc=np.arange(1, K+1), batch_eta2_pre=pre_b, batch_eta2_post=post_b,
                  label_eta2_pre=pre_l, label_eta2_post=post_l)).to_csv(
    os.path.join(A.out_dir, f"{A.prefix}_harmony_{TAG}_eta2.csv"), index=False)
np.save(os.path.join(A.out_dir, f"{A.prefix}_harmony_{TAG}.npy"), Z)
json.dump(dict(n_cells=int(N), n_comps=int(K), batch_key=A.batch_key,
               batch_keys=KEYS,
               label_key=A.label_key, n_batches=int(obs["_batch"].nunique()),
               nclust=int(A.nclust) or None, max_iter=A.max_iter, seed=A.seed,
               batch_eta2_pre=float(pre_b.mean()), batch_eta2_post=float(post_b.mean()),
               label_eta2_pre=float(pre_l.mean()), label_eta2_post=float(post_l.mean()),
               harmony_minutes=float((time.time()-t1)/60)),
          open(os.path.join(A.out_dir, f"{A.prefix}_harmony_{TAG}.json"), "w"), indent=2)
log("HARMONY OK")
