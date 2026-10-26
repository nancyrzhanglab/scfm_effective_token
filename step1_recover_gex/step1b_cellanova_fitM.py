#!/usr/bin/env python
"""CellANOVA fit_M 

Regress expression on the cell embedding for each dataset to obtain a regression matrix
to convert between embedding and expression.

Then take an average of the regression matrix across dataset to obtain Mmat which represent
the main effect of cell state on gene expression.

Reimplementation of CellANOVA to only compute the Main Effect

"""
import argparse, json, os, time
import h5py, numpy as np, pandas as pd
from scipy.sparse import csr_matrix

ap = argparse.ArgumentParser()
ap.add_argument("--h5ad", required=True)
ap.add_argument("--in-dir", required=True)
ap.add_argument("--out-dir", required=True)
ap.add_argument("--prefix", default="allcd8")
ap.add_argument("--embedding", required=True)
ap.add_argument("--tag", default="harmony")
ap.add_argument("--integrate-key", default="dataset_id")
ap.add_argument("--groups", required=True)
ap.add_argument("--target-sum", type=float, default=1e4)
ap.add_argument("--block", type=int, default=100_000)
ap.add_argument("--min-cells", type=int, default=0)
ap.add_argument("--min-group", type=int, default=50)
ap.add_argument("--validate", default="")
a = ap.parse_args()
t0 = time.time(); log = lambda m: print(f"[{time.time()-t0:8.1f}s] {m}", flush=True)

# read data
gs = pd.read_csv(os.path.join(a.in_dir, f"{a.prefix}_genestats.csv"))
mu_g = gs.ln_mean.to_numpy(np.float64); sd_g = gs.ln_std.to_numpy(np.float64)
sd_safe = np.where(sd_g > 0, sd_g, 1.0)
L_c = np.load(os.path.join(a.in_dir, f"{a.prefix}_libsize.npy")).astype(np.float64)
C = np.load(a.embedding).astype(np.float64)
obs = pd.read_csv(os.path.join(a.in_dir, f"{a.prefix}_obs.csv.gz"),
                  usecols=[a.integrate_key], low_memory=False)
lev = pd.Categorical(obs[a.integrate_key]).codes.astype(np.int64)
levels_all = np.unique(lev); Ln = len(levels_all)
gcode = np.load(a.groups).astype(np.int64)
n_grp = int(gcode.max()) + 1
N, d, G = C.shape[0], C.shape[1], len(mu_g)
assert len(L_c) == N == len(lev) == len(gcode)
log(f"{N:,} cells x {G} genes, d={d}, {Ln} levels of {a.integrate_key}, {n_grp} clip groups")

# compute cell embedding matrix inner product, cross-product between embedding and normalized expression,
# observed cell total, gene-specific ceiling, and integration level
Gxx = np.zeros((Ln, d, d)); Pxy = np.zeros((Ln, d, G))
obs_tot = np.zeros(N); ceil_tab = np.zeros((n_grp, G))
lev_n = np.bincount(lev, minlength=Ln)

with h5py.File(a.h5ad, "r") as f:
    ip = f["X/indptr"][:]
    for lo in range(0, N, a.block):
        hi = min(lo + a.block, N)
        p, q = int(ip[lo]), int(ip[hi])
        X = csr_matrix((f["X/data"][p:q], f["X/indices"][p:q],
                        ip[lo:hi+1] - ip[lo]), shape=(hi-lo, G)).toarray().astype(np.float64)
        obs_tot[lo:hi] = X.sum(1)
        ln = np.log1p(X / L_c[lo:hi, None] * a.target_sum); del X
        gh = gcode[lo:hi]
        for gg in np.unique(gh):
            np.maximum(ceil_tab[gg], ln[gh == gg].max(0), out=ceil_tab[gg])
        Sb = (ln - mu_g[None, :]) / sd_safe[None, :]; del ln
        Cb = C[lo:hi]; lb = lev[lo:hi]
        for x in np.unique(lb):
            m = lb == x
            Cm = Cb[m]
            Gxx[x] += Cm.T @ Cm
            Pxy[x] += Cm.T @ Sb[m]
        del Sb
        if (lo // a.block) % 5 == 0: log(f"  {hi:,}/{N:,}")
log("accumulation done")

# least-squares fit between expression and embedding
small = np.flatnonzero(lev_n < a.min_cells) if a.min_cells else np.array([], int)
B = np.zeros((Ln, d, G)); rank = np.zeros(Ln, int); cond = np.zeros(Ln)
for x in range(Ln):
    w = np.linalg.eigvalsh(Gxx[x])
    rank[x] = int((w > max(w.max(), 0) * 1e-10).sum()) if w.max() > 0 else 0
    cond[x] = float(w.max() / max(w[w > 0].min(), 1e-300)) if rank[x] else np.inf
    B[x] = np.linalg.pinv(Gxx[x], rcond=1e-10) @ Pxy[x]
bad = set(np.flatnonzero(rank < d).tolist()) | set(small.tolist())
good = np.array([x for x in range(Ln) if x not in bad])
assert len(good) > 0, "every level is rank-deficient"

# compute Mmat by taking average across regression matrices
Mmat = B[good].mean(0)
log(f"Mmat = mean over {len(good)} of {Ln} levels "
    f"({int((rank<d).sum())} rank-deficient, {len(small)} below --min-cells)")

if a.validate:
    ref = np.load(a.validate).astype(np.float64)
    rel = np.abs(Mmat - ref).max() / max(np.abs(ref).max(), 1e-300)
    log(f"VALIDATE against {os.path.basename(a.validate)}: max rel diff {rel:.3g}")
    assert rel < 1e-6, "streamed fit_M disagrees with the reference Mmat"

# make sure the gene-and-group-specific ceiling is not taken on only a few cells
grp_n = np.bincount(gcode, minlength=n_grp)
sm = np.flatnonzero(grp_n < a.min_group)
if sm.size:
    ceil_tab[sm] = ceil_tab.max(0)
    log(f"small clip groups (< {a.min_group} cells): {sm.size} of {n_grp} reset to "
        f"the global per-gene ceiling; sizes {grp_n[sm].tolist()}")
else:
    log(f"all {n_grp} clip groups >= {a.min_group} cells; no ceiling fallback needed")

# write output
O = a.out_dir
np.save(os.path.join(O, f"{a.prefix}_{a.tag}_Mmat_{a.integrate_key}.npy"), Mmat)
np.save(os.path.join(O, f"{a.prefix}_ceiling_table.npy"), ceil_tab)
np.save(os.path.join(O, "observed_hvg_libsize.npy"), obs_tot)
dev = [float(np.linalg.norm(C[lev == x] @ (B[x] - Mmat)) /
             max(np.linalg.norm(C[lev == x] @ Mmat), 1e-300)) for x in good]
pd.DataFrame(dict(level=good, n_cells=lev_n[good], rank=rank[good],
                  cond=cond[good], dev_ratio=dev)).sort_values(
    "dev_ratio", ascending=False).to_csv(
    os.path.join(O, f"{a.prefix}_{a.tag}_persample_{a.integrate_key}.csv"), index=False)
json.dump(dict(n_cells=int(N), d=int(d), n_genes=int(G), n_levels=int(Ln),
               n_good=int(len(good)), rank_deficient=int((rank < d).sum()),
               min_cells=a.min_cells, integrate_key=a.integrate_key,
               median_dev_ratio=float(np.median(dev)), route="streamed"),
          open(os.path.join(O, f"{a.prefix}_{a.tag}_main_effect_{a.integrate_key}.json"), "w"),
          indent=2)
log("FITM STREAM OK")
