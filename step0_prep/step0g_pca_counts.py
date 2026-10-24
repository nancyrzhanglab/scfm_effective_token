#!/usr/bin/env python
"""PCA on the CELLxGENE CD8 HVG matrix, streamed.


The z-scored matrix would be ~60 GiB dense, so it is never materialised:
because log1p(0) = 0, the lognorm layer has EXACTLY the counts' sparsity, and
the covariance of the z-scored data follows in closed form from two sparse
accumulations,

    G2 = Ln^T Ln          (sparse x sparse -> 3000 x 3000)
    s1 = colsum(Ln)
    cov = (G2 - n mu mu^T) / (n - 1),   corr = cov / (sd sd^T)

`corr` IS the covariance of the z-scored matrix, so its top eigenvectors are
exactly the PCA loadings that sc.pp.scale + sc.pp.pca would produce.  Genes
(3000) are far fewer than cells, so eigh on 3000 x 3000 is exact and cheap.

NORMALISER.  obs['raw_sum'] is the cell's total over ALL measured genes in the
census, not just these 3000 HVGs.  That is the true library size and is what is
used.  sc.pp.normalize_total on this file would instead divide by the HVG-only
row sum, which is a different (and smaller) quantity; both are recorded and the
ratio reported, so the choice is visible rather than implicit.

Outputs (--out-dir):
    cd8_pca<K>.npy           float32 (N, K)   PCA scores, input to Harmony
    cd8_pca_loadings.npy     float64 (3000, K)
    cd8_genestats.csv        gene, ln_mean, ln_std, hvg_colsum
    cd8_libsize.npy          float64 (N,)     raw_sum used as normaliser
    cd8_obs.csv.gz           per-cell metadata (batch keys + labels)
    cd8_pca.json
"""
import argparse, json, os, time
import h5py, numpy as np, pandas as pd
from scipy.sparse import csr_matrix

ROOT = "/home/mnt/weka/nzh/nzhanglab/project/emiliac/FM_effective"

ap = argparse.ArgumentParser()
ap.add_argument("--h5ad", default=os.path.join(ROOT, "data/scGPT_related/mat/CD8T/CD8T_blood_hvg.h5ad"))
ap.add_argument("--out-dir", default=os.path.join(ROOT, "results/poisson/task8_scGPTemb2gex"))
ap.add_argument("--n-comps", type=int, default=50)
ap.add_argument("--target-sum", type=float, default=1e4)
ap.add_argument("--block", type=int, default=100_000)
ap.add_argument("--normaliser", default="raw_sum", choices=["raw_sum", "hvg_sum"])
ap.add_argument("--verify-cells", type=int, default=20000)
ap.add_argument("--prefix", default="cd8", help="output filename prefix")
ap.add_argument("--gene-ids", default="", help="optional sidecar of gene ids, one per "
                "line, in column order.  The CELLxGENE extract stores a POSITIONAL "
                "var index ('0','1',...), so without this the gene column is "
                "meaningless -- the same defect that had to be repaired after the "
                "first full-atlas run.")
ap.add_argument("--seed", type=int, default=0)
A = ap.parse_args()
t0 = time.time(); log = lambda m: print(f"[{time.time()-t0:8.1f}s] {m}", flush=True)
os.makedirs(A.out_dir, exist_ok=True)
log(f"args: {vars(A)}")

f = h5py.File(A.h5ad, "r")
Xp, Xi, Xd = f["X/indptr"], f["X/indices"], f["X/data"]
N = Xp.shape[0] - 1
vkey = f["var"].attrs.get("_index", "_index")
vkey = vkey.decode() if isinstance(vkey, bytes) else vkey
G = f["var"][vkey].shape[0]
indptr = Xp[:]
genes = np.array([x.decode() if isinstance(x, bytes) else str(x)
                  for x in f["var"][vkey][:]])
if A.gene_ids:
    sidecar = np.array([l.strip() for l in open(A.gene_ids)])
    assert len(sidecar) == G, f"sidecar has {len(sidecar)} ids, matrix has {G} columns"
    log(f"gene names from sidecar (h5ad var index was e.g. {list(genes[:3])})")
    genes = sidecar
log(f"{N:,} cells x {G:,} HVGs, {Xd.shape[0]:,} nonzeros")

raw_sum = f["obs/raw_sum"][:].astype(np.float64)
BL = [(i, min(i + A.block, N)) for i in range(0, N, A.block)]

def read_block(lo, hi):
    a, b = int(indptr[lo]), int(indptr[hi])
    return csr_matrix((Xd[a:b].astype(np.float64), Xi[a:b],
                       indptr[lo:hi+1] - indptr[lo]), shape=(hi - lo, G))

# ---------------------------------------------------------------- pass 1
log("pass 1: HVG row sums + lognorm moments (all sparse) ...")
hvg_sum = np.zeros(N, np.float64)
s1 = np.zeros(G, np.float64)
G2 = np.zeros((G, G), np.float64)
for k, (lo, hi) in enumerate(BL):
    Xb = read_block(lo, hi)
    hvg_sum[lo:hi] = np.asarray(Xb.sum(1)).ravel()
    L = raw_sum[lo:hi] if A.normaliser == "raw_sum" else hvg_sum[lo:hi]
    inv = np.where(L > 0, A.target_sum / np.maximum(L, 1), 0.0)
    Ln = Xb.multiply(inv[:, None]).tocsr()
    Ln.data = np.log1p(Ln.data)                    # log1p(0)=0 keeps sparsity
    s1 += np.asarray(Ln.sum(0)).ravel()
    G2 += (Ln.T @ Ln).toarray()
    if k % 5 == 0: log(f"   block {k+1}/{len(BL)}")

mu = s1 / N
cov = (G2 - N * np.outer(mu, mu)) / (N - 1)
sd = np.sqrt(np.maximum(np.diag(cov), 0))
log(f"lognorm: mean in [{mu.min():.4f},{mu.max():.4f}], sd in [{sd.min():.4g},{sd.max():.4g}]")
# A gene can be all-zero in a SUBSET while varying in the parent (the extract
# keeps such columns on purpose, to preserve correspondence).  They carry no
# information and cannot be z-scored, so they are excluded from the covariance
# and given zero loadings -- the column is kept, its contribution is zero.
dead = sd <= 0
if dead.any():
    log(f"{int(dead.sum())} of {G} genes have zero variance here "
        f"({list(genes[dead][:6])}); excluded from the eigendecomposition and "
        f"given zero loadings, columns retained")
alive = ~dead
log(f"normaliser={A.normaliser}; raw_sum/hvg_sum ratio median="
    f"{np.median(raw_sum/np.maximum(hvg_sum,1)):.2f} "
    f"(raw_sum covers all measured genes, hvg_sum only these {G})")

sd_a, mu_a = sd[alive], mu[alive]
corr = cov[np.ix_(alive, alive)] / np.outer(sd_a, sd_a)
np.fill_diagonal(corr, 1.0)

# ---- gate 1: streamed moments match scanpy on a subsample --------------------
import scanpy as sc, anndata as ad
from scipy.sparse import vstack
rng = np.random.default_rng(A.seed)
sub = np.sort(rng.choice(N, size=min(A.verify_cells, N), replace=False))
rows = []
for c in sub:
    a, b = int(indptr[c]), int(indptr[c+1])
    rows.append(csr_matrix((Xd[a:b].astype(np.float32), Xi[a:b], np.array([0, b-a])), shape=(1, G)))
Xs = vstack(rows).tocsr()
As = ad.AnnData(X=Xs.copy())                        # copy: normalize_total mutates in place
Ls = raw_sum[sub] if A.normaliser == "raw_sum" else np.asarray(Xs.sum(1)).ravel()
As.X = csr_matrix(As.X.multiply((A.target_sum / np.maximum(Ls, 1))[:, None]))
sc.pp.log1p(As)
Dn = np.asarray(As.X.todense(), np.float64)
g_mu, g_sd = Dn.mean(0), Dn.std(0, ddof=1)
# our streamed mu/sd are over ALL cells, so compare the machinery on these cells
n_s = len(sub)
s1s = np.asarray(As.X.sum(0)).ravel(); mus = s1s / n_s
G2s = (As.X.T @ As.X).toarray()
covs = (G2s - n_s * np.outer(mus, mus)) / (n_s - 1)
d_mu = float(np.abs(mus - g_mu).max())
d_sd = float(np.abs(np.sqrt(np.maximum(np.diag(covs), 0)) - g_sd).max())
log(f"gate 1: sparse-moment route matches dense numpy on {n_s} cells -- "
    f"mean maxdiff {d_mu:.3g}, sd maxdiff {d_sd:.3g}")
assert d_mu < 1e-8 and d_sd < 1e-6, "the sparse moment identity is wrong"

# ---------------------------------------------------------------- eigen
K = A.n_comps
ev, V = np.linalg.eigh(corr)                        # ascending
order = np.argsort(ev)[::-1][:K]
lam = ev[order]
Vk = np.zeros((G, K))                              # zero loadings for dead genes
Vk[alive] = V[:, order]
# make the dead columns contribute exactly 0 to the projection: their values are
# all zero, so mu=0 / sd=1 sends them to z=0
mu = mu.copy(); sd = sd.copy()
mu[dead] = 0.0; sd[dead] = 1.0
assert (lam > 0).all(), "non-positive leading eigenvalue"
tot = float(np.trace(corr))   # over the live genes only
log(f"top {K} eigenvalues explain {lam.sum()/tot*100:.1f}% of total variance "
    f"(lam1={lam[0]:.2f} .. lam{K}={lam[-1]:.3f})")

# ---- gate 2: agrees with sklearn PCA on the subsample ------------------------
from sklearn.decomposition import PCA
# A gene can be all-zero among these few thousand cells even though it varies
# across the full 4.13M (production sd > 0 is asserted separately).  Such genes
# give g_sd = 0 and would make Zs all-NaN, so the gate is run on the genes that
# are actually informative in the subsample.
ok = g_sd > 0
nk = min(K, 40, int(ok.sum()) - 1, len(sub) - 1)
log(f"gate 2: {int(ok.sum())} of {G} genes vary in the subsample; comparing {nk} components")
Zs = (Dn[:, ok] - g_mu[ok]) / g_sd[ok]
assert np.isfinite(Zs).all(), "subsample z-scores still non-finite after masking"
p = PCA(n_components=nk, svd_solver="full").fit(Zs)
covs_z = np.cov(Zs, rowvar=False)
ev2, V2 = np.linalg.eigh(covs_z)
o2 = np.argsort(ev2)[::-1][:nk]
rel = np.abs(ev2[o2] - p.explained_variance_).max() / p.explained_variance_.max()
log(f"gate 2: eigh(cov) reproduces sklearn PCA's explained variance on the "
    f"subsample, max rel err {rel:.3g}")
assert rel < 1e-6, "the eigendecomposition route disagrees with sklearn PCA"

# ---------------------------------------------------------------- pass 2
log("pass 2: projecting ...")
P = np.empty((N, K), np.float32)
inv_sd = 1.0 / sd
for k, (lo, hi) in enumerate(BL):
    Xb = read_block(lo, hi)
    L = raw_sum[lo:hi] if A.normaliser == "raw_sum" else hvg_sum[lo:hi]
    inv = np.where(L > 0, A.target_sum / np.maximum(L, 1), 0.0)
    Ln = Xb.multiply(inv[:, None]).tocsr(); Ln.data = np.log1p(Ln.data)
    D = np.asarray(Ln.todense(), np.float64)
    D -= mu; D *= inv_sd
    P[lo:hi] = (D @ Vk).astype(np.float32)
    if k % 5 == 0: log(f"   block {k+1}/{len(BL)}")

# ---- gate 3: scores carry the eigenvalues -----------------------------------
emp = P.astype(np.float64).var(0, ddof=1)
rel3 = float(np.abs(emp - lam).max() / lam.max())
log(f"gate 3: empirical variance of the scores matches the eigenvalues, "
    f"max rel err {rel3:.3g}")
assert rel3 < 1e-3, "projected scores do not carry the expected variance"
# Orthogonality holds over ALL cells, not over any subset.  Cells here are
# ordered by soma_joinid, i.e. grouped by dataset, so a CONTIGUOUS slice is a
# biased sample in which the components really are correlated (0.364 on the
# all-tissue file vs 1.9e-10 over all cells).  Accumulate the full Gram in
# blocks -- it is only 50x50 -- and report a random slice alongside it.
gram = np.zeros((K, K)); nacc = 0
for lo in range(0, N, A.block):
    B = P[lo:lo + A.block].astype(np.float64)
    gram += B.T @ B; nacc += B.shape[0]
gram /= (nacc - 1)
offdiag = float(np.abs(gram - np.diag(np.diag(gram))).max() / np.diag(gram).max())
_r = np.random.default_rng(A.seed).choice(N, min(200_000, N), replace=False)
_o = P[np.sort(_r)].astype(np.float64)
_g = _o.T @ _o / (_o.shape[0] - 1)
offdiag_sub = float(np.abs(_g - np.diag(np.diag(_g))).max() / np.diag(_g).max())
log(f"        component orthogonality, ALL {N:,} cells: max |offdiag|/lam1 = {offdiag:.3g} "
    f"(random 200k subsample: {offdiag_sub:.3g} -- subsample noise, not a bound)")
assert offdiag < 1e-6, f"scores are not orthogonal over all cells: {offdiag}"

# ---------------------------------------------------------------- save
np.save(os.path.join(A.out_dir, f"{A.prefix}_pca{K}.npy"), P)
np.save(os.path.join(A.out_dir, f"{A.prefix}_pca_loadings.npy"), Vk)
np.save(os.path.join(A.out_dir, f"{A.prefix}_libsize.npy"), raw_sum)
pd.DataFrame(dict(gene=genes, ln_mean=mu, ln_std=sd, hvg_colsum=s1,
                  zero_variance=dead)).to_csv(
    os.path.join(A.out_dir, f"{A.prefix}_genestats.csv"), index=False)

OBS = ["dataset_id","assay","donor_id","cell_type","disease","sex","tissue",
       "tissue_general","suspension_type","development_stage",
       "self_reported_ethnicity","is_primary_data","nnz","n_measured_vars","raw_sum"]
cols = {}
for k in OBS:
    if k not in f["obs"]: continue
    o = f["obs"][k]
    if isinstance(o, h5py.Group) and "categories" in o:
        cc = [x.decode() if isinstance(x, bytes) else x for x in o["categories"][:]]
        cols[k] = pd.Categorical.from_codes(o["codes"][:], categories=cc)
    else:
        cols[k] = o[:]
obs = pd.DataFrame(cols); obs.insert(0, "cell_id", np.arange(N))
obs["hvg_sum"] = hvg_sum
obs.to_csv(os.path.join(A.out_dir, f"{A.prefix}_obs.csv.gz"), index=False, compression="gzip")
log(f"obs saved {obs.shape}; dataset_id={obs.dataset_id.nunique()} "
    f"donor_id={obs.donor_id.nunique()} assay={obs.assay.nunique()} "
    f"cell_type={obs.cell_type.nunique()}")

json.dump(dict(n_cells=int(N), n_genes=int(G), n_comps=int(K),
               normaliser=A.normaliser, target_sum=A.target_sum,
               var_explained_frac=float(lam.sum()/tot),
               eigenvalues=[float(x) for x in lam],
               gate1_mean_maxdiff=d_mu, gate1_sd_maxdiff=d_sd,
               gate2_sklearn_rel=float(rel), gate3_score_var_rel=float(rel3),
               gate3_offdiag=float(offdiag),
               n_zero_variance_genes=int(dead.sum()),
               n_dataset_id=int(obs.dataset_id.nunique()),
               n_donor_id=int(obs.donor_id.nunique()), seed=A.seed),
          open(os.path.join(A.out_dir, f"{A.prefix}_pca.json"), "w"), indent=2)
f.close()
log("PCA OK")
