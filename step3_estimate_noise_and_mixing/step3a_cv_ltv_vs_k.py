#!/usr/bin/env python
"""Measurement noise (Poisson CV) and mixing (LTV) vs metacell size k.


  MEASUREMENT NOISE      X_cg ~ Poisson(N_c lambda_cg)
                         Mu_j = sum_{c in j} mu_cg,   N_j = sum_{c in j} N_c
                         CV = 1 / sqrt(Mu_j)                       -- falls with k

  MIXING         law of total variance on lam_cg = mu_cg / N_c for each gene g
                 total   = sum_c (lam_cg - lam_bar)^2 / C
                 within  = sum_j sum_{c in j} (lam_cg - lam_j)^2 / C
                 between = sum_j C_j (lam_j - lam_bar)^2 / C,  C_j = cells in j
                 within + between == total is ASSERTED

N_c is the reconstructed HVG total
(after per-cell renormalisation that equals the observed HVG total).


Outputs (--out-dir):
    cv_ltv_curves_<outtag>.csv.gz    gene x k: CV quantiles; the w_c = 1 LTV as
                                     within1 / between1 / total1 / lam_bar1 /
                                     share_between1; lam_bar
    cv_ltv_diag_<outtag>.csv         per k: pools, sizes, coverage, LTV closure
    cv_ltv_<outtag>.json

Column names follow cv_ltv_weighting_compare.py, where the suffix 1 marks the
w_c = 1 split and the unsuffixed within/between/total/share_between are the
N_c-weighted one.  Only the w_c = 1 split is computed here, so only the
suffixed columns are written: a reader asking for share_between1 gets this
split from either script, and a weighted curves file cannot be read as an
unweighted one by mistake.

lam_bar is the gene's POOLED cohort rate, sum_c mu_cg / sum_c N_c over the
labelled cells -- the per-gene expression level the expression deciles are cut
on (step4c-e), identical to lam_bar in cv_ltv_weighting_compare.py.  It is not
lam_bar1, the unweighted mean of the per-cell rates that the LTV centres on.
"""
import argparse, json, os, time
import numpy as np, pandas as pd
from scipy.sparse import csr_matrix

# parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("--in-dir", required=True)
ap.add_argument("--out-dir", required=True)
ap.add_argument("--counts", default="main_effect_counts_perCTclip.npy")
ap.add_argument("--observed", default="observed_counts_hvg.npy")
ap.add_argument("--gene-obs", default="", help="precomputed (G,) observed per-gene count sums")
ap.add_argument("--genestats", default="normal_genestats.csv")
ap.add_argument("--labels", default="metacell_labels_normal_leiden.npz")
ap.add_argument("--keyfmt", default="lab_mean{k}_leiden_s0")
ap.add_argument("--ks", default="2,4,8,16,32,64,128,256,512,1024")
ap.add_argument("--gene-block", type=int, default=200)
ap.add_argument("--outtag", default="normal_me")
A = ap.parse_args()
t0=time.time(); log=lambda m: print(f"[{time.time()-t0:8.1f}s] {m}", flush=True)
os.makedirs(A.out_dir, exist_ok=True); log(f"args: {vars(A)}")

# read GEX data
mu  = np.load(os.path.join(A.in_dir, A.counts),   mmap_mode="r")
genes = pd.read_csv(os.path.join(A.in_dir, A.genestats)).gene.to_numpy()
N, G = mu.shape
assert len(genes) == G

# The observed matrix is used ONLY for comparing recovered gene expression from CellANOVA and the observed count
if A.gene_obs:
    Yob = None
    gene_obs_pre = np.load(os.path.join(A.in_dir, A.gene_obs)).astype(np.float64)
    assert gene_obs_pre.shape == (G,), f"--gene-obs has shape {gene_obs_pre.shape}, want ({G},)"
else:
    Yob = np.load(os.path.join(A.in_dir, A.observed), mmap_mode="r")
    gene_obs_pre = None
    assert Yob.shape == (N, G)

# read metacell assignment
KS = [int(x) for x in A.ks.split(",")]
Z = np.load(os.path.join(A.in_dir, A.labels))
labs = {}
for k in KS:
    key = A.keyfmt.format(k=k)
    assert key in Z.files, f"{key} not in {A.labels}: {Z.files[:4]}..."
    labs[k] = Z[key].astype(np.int64)
    assert labs[k].shape == (N,), f"{key}: {labs[k].shape} vs {N:,} rows"
keep = labs[KS[0]] >= 0
for k in KS:
    assert np.array_equal(labs[k] >= 0, keep), \
        f"k={k} has a different cell support; a single mask is not well defined"
log(f"{N:,} cells x {G} genes; rungs {KS}; "
    f"{int(keep.sum()):,} cells labelled ({100*keep.mean():.2f}%)")

# Compute library size per cell from recovered HVG total (N_c)
N_c = np.zeros(N)
RB = [(i, min(i+100_000, N)) for i in range(0, N, 100_000)]

# sum by block to avoid materialising all data in RAM
for lo,hi in RB: N_c[lo:hi] = np.asarray(mu[lo:hi], np.float64).sum(1)

# Check cells to use (i.e LABELED) all have a positive N_c. 
assert (N_c[keep] > 0).all(), \
    f"{int((N_c[keep] <= 0).sum())} LABELLED cells have zero reconstructed total"
if not keep.all():
    log(f"{int((N_c[~keep] <= 0).sum()):,} of {int((~keep).sum()):,} unlabelled "
        f"cells have zero exposure (not used)")
log(f"N_c median {np.median(N_c):.0f}  range {N_c.min():.0f}-{N_c.max():.0f}")

# per-k incidence matrices.  Sk is the 0/1 membership used for the pooled count
# Mu_j; Skw carries the LTV cell weights.  Those weights are w_c = 1 -- one vote
# per labelled cell -- so here Skw equals Sk and Wj is the pool's cell count.

Sk, Skw, Wj, m_of, sizes_of = {}, {}, {}, {}, {}
for k in KS:
    lab = labs[k]; ok = lab >= 0; idx = np.flatnonzero(ok); m = int(lab.max())+1
    Sk[k]  = csr_matrix((np.ones(idx.size), (lab[idx], idx)), shape=(m, N))
    Skw[k] = csr_matrix((np.ones(idx.size), (lab[idx], idx)), shape=(m, N))
    Wj[k]  = np.asarray(Skw[k] @ np.ones(N)); m_of[k] = m
    sizes_of[k] = np.bincount(lab[ok], minlength=m)
    assert (Wj[k] > 0).all(), f"k={k}: a pool has no labelled cell"
log("incidence matrices built: " + ", ".join(f"k={k}:{m_of[k]:,}" for k in KS))


w_keep = keep.astype(np.float64)
N_safe = np.where(N_c > 0, N_c, 1.0)
C = w_keep.sum()          # labelled cells; the docstring's C
log(f"LTV weighting: w_c = 1 over {int(C):,} labelled cells "
    f"({100*C/N:.2f}% of rows; they hold "
    f"{100*N_c[keep].sum()/N_c.sum():.2f}% of the exposure)")
acc = {k: dict(medCV=[], q25=[], q75=[], medMu=[], fz=[],
               wi=[], be=[], to=[], lb=[]) for k in KS}
gene_rec = np.zeros(G)
gene_obs = gene_obs_pre.copy() if gene_obs_pre is not None else np.zeros(G)
BLK = [(i, min(i+A.gene_block, G)) for i in range(0, G, A.gene_block)]
for bi,(g0,g1) in enumerate(BLK):
    M = np.asarray(mu[:, g0:g1], np.float64)               # (N, nb)
    gene_rec[g0:g1] = M[keep].sum(0)
    if Yob is not None:
        O = np.asarray(Yob[:, g0:g1], np.float64)
        gene_obs[g0:g1] = O.sum(0); del O

    lam = M / N_safe[:, None]
    for k in KS:
        Mu = np.asarray(Sk[k] @ M)                        # (m, nb)

        # Measurement noise quantification
        with np.errstate(divide="ignore", invalid="ignore"):
            CV = np.where(Mu > 0, 1.0/np.sqrt(Mu), np.nan)

        q25c, medc, q75c = np.nanpercentile(CV, [25, 50, 75], axis=0)
        acc[k]["medCV"].append(medc)
        acc[k]["q25"].append(q25c)
        acc[k]["q75"].append(q75c)
        acc[k]["medMu"].append(np.median(Mu, axis=0))
        acc[k]["fz"].append((Mu == 0).mean(axis=0)); del Mu, CV

        # Mixing quantification
        C_j     = Wj[k][:, None]                   # cells in each metacell
        lam_j   = np.asarray(Skw[k] @ lam) / C_j   # mean rate within a metacell
        lam_bar = (w_keep[:, None] * lam).sum(0) / C     # mean rate overall
        sumsq_j = np.asarray(Skw[k] @ (lam ** 2))  # sum of lam^2 in a metacell

        # within is, metacell by metacell, the spread of its own cells' rates
        # about its own mean rate:
        # for each metacell j:  sum_{c in j} (lam_cg - lam_j)^2
        # then sum over j and divide by C
        # which is what an explicit loop over metacells would compute.
        within  = (sumsq_j - C_j * lam_j ** 2).sum(0) / C

        # between is the spread of metacell mean around the total mean
        between = (C_j * (lam_j - lam_bar) ** 2).sum(0) / C

        # total is the spread of all cells around the total mean
        total   = (w_keep[:, None] * (lam - lam_bar) ** 2).sum(0) / C

        acc[k]["wi"].append(within)
        acc[k]["be"].append(between)
        acc[k]["to"].append(total)
        acc[k]["lb"].append(lam_bar)
        del lam_j, sumsq_j, within, between, total
    del M, lam
    if bi % 3 == 0: log(f"  gene block {bi+1}/{len(BLK)}")

# Assemble and write results
# pooled cohort rate per gene over the labelled cells (the expression level the
# deciles are cut on; see the docstring) -- the same at every k
lam_pooled = gene_rec / N_c[keep].sum()
rows=[]; diag=[]
for k in KS:
    d = {p: np.concatenate(v) for p,v in acc[k].items()}
    clo = np.abs(d["wi"] + d["be"] - d["to"]) / np.maximum(d["to"], 1e-300)
    assert np.nanmax(clo) < 1e-6, f"k={k}: within+between != total (max {np.nanmax(clo):.3g})"
    rows.append(pd.DataFrame(dict(gene=genes, k=k,
        median_CV=d["medCV"], q25_CV=d["q25"], q75_CV=d["q75"],
        median_Mu=d["medMu"], frac_pools_Mu0=d["fz"],
        within1=d["wi"], between1=d["be"], total1=d["to"], lam_bar1=d["lb"],
        share_between1=np.where(d["to"]>0, d["be"]/np.maximum(d["to"],1e-300), np.nan),
        lam_bar=lam_pooled)))
    s = sizes_of[k]
    diag.append(dict(k=k, n_pools=m_of[k], size_min=int(s.min()),
                     size_median=float(np.median(s)), size_max=int(s.max()),
                     size_mean=float(s.mean()), coverage=float((labs[k]>=0).mean()),
                     ltv_closure_max_rel=float(np.nanmax(clo)),
                     median_CV_over_genes=float(np.nanmedian(d["medCV"])),
                     median_share_between1=float(np.nanmedian(
                         np.where(d["to"]>0, d["be"]/np.maximum(d["to"],1e-300), np.nan)))))
    log(f"k={k:4d}: {m_of[k]:8,} pools | median CV {np.nanmedian(d['medCV']):.4f} | "
        f"median between/total {diag[-1]['median_share_between1']:.4f} | "
        f"LTV closure {np.nanmax(clo):.2g}")

# Mu is the internal name (Mu_j = sum_{c in j} mu_cg, a pooled COUNT; lambda is
# the rate).  The COLUMN names on disk keep their original spelling, because
# every published curve file uses them and the readers accept both -- so the
# vocabulary change stops at the file boundary, in one place, here.
_INTERNAL2DISK = {"median_Mu": "median_Lambda",
                  "frac_pools_Mu0": "frac_pools_Lambda0"}
cur = pd.concat(rows, ignore_index=True).rename(columns=_INTERNAL2DISK)
cur.to_csv(os.path.join(A.out_dir, f"cv_ltv_curves_{A.outtag}.csv.gz"),
           index=False, compression="gzip")
pd.DataFrame(diag).to_csv(os.path.join(A.out_dir, f"cv_ltv_diag_{A.outtag}.csv"), index=False)

# calibration check to compare the scale of the reconstructed gene expression counts (mu) and the observed counts
ratio = gene_rec/np.maximum(gene_obs,1e-12)
log(f"CALIBRATION: median per-gene recon/observed = {np.median(ratio):.3f}. ")

json.dump(dict(n_cells=int(N), n_genes=int(G), ks=KS, outtag=A.outtag,
               counts=A.counts, labels=A.labels,
               observed_source=(A.gene_obs or A.observed),
               median_gene_calibration=float(np.median(ratio)),
               diag=diag),
          open(os.path.join(A.out_dir, f"cv_ltv_{A.outtag}.json"), "w"), indent=2)
log("CV/LTV OK")
