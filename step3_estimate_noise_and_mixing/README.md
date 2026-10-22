# step3 — measurement noise (Poisson CV) and within-metacell mixing (LTV)

Two per-gene statistics, computed on the step1 count matrix under a step2
metacell ladder, in ONE pass because both need the same pooling.

`step3a_cv_ltv_vs_k.py`, copied from `t10/cv_ltv_vs_k_task10.py`.

## The two statistics

Pool cells into metacells, then per gene and per rung k:

**Poisson CV — how precisely a metacell measures the gene.**  A pooled count
`Y ~ Poisson(Lambda)` has `CV = sqrt(Lambda)/Lambda = 1/sqrt(Lambda)`, so
`CV <= 0.25` is exactly `Lambda >= 16` counts.  Reported as the median over
metacells at that rung.

**LTV — how much of the gene's variation survives inside a metacell.**  With
`lamhat_c = X_c/N_c` and exposure weights `w_c = N_c`, the law of total variance
splits the total into a within- and a between-metacell part:

    within + between == total      (asserted, not assumed)

`total` is partition-independent, so `within/total` is comparable across k and
across methods.  Low within/total means the metacell is homogeneous for that
gene; the two cuts used downstream are `CV <= 0.25` and `within/total <= 0.25`.

Outputs per run (`--outtag`): `cv_ltv_curves_<tag>.csv.gz` (per gene x rung:
`median_CV`, `q25/q75`, `median_Lambda`, `frac_pools_Lambda0`, `within`,
`between`, `total`, `lam_bar`, `share_between`), `cv_ltv_diag_<tag>.csv`
(per rung: `n_pools`, size min/median/max, coverage, `ltv_closure_max_rel`),
and a JSON.

**Why this is a fork of the task8 version.**  It differs in exactly one respect:
the cell support may be a STRICT SUBSET of the reconstruction's rows.  A ladder
built on the filtered cohort and lifted to the full one carries `-1` on 28,014
cells; task8 drops `lab < 0` from the incidence matrices but computes `Wtot`,
`lam_bar` and `total` over ALL N rows, so within+between would exclude those
cells while total included them and the closure assertion would fire.  Here the
same mask is applied to the totals via `w_keep = N_c * 1[lab >= 0]`, and all
rungs are asserted to share one support so a single mask is well defined.

**Calibration caveat, left in deliberately.**  `--gene-obs` supplies observed
per-gene sums over the FULL cohort while `gene_rec` is over the masked subset,
so the printed reconstructed/observed ratio is low by the masked fraction
(~0.52%).  That line is reported, never gated, so it is not silently rescaled.

## Drivers — one per (method x embedding)

| labels | scGPT | Harmony |
|---|---|---|
| Leiden (rungs 1-1024) | `cv_ltv_leiden_masked_all_sg.sub` | `cv_ltv_leiden_masked_all_h.sub` |
| SEACells+Ward (to 1024) | `cv_ltv_allcd8sg_seacells_1024.sub` | `cv_ltv_allcd8h_seacells_1024.sub` |
| MetaCell-1+Ward (to 1024) | `cv_ltv_allcd8sg_metacell1_1024.sub` | `cv_ltv_allcd8h_metacell1_1024.sub` |

All of them call the same script with the same inputs except `--labels` and
`--outtag`; the count matrix is `main_effect_counts_scgptpca_perCTclip.npy`
(scGPT) or `main_effect_counts_perCTclip.npy` (Harmony) from step1, with
`--genestats allcd8_genestats.csv` and `--gene-obs observed_hvg_genesum.npy`.
