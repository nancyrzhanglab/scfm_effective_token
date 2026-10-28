# step3 — measurement noise (Poisson CV) and within-metacell mixing (LTV)

Two per-gene statistics, computed on the step1 count matrix under a step2
metacell ladder, in ONE pass because both need the same pooling.

`step3a_cv_ltv_vs_k.py`.

## The two statistics

Pool cells into metacells, then per gene and per rung k:

**Poisson CV — how precisely a metacell measures the gene.**  A pooled count
`Y ~ Poisson(Lambda)` has `CV = sqrt(Lambda)/Lambda = 1/sqrt(Lambda)`, so
`CV <= 0.25` is exactly `Lambda >= 16` counts.  Reported as the median over
metacells at that rung.

**LTV — how much of the gene's variation survives inside a metacell.**  With
`lamhat_c = X_c/N_c`, each labelled cell weighted equally (`w_c = 1`), the law of
total variance splits the total into a within- and a between-metacell part:

    within + between == total      (asserted, not assumed)

`total` is partition-independent, so `within/total` is comparable across k and
across methods.  Low within/total means the metacell is homogeneous for that
gene; the two cuts used downstream are `CV <= 0.25` and `within/total <= 0.25`.

Outputs per run (`--outtag`): `cv_ltv_curves_<tag>.csv.gz` (per gene x rung:
`median_CV`, `q25/q75`, `median_Lambda`, `frac_pools_Lambda0`, the LTV as
`within1`, `between1`, `total1`, `lam_bar1`, `share_between1`, and `lam_bar`),
`cv_ltv_diag_<tag>.csv` (per rung: `n_pools`, size min/median/max, coverage,
`ltv_closure_max_rel`, `median_share_between1`), and a JSON.

**Column names.**  The suffix `1` marks the `w_c = 1` split, as in
`cv_ltv_weighting_compare.py`, which also writes the `N_c`-weighted split
unsuffixed.  Only the `w_c = 1` split is computed here, so only suffixed LTV
columns are written; every step4 and step5 script reads `share_between1` and
rejects a curves file without it, so a weighted file cannot be read as an
unweighted one.  On the same inputs the two scripts' `*1` columns agree exactly.

`lam_bar` is the gene's POOLED cohort rate, `sum_c mu_cg / sum_c N_c` over the
labelled cells — the expression level the deciles in step4c–e are cut on, and
identical to `lam_bar` in `cv_ltv_weighting_compare.py`.  `lam_bar1` is the
unweighted mean of the per-cell rates, the centre of the `w_c = 1` LTV; the two
differ by a few percent.

**Cell support.**  The cell support may be a STRICT SUBSET of the
reconstruction's rows: the 28,014 non-UMI cells, and MetaCell-1's outliers, carry
`-1`.  Unlabelled cells are dropped from the pooled terms AND from the totals
(`C` counts labelled cells only), so within + between == total still holds, and
all rungs are asserted to share one support so a single mask is well defined.

**Calibration caveat, left in deliberately.**  `--gene-obs` supplies observed
per-gene sums over the FULL cohort while `gene_rec` is over the masked subset,
so the printed reconstructed/observed ratio is low by the masked fraction
(~0.52%).  That line is reported, never gated, so it is not silently rescaled.

## Runs behind the reported curves — one per (method x embedding)

The curves in use were written by `cv_ltv_weighting_compare.py` (task10), which
computes both weightings in one pass; their `*1` columns are this step's output.

| labels | scGPT | Harmony |
|---|---|---|
| Leiden (rungs 1-1024) | `cv_ltv_weighting_compare.sub` + `_fine.sub` (merged) | `cv_ltv_weighting_compare_h.sub` |
| SEACells+Ward (64-1024) | `cv_ltv_weighting_compare_seacells_sg.sub` | `cv_ltv_weighting_compare_seacells_h.sub` |
| MetaCell-1+Ward, tgstat (64-1024) | `step13_verify_MetaCell1/cv_ltv_mc1tg_sg.sub` | `step13_verify_MetaCell1/cv_ltv_mc1tg_h.sub` |

All of them use the same inputs except `--labels` and `--outtag`; the count matrix
is `main_effect_counts_scgptpca_perCTclip.npy` (scGPT) or
`main_effect_counts_perCTclip.npy` (Harmony) from step1, with
`--genestats allcd8_genestats.csv` and `--gene-obs observed_hvg_genesum.npy`.
