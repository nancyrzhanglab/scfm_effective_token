# step4 — how the two criteria behave against metacell size

Three views of the same step3 curves, before step5 turns them into token counts:
the threshold heatmaps (a), one gene (b), and a whole expression decile (c).

| | file | original |
|---|---|---|
| a | `step4a_threshold_heatmap.py` | `t10/plot_precision_heatmap.py` |
| b | `step4b_cv_ltv_vs_k_gene.py` | `t10/plot_cv_vs_k_gene.py` |
| c | `step4c_cv_ltv_vs_k_decile.py` | `t10/plot_decile_curves.py` |

## a. Threshold heatmaps — `step4a_threshold_heatmap.py`

Cell (t, k) = the number of genes meeting threshold t at metacell size k, for
ONE criterion.  Both axes are ordinal and the value is a magnitude, so it is a
one-hue sequential heatmap.  This is the step before step5, which asks the same
question with both criteria at once and converts the answer into token counts.

    --metric median_CV      precision    -> <prefix>_precision_counts.csv
    --metric share_within   homogeneity  -> <prefix>_homogeneity_counts.csv
    --pair                  both metrics side by side, one ramp each

Thresholds come from `--thresholds`; `--min-lambda` sets the pooled-count floor.
Figure controls: `--square`, `--cell-size`, `--annot`, `--cell-format`,
`--cell-fontsize`, `--axis-fontsize`, `--pdf`, `--pdf-fontsize`.

**Genes whose CV is undefined at a given k** (no metacell with a non-zero rate)
simply fail the threshold.  That is the intended behaviour — they are not
precise — rather than being dropped from the denominator, which would quietly
inflate the pass rate at coarse rungs.

**Reading the two metrics together.** Precision improves monotonically with k
(pooling more cells raises Lambda), while homogeneity degrades with k (a bigger
metacell mixes more states).  The counts therefore move in opposite directions
down the two heatmaps, and the k that is best for a gene is wherever the two
cross — which is what step5 turns into a number.

The same file also lives in `step5_cal_eff_tokens/` under its original name,
because `plot_three_panels.py` there imports its ramps and ink colours.  The two
copies are identical; edit both or neither.

## b. One gene — `step4b_cv_ltv_vs_k_gene.py`

CV and LTV against k for a single gene, across construction methods.  The line
is the median over metacells at each k; `--band` adds the inter-quartile range,
so the spread shown is ACROSS METACELLS.

`--metric` takes one name or several comma-separated for side-by-side panels:
`median_CV`, `share_within`, `share_between`, the raw `within`/`between`/`total`,
or `components` (within and between overlaid with flat total rules; filled =
within, hollow = between).  `--stack` puts the panels one above the other, with
the x label only on the bottom one.

Pure Poisson sampling falls as `k^-1/2`; the guide line draws that slope so
departures are readable.  `--floor LABEL::CSV` overlays a simulated floor,
divided by the FIRST curve's total so the ratio is on one denominator.
`--yscale log` masks non-positive values and pins the limits from the data,
because `within = 0` at k=1 would otherwise draw a spike to the axis floor.

**Not every curve comes from the same reconstruction.**  The source file for each
is printed and carried into the legend, because a method scored on a different
main-effect matrix is not directly comparable to the others.

## c. One expression decile — `step4c_cv_ltv_vs_k_decile.py`

The same two metrics aggregated over the genes of one expression decile
(`--decile`, `--n-deciles`).  A single gene's curve is noisy and, as GZMK vs
MKI67 showed, the method ranking is gene-specific, so this is the version to
quote.

**The band means something different here.**  The line is the MEDIAN across
genes and the band is their IQR, so the spread is BETWEEN GENES — not across
metacells as in b.

Deciles are cut on `lam_bar`, the cohort mean rate, taken from ONE reference
curve set (`--reference`) and that same gene list is then applied to every
method, so all curves describe the same genes.
