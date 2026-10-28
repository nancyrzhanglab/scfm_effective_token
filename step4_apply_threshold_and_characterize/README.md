# step4 — how the two criteria behave against metacell size

Views of the same step3 curves, before step5 turns them into token counts: the
threshold heatmaps (a), one gene (b), a whole expression decile (c), and the
genes passing BOTH criteria by decile and k (d, e).

Every script here reads within/total from the UNWEIGHTED split
(`share_between1` with `within1` / `between1` / `total1`, w_c = 1), as step5
does, and stops on a curves file without it.  Expression deciles are cut on
`lam_bar`, the pooled cohort rate.

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

**Reading the two metrics together.** Precision improves monotonically with k, 
while homogeneity degrades with k (a bigger metacell mixes more states).  
The counts therefore move in opposite directions down the two heatmaps, 
and the k that is best for a gene is wherever the two overlap.

## b. One gene — `step4b_cv_ltv_vs_k_gene.py`

CV and LTV against k for a single gene, across construction methods.  The line
is the median over metacells at each k; `--band` adds the inter-quartile range,
so the spread shown is ACROSS METACELLS.

`--metric` takes one name or several comma-separated for side-by-side panels:
`median_CV`, `share_within`, `share_between`, the raw `within`/`between`/`total`,
or `components` (within and between overlaid with flat total rules; filled =
within, hollow = between).  These are plot names; the values are the w_c = 1
columns (`share_between1`, `within1`, ...).  `--stack` puts the panels one above the other, with
the x label only on the bottom one.

## c. One expression decile — `step4c_cv_ltv_vs_k_decile.py`

The same two metrics aggregated over the genes of one expression decile
(`--decile`, `--n-deciles`).  A single gene's curve is noisy, so we examine 
a group of genes together.

**The band means something different here.**  The line is the MEDIAN across
genes and the band is their IQR, so the spread is between genes — not across
metacells as in b.

Deciles are cut on `lam_bar`, the cohort mean rate, taken from one reference
curve set (`--reference`) and that same gene list is then applied to every
method, so all curves describe the same genes.

## d. Genes passing both criteria, by decile — `step4d_pass_by_decile.py`

Cell (decile, k) = the number of genes with `median_CV <= --cv-cut` AND
`within/total <= --within-cut` at metacell size k: the set step5 banks.  Genes
are binned by `lam_bar`, with deciles below `--lump-below` shown as one row
because almost nothing there passes.  `--fraction` shows the share of each
decile instead of the count.  Setting one cut out of reach gives the
single-criterion view (`_cv_only` / `_within_only` outputs).

## e. The same table with its total — `step4e_passing_genes_panel.py`

d's table drawn as two stacked panels on one k axis: the total number of passing
genes (top) and their split by decile (bottom); the column sums of the grid are
asserted to reproduce the line.  `--metacells --diag` adds the metacell count per
rung on a right-hand log axis.  Exact page size via `--size` (`--solve-height`
for square cells).
