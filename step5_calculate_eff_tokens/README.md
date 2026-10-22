# step5 — effective (usable) tokens

Turns the step3 per-gene curves into a count of usable measurements.

## The banking rule — the one definition everything here shares

A gene is counted **once**, at the **finest rung where it first passes both
cuts**, and valued at that rung's metacell count:

    usable(K) = sum over k <= K of  n_pools(k) * |{genes first passing at k}|

`new = passing(k) - banked` is a SET difference, never a difference of counts: a
gene can drop out of the passing set at a coarser rung and a count-difference
would then subtract it twice.  The two cuts are `median_CV <= cv_cut` and
`within/total <= w_cut`, both read off the step3 curves.

Finest-rung banking is the conservative choice — a gene banked at k=64 is worth
far more metacells than the same gene banked at k=1024 — and it is what makes
the total monotone in K.

| | file | original |
|---|---|---|
| a | `step5a_usable_measurements.py` | `t10/plot_usable_measurements.py` |
| b | `step5b_usable_grid.py` | `t10/plot_usable_grid.py` |

Four support scripts keep their original names because the first two are read by
the above (for the ramps and the ink colours) and the others are figures
assembled from their CSVs: `plot_pass_by_decile.py`, `plot_precision_heatmap.py`,
`plot_passing_genes_panel.py`, `plot_three_panels.py`.

## a. The total — `step5a_usable_measurements.py`

Usable measurements against metacell size, at ONE pair of cuts (default
0.25 / 0.25).  `--cumulative` applies the banking rule above; without it each
rung is scored independently.  `--no-genes-panel` drops the gene-count line.

## b. The grid — `step5b_usable_grid.py`

The same total over a 5 x 5 grid of (CV cut, within cut), each in
{0.05, 0.15, 0.25, 0.35, 0.45}.  `--highlight CV,W` rings one cell.

**Read this one carefully.**  The quantity is monotone in BOTH cuts by
construction, so the maximum is always the loosest corner and the grid on its
own argues for thresholds nobody would defend.  The options exist to counter
that: `--relative` (ratio to a reference cell), `--marginal` (fold gain per CV
step), `--annotate-cost` + `--floor` (pooled counts `1/CV^2` under the columns,
and whether each row sits above or below the Poisson floor), `--bars` (the 5 x 5
accumulation panel).

On the marginal view: the fold gain falls across the grid (9.6x -> 1.33x) only
because the base grows.  In ABSOLUTE terms the return is roughly constant at
~60 M tokens per +0.1 of CV, so "diminishing returns" is the wrong reading.
