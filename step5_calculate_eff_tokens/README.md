# step5 — effective (usable) tokens

Turns the step3 per-gene curves into a count of usable measurements.

## The banking rule — the one definition everything here shares

A gene is counted **once**, at the **finest rung where it first passes both
cuts**, and valued at that rung's metacell count:

    k_min(g)  = the smallest k at which gene g passes BOTH cuts
    usable    = sum over genes of  n_pools(k_min(g))
    usable(K) = the same sum over the genes with k_min(g) <= K

A gene that never passes adds 0; a gene that passes at one rung and fails at a
coarser one (within/total grows with k) is still counted once, at k_min.  The
two cuts are `median_CV <= cv_cut` and `within/total <= w_cut`, both read off
the step3 curves.  within/total is the UNWEIGHTED split (`1 - share_between1`,
w_c = 1), so the curves must come from `cv_ltv_weighting_compare.py` (`*_wcmp*`);
Leiden on scGPT at 0.25 / 0.25 gives 113,840,474.

Finest-rung banking is the conservative choice — a gene banked at k=64 is worth
far more metacells than the same gene banked at k=1024 — and it is what makes
the total monotone in K.

## a. The total — `step5a_usable_measurements.py`

Usable measurements against metacell size, at one pair of cuts (default
0.25 / 0.25).  `--cumulative` applies the banking rule above; without it each
rung is scored independently.  `--no-genes-panel` drops the gene-count line.

## b. The grid — `step5b_usable_grid.py`

The same total over a 5 x 5 grid of (CV cut, within cut), each in
{0.05, 0.15, 0.25, 0.35, 0.45}.  `--highlight CV,W` rings one cell.

The quantity is monotone in both cuts by construction, so the maximum is always 
the loosest corner and the grid on its own argues for thresholds nobody would defend.  
