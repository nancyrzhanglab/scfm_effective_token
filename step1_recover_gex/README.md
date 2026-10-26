# step1 — CellANOVA: embedding → cell × gene counts

Recovers a count-scale cell × gene matrix from a cell embedding, by taking
CellANOVA's **main effect** — the part of expression the embedding explains once
the per-batch offsets are averaged out.

**The same four scripts serve both embeddings.** scGPT and Harmony differ only
in `--embedding`, `--mmat` and `--tag`

## The model

    lnr    = (C @ Mmat) * sd_g + mu_g        main effect, lognorm scale
    clip   -> [0, ceil[group, gene]]         cap the recovered gene expression by the observed ceiling
    counts = expm1(lnr) * L_c / target_sum   back to counts
    renorm -> each cell's HVG total matched to its observed HVG total

`C` is the (N × d) embedding, `Mmat` the (d × G) main-effect matrix, `mu_g`/`sd_g`
the per-gene lognorm moments, `L_c` the cell's library size.

## a. Clip groups — `step1a_make_clipgroups.py`

Integer codes of `obs.cell_type`, one per cell. 

This grouping is used in step c where it takes the ceiling per (cell type, gene), 
so a gene can be reconstructed as high as the most extreme cell of its own type but no higher.

## b. fit_M — `step1b_cellanova_fitM.py`

CellANOVA's `fit_M` regresses real expression on the latent embedding per `dataset_id`
to obtain the main effect of gene expression

    B_x  = lstsq(C_x, S_x) = pinv(C_x^T C_x) @ (C_x^T S_x)
    Mmat = mean over good levels of B_x

and both `C_x^T C_x` (d × d) and `C_x^T S_x` (d × G) are sums over cells.

Outputs: `<prefix>_<tag>_Mmat_dataset_id.npy`, `<prefix>_ceiling_table.npy`,
`observed_hvg_libsize.npy`.


## c. Counts — `step1c_cellanova_counts.py`

Convert recovered gene expression rates into count scale reverting the normalization step 
(i.e. mulitplying by the standard deviation and adding the mean of gene expression). 


## d. Verify — `step1d_verify_counts.py`

Check the output of 1c is the same shape as the oberved count, and that the per cell sum 
matches the observed as well.

## e. Observed per-gene totals — `step1e_observed_gene_sums.py`

This rebuilds:

    out[g] = sum over all cells of X[c, g]       X = the h5ad's raw counts
