# step2 — metacell construction

Three methods, each built on BOTH embeddings (scGPT and Harmony), each producing
a **nested** ladder of partitions indexed by target mean metacell size k.

Nesting is the whole requirement: a metacell at k=256 must be a union of
metacells at k=128, or a quantity cannot be tracked across k and the ladder is
eleven unrelated clusterings rather than one object at eleven resolutions.
The three methods buy it in two different ways.

## Leiden — step2a + b (nested by construction)

`step2a` builds a k=30 kNN graph on the embedding; `step2b` runs a max-heap of
pools: pop the largest, Leiden-split it, push the children, stop when the pool
count reaches N/mean.  Because the recursion only ever splits, the ladder is
strictly nested by construction and coverage is 1.0 at every rung — and the
nesting is ASSERTED between consecutive rungs.

Rungs `1024,512,256,128,64,32,16,8,4,2,1`, resolution 1.0, seed 0, 4 retries.
Every rung is written as soon as it is reached, so a run too slow to reach
mean=2 still leaves the coarse rungs usable.

## SEACells — step2c + d + f + g (flat base, then nested by Ward)

SEACells is kernel archetypal analysis: it holds an (n_cells × n_archetypes)
weight matrix and iterates Frank-Wolfe over it.  At 5.35 M cells that matrix is
not representable, so:

`step2c` cuts the cohort into locality-based piles (`--target 15000` cells each) —
  the Metacell-2 "divide and conquer" device, used purely for tractability.

`step2d` runs SEACells on one pile per array task (`--target-size 64`), producing
  the BASE rung and the size-weighted metacell centroid in embedding space.

`step2f` merges that flat base into a nested ladder (`--base-k 64`, rungs
  64–1024), and `step2g` lifts it to the full cohort.

## MetaCell-1 — step2c + e + f + g

`step2e` runs Baran et al. 2019 (Genome Biology 20:206) through the authors'
own engine: `step2e_metacell1_driver.py` builds the exact Euclidean k-NN in the
embedding and hands it to `step2e_metacell1_tgstat.R`, which is the MetaCell R
package's pipeline calling tgstat (balanced k-NN graph, 500 resampled graph
covers, balanced co-cluster cover).  Cells MetaCell leaves uncovered are kept
out as outliers (label -1 downstream).  Needs `module load R/4.5.2` with tgstat.
One pile per array task, same piles as SEACells.

Parameters: `--K 64 --min-mc-size 25 --balanced-K 20`, everything else at the
package defaults (k_expand 10, k_beta 3, 500 resamples at p = 0.75, resampling
min_mc_size 20, alpha 2, cooling 1.05, burn-in 10, seed 42).  The vignette's
final cover (30 / 30) gave a base mean of ~78 cells; 25 / 20 gives ~59–64.
Outliers: 1.95% of cells on Harmony, 0.02% on scGPT.

MetaCell is FLAT, so `step2f` supplies the ladder as for SEACells, with
`--base-k auto` (rungs sized from the realised base mean), then `step2g`.

## f. The Ward step — what "nested" costs

A flat method run at k=64 and again at k=128 gives two partitions with no
containment relation.  `step2f` fixes that structurally: one dendrogram is built
over the base metacells and cut at each rung, so every coarse metacell is
exactly a union of finer ones.

Rung k gets round(NB · base_k / k) metacells (NB = base metacells).  With
`--base-k 64` the k=64 rung IS the base, unmerged, at its realised mean (the
SEACells ladders, 64.0).  With `--base-k auto`, base_k = N/NB and every rung's
mean is exactly k, so a base below 64 is merged slightly at k=64 (the MetaCell-1
ladders: Harmony base 64.3, so k=64 is the base; scGPT base 58.8, merged to 64).

`--linkage ward_weighted`: the merge criterion weights each base metacell by its
cell count, so a metacell of 400 cells counts 400× one of a single cell.  Plain
Ward on centroids would treat them as equal points and let singletons distort
the tree.

## g. Index cells in the 5.38M h5ad file

* **`step2g_expand_to_full_cohort.py`** lifts a step2f ladder from the filtered
  cohort to the full one, because the reconstruction and the cv/ltv machinery
  are indexed by the full 5,381,464-row h5ad.  It composes two maps: the
  ladder's own rows (`<prefix>_rowidx_filtered.npy` from step2f) into the
  filtered cohort, then `allcd8_filtered_rowidx.npy` into the full one.  Cells
  without a label get -1 — the 28,014 non-UMI cells, plus MetaCell-1's
  outliers — and cv/ltv drops them from every term.
