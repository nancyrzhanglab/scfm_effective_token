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

## SEACells — step2c + d + g (flat base, then nested by Ward)

SEACells is kernel archetypal analysis: it holds an (n_cells × n_archetypes)
weight matrix and iterates Frank-Wolfe over it.  At 5.35 M cells that matrix is
not representable, so:

`step2c` cuts the cohort into locality-based piles (`--target 15000` cells each) —
  the Metacell-2 "divide and conquer" device, used purely for tractability.

`step2d` runs SEACells on one pile per array task, producing the BASE rung and
  the size-weighted metacell centroid in embedding space.

`step2g` merges that flat base into a nested ladder.

## MetaCell-1 — step2c + e + f + g

`step2e` runs Baran et al. 2019 (Genome Biology 20:206) through the authors'
own engine: `step2e_metacell1_pile.py` builds the exact Euclidean k-NN in the
embedding and hands it to `step2e_metacell1_pile.R`, which is the MetaCell R
package's pipeline calling tgstat (balanced k-NN graph, 500 resampled graph
covers, balanced co-cluster cover).  Cells MetaCell leaves uncovered are kept
out as outliers (label -1 downstream).  Needs `module load R/4.5.2` with tgstat.
One pile per array task, same piles as SEACells.

`step2f` concatenates the per-pile partitions — concatenation only, no merging
across piles.  MetaCell is FLAT: there is no tree to cut, so pile boundaries are
permanent, which is the price of using piles for tractability.  

`step2g` then supplies the ladder, same as for SEACells.

## g. The Ward step — what "nested" costs

A flat method run at k=16 and again at k=64 gives two partitions with no
containment relation.  `step2g` fixes that structurally: one dendrogram is built
over the base metacells and cut at each rung, so every coarse metacell is
exactly a union of finer ones.

`--linkage ward_weighted`: the merge criterion weights each base metacell by its
cell count, so a metacell of 400 cells counts 400× one of a single cell.  Plain
Ward on centroids would treat them as equal points and let singletons distort
the tree.

## h. Index cells in the 5.38M h5ad file

* **`step2h_expand_to_full_cohort.py`** lifts task10's labels from the filtered
  cohort to the full one, via `allcd8_filtered_rowidx.npy`, because the
  reconstruction and the cv/ltv machinery are indexed by the full 5,381,464-row
  h5ad.
