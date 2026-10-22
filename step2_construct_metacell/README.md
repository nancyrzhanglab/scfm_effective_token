# step2 — metacell construction

Three methods, each built on BOTH embeddings (scGPT and Harmony), each producing
a **nested** ladder of partitions indexed by target mean metacell size k.

Nesting is the whole requirement: a metacell at k=256 must be a union of
metacells at k=128, or a quantity cannot be tracked across k and the ladder is
eleven unrelated clusterings rather than one object at eleven resolutions.
The three methods buy it in two different ways.

| | file here | original (task) |
|---|---|---|
| a | `step2a_build_knn.py` | `t8/build_knn_task8.py` |
| b | `step2b_leiden_ladder.py` | `t8/build_metacells_task8.py` |
| c | `step2c_make_piles.py` | `t10/make_piles.py` |
| d | `step2d_seacells_pile.py` | `t10/seacells_pile.py` |
| e | `step2e_metacell1_pile.py` | `t10/metacell1_pile.py` |
| f | `step2f_metacell1_gather.py` | `t10/metacell1_gather.py` |
| g | `step2g_merge_ward.py` | `t10/merge_ward.py` |
| h | `step2h_mask_labels.py` | `t10/mask_labels.py` |
| i | `step2i_expand_to_full_cohort.py` | `t10/expand_labels_to_full_cohort.py` |

## Leiden — a + b (nested by construction)

`step2a` builds a k=30 kNN graph on the embedding; `step2b` runs a max-heap of
pools: pop the largest, Leiden-split it, push the children, stop when the pool
count reaches N/mean.  Because the recursion only ever SPLITS, the ladder is
strictly nested by construction and coverage is 1.0 at every rung — and the
nesting is ASSERTED between consecutive rungs, not assumed.

Rungs `1024,512,256,128,64,32,16,8,4,2,1`, resolution 1.0, seed 0, 4 retries.
Every rung is written as soon as it is reached, so a run too slow to reach
mean=2 still leaves the coarse rungs usable.

Two changes from the 255k-cell ancestor (`task5/build_metacells_leiden.py`):
the igraph is built from the scipy sparse adjacency directly (the old
`zip(...)` form would materialise ~59 M Python tuples), and rungs are flushed
as they complete.

Drivers: `step2_leiden_scgpt.sub`, `step2_leiden_harmony.sub` (each runs a then b).

**The ladder is entry-rung independent**: `--target-means` ladders starting at
different top rungs reproduce bit-for-bit at shared rungs, so cross-run
validation is exact.

## SEACells — c + d + g (flat base, then nested by Ward)

SEACells is kernel archetypal analysis: it holds an (n_cells × n_archetypes)
weight matrix and iterates Frank-Wolfe over it.  At 5.35 M cells that matrix is
not representable, so:

* **c** cuts the cohort into locality-based piles (`--target 15000` cells each) —
  the Metacell-2 "divide and conquer" device, used purely for tractability.
* **d** runs SEACells on one pile per array task, producing the BASE rung and
  the size-weighted metacell centroid in embedding space.
* **g** merges that flat base into a nested ladder.

Base k=16 for Harmony (`merge.sub`), k=64 for scGPT (`merge_scgpt.sub`,
`merge_scgpt_1024.sub`), rungs 64/128/256/512 (+1024).

## MetaCell-1 — c + e + f + g

`step2e` implements Baran et al. 2019 (Genome Biology 20:206) **directly**: the
authors' R package is archived, and the maintained `metacells` PyPI package is
MetaCell-2 (Ben-Kiki 2022), a different algorithm.  One pile per array task,
same piles as SEACells.

`step2f` concatenates the per-pile partitions — concatenation only, no merging
across piles.  MetaCell is FLAT: there is no tree to cut, so pile boundaries are
permanent, which is the price of using piles for tractability.  `step2g` then
supplies the ladder, same as for SEACells.

## g. The Ward step — what "nested" costs

A flat method run at k=16 and again at k=64 gives two partitions with no
containment relation.  `step2g` fixes that structurally: ONE dendrogram is built
over the base metacells and cut at each rung, so every coarse metacell is
exactly a union of finer ones.

`--linkage ward_weighted`: the merge criterion weights each base metacell by its
CELL COUNT, so a metacell of 400 cells counts 400× one of a single cell.  Plain
Ward on centroids would treat them as equal points and let singletons distort
the tree.

## h, i — making the ladders comparable

The two families are built on different cell sets, and the adapters below
reconcile them.  Both must be applied or the methods are scored on different
cells:

* Leiden (a+b) is BUILT on all 5,381,464 cells -- its drivers point at
  `allcd8_harmony_dataset_id.npy` / `allcd8_scgpt_pca50.npy`, never the
  `allcd8f` ones -- and is then MASKED down to the UMI cells for scoring.
* SEACells and MetaCell-1 are built on the UMI-filtered 5,353,450 from the
  start (`allcd8f_harmony_dataset_id.npy`).

* **`step2h_mask_labels.py`** cuts a ladder down to one cell set while keeping it
  a valid partition: excluded cells go to -1 and the surviving pools are
  renumbered so ids stay contiguous (a pool that loses all its cells disappears
  rather than leaving a hole).  Outputs land in
  `results/.../Leiden_masked/metacell_labels_<prefix>_leiden_masked[_all].npz`
  with a sidecar JSON recording source, keep_rowidx and the two cell counts.
  `_masked` carries rungs 64-512; `_masked_all` carries all eleven (1-1024).
* **`step2i_expand_to_full_cohort.py`** lifts task10's labels from the filtered
  cohort to the full one, via `allcd8_filtered_rowidx.npy`, because the
  reconstruction and the cv/ltv machinery are indexed by the full 5,381,464-row
  h5ad.

## Drivers

Harmony: `piles.sub`, `seacells_array.sub`, `merge.sub`, `metacell1_array.sub`,
`metacell1_K64_array.sub`, `metacell1_K64_merge.sub`.
scGPT: `piles_scgpt.sub`, `seacells_scgpt_array.sub`, `merge_scgpt.sub`,
`merge_scgpt_1024.sub`, `metacell1_scgpt_array.sub`, `metacell1_scgpt_merge.sub`.

These keep their original names: they are per-embedding variants of the same
four scripts, and renaming them to a letter sequence would imply an order they
do not have.  All carry absolute paths into
`results/poisson/task10_metacell_construction/`.
