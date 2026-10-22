#!/usr/bin/env python
"""Cut a metacell ladder down to one cell set, keeping it a valid partition.

The Leiden ladders cover all 5,381,464 cells; the SEACells and MetaCell-1
ladders cover only the 5,353,450 UMI cells.  Comparing them means scoring the
Leiden one on the same cells, which is this: set the excluded cells to -1 and
renumber the surviving pools so the ids stay contiguous (a pool that loses
every cell disappears).  Every rung gets the SAME mask, which cv_ltv_vs_k_task10
requires -- it asserts one support across rungs.
"""
import argparse, json, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--labels", required=True)
ap.add_argument("--keep-rowidx", required=True)
ap.add_argument("--keyfmt", default="lab_mean{k}_leiden_s0")
ap.add_argument("--ks", default="1,2,4,8,16,32,64,128,256,512,1024")
ap.add_argument("--out", required=True)
A = ap.parse_args()

z = np.load(A.labels)
keep = None
out, stats = {}, []
for k in (int(x) for x in A.ks.split(",")):
    key = A.keyfmt.format(k=k)
    assert key in z.files, f"{key} not in {A.labels}"
    lab = z[key].astype(np.int32)
    if keep is None:
        keep = np.zeros(lab.size, bool)
        keep[np.load(A.keep_rowidx)] = True
    m = np.where(keep, lab, -1)
    used = np.unique(m[m >= 0])
    remap = np.full(int(lab.max()) + 1, -1, np.int32)
    remap[used] = np.arange(used.size, dtype=np.int32)
    out[key] = np.where(m >= 0, remap[np.where(m >= 0, m, 0)], -1).astype(np.int32)
    stats.append(dict(k=k, pools_before=int(lab.max()) + 1, pools_after=int(used.size)))
    print(f"k={k:5d}: {stats[-1]['pools_before']:>9,} -> {used.size:>9,} pools")
print(f"{int(keep.sum()):,} of {keep.size:,} cells kept")
np.savez_compressed(A.out, **out)
json.dump(dict(source=A.labels, keep_rowidx=A.keep_rowidx,
               cells_kept=int(keep.sum()), cells_total=int(keep.size), rungs=stats),
          open(os.path.splitext(A.out)[0] + ".json", "w"), indent=2)
print("wrote", A.out)
