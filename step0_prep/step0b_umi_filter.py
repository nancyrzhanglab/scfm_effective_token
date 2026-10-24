"""Impact of (1) dropping Smart-seq2 / BD Rhapsody Targeted and (2) dropping
datasets with <1000 cells, on the all-tissue CD8 atlas.

Order matters: the assay filter is applied FIRST, then dataset sizes are
recounted, so a dataset that falls below the threshold only after losing its
Smart-seq2 cells is also dropped.
"""
import pandas as pd, numpy as np, json
D = "results/poisson/task8_scGPTemb2gex"
DROP = ["Smart-seq2", "BD Rhapsody Targeted mRNA"]
MIN = 1000
o = pd.read_csv(f"{D}/all_CD8T/allcd8_obs.csv.gz",
                usecols=["dataset_id", "assay", "cell_type", "tissue_general", "donor_id"],
                low_memory=False)
def desc(t, m):
    s = o[m]
    return dict(step=t, cells=len(s), datasets=s.dataset_id.nunique(),
                donors=s.donor_id.nunique(), assays=s.assay.nunique(),
                cell_types=s.cell_type.nunique(), tissues=s.tissue_general.nunique())
rows = [desc("start", np.ones(len(o), bool))]
m1 = ~o.assay.isin(DROP)
rows.append(desc("after assay filter", m1))
sz = o[m1].dataset_id.value_counts()
keep_ds = set(sz[sz >= MIN].index)
m2 = m1 & o.dataset_id.isin(keep_ds)
rows.append(desc(f"after dataset >= {MIN}", m2))
r = pd.DataFrame(rows)
r["pct_cells_kept"] = r.cells / len(o) * 100
print(r.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
small = sz[sz < MIN]
print(f"\ndatasets dropped for size: {len(small)}  holding {int(small.sum()):,} cells "
      f"({small.sum()/len(o)*100:.2f}%)")
print(f"  their sizes: min={small.min()} median={int(small.median())} max={small.max()}")
print(f"\ncells dropped by assay filter: {int((~m1).sum()):,} "
      f"({(~m1).mean()*100:.2f}%)  -> {o[~m1].assay.value_counts().to_dict()}")
np.save(f"{D}/all_CD8T/allcd8_filtered_rowidx.npy", np.flatnonzero(m2.values))
json.dump(dict(drop_assays=DROP, min_dataset_cells=MIN,
               n_start=len(o), n_kept=int(m2.sum()),
               datasets_start=int(o.dataset_id.nunique()),
               datasets_kept=int(o[m2].dataset_id.nunique())),
          open(f"{D}/all_CD8T/allcd8_filter.json", "w"), indent=1)
print(f"\nsaved row index: {int(m2.sum()):,} cells kept")
