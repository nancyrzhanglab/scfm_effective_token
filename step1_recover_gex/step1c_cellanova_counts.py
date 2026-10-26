#!/usr/bin/env python
"""Convert the CellANOVA main effect to count scale, one row shard per task.

For each dataset, compute the recovered gene expression rate given Mmat, 
and can be parallelised

    lnr    = (C @ Mmat) * sd_g + mu_g
    clip   -> [0, ceil[group, gene]]
    counts = expm1(lnr) * L_c / target_sum
    renorm -> each cell's HVG total matched to its observed HVG total

Only the renormalised matrix is written.
--init preallocates the output once before the array runs.

"""
import argparse, json, os, time
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--in-dir", required=True)
ap.add_argument("--out", required=True, help="path of the (N,G) output .npy")
ap.add_argument("--prefix", default="allcd8")
ap.add_argument("--embedding", required=True)
ap.add_argument("--mmat", required=True)
ap.add_argument("--ceil", required=True)
ap.add_argument("--groups", required=True)
ap.add_argument("--obs-tot", required=True)
ap.add_argument("--target-sum", type=float, default=1e4)
ap.add_argument("--chunk", type=int, default=25_000)
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--n-shards", type=int, default=1)
ap.add_argument("--check-rows", type=int, default=5_000, help="rows sampled for the readback check")
ap.add_argument("--init", action="store_true", help="preallocate the output and exit")
a = ap.parse_args()
t0 = time.time(); log = lambda m: print(f"[{time.time()-t0:8.1f}s] s{a.shard}: {m}", flush=True)

# read data
gs = pd.read_csv(os.path.join(a.in_dir, f"{a.prefix}_genestats.csv"))
mu_g = gs.ln_mean.to_numpy(np.float64); sd_g = gs.ln_std.to_numpy(np.float64)
L_c = np.load(os.path.join(a.in_dir, f"{a.prefix}_libsize.npy")).astype(np.float64)
N, G = len(L_c), len(mu_g)

if a.init:
    m = np.lib.format.open_memmap(a.out, mode="w+", dtype=np.float32, shape=(N, G))
    del m
    log(f"preallocated {a.out}  ({N:,} x {G})  {N*G*4/1e9:.1f} GB")
    raise SystemExit(0)


def npy_data_offset(path, want_shape):
    """Byte offset of the first data element for header, verified against the header."""
    with open(path, "rb") as f:
        ver = np.lib.format.read_magic(f)
        shape, fortran, dtype = np.lib.format._read_array_header(f, ver)
        off = f.tell()
    assert shape == want_shape, f"{path} header shape {shape} != {want_shape}"
    assert not fortran, f"{path} is Fortran-ordered; positional row writes assume C order"
    assert dtype == np.dtype(np.float32), f"{path} dtype {dtype} != float32"
    return off


C = np.load(a.embedding, mmap_mode="r")
Mmat = np.load(a.mmat).astype(np.float64)
ceil_tab = np.load(a.ceil).astype(np.float64)
gcode = np.load(a.groups).astype(np.int64)
obs_tot = np.load(a.obs_tot).astype(np.float64)
assert Mmat.shape == (C.shape[1], G) and len(gcode) == N == len(obs_tot)
sf = L_c / a.target_sum

edges = np.linspace(0, N, a.n_shards + 1).astype(np.int64)
LO, HI = int(edges[a.shard]), int(edges[a.shard + 1])
log(f"rows [{LO:,},{HI:,}) of {N:,}")

DATA_OFF = npy_data_offset(a.out, (N, G))
ROW = G * 4  # float32
fh = open(a.out, "r+b", buffering=0)

clipped = nonzero = 0
for lo in range(LO, HI, a.chunk):
    hi = min(lo + a.chunk, HI)
    lnr = (np.asarray(C[lo:hi], np.float64) @ Mmat) * sd_g[None, :] + mu_g[None, :] # revert the normalization step
    ceil = ceil_tab[gcode[lo:hi]]
    clipped += int((lnr > ceil).sum()); nonzero += int((lnr > 0).sum())
    np.clip(lnr, 0.0, ceil, out=lnr); del ceil
    c = np.expm1(lnr) * sf[lo:hi, None]; del lnr # revert to count scale and match to HVG library size
    tot = c.sum(1)
    c *= np.where(tot > 0, obs_tot[lo:hi] / np.maximum(tot, 1e-300), 0.0)[:, None]
    blk = np.ascontiguousarray(c, dtype=np.float32); del c
    fh.seek(DATA_OFF + lo * ROW)
    n = fh.write(memoryview(blk).cast("B"))
    assert n == blk.nbytes, f"short write at row {lo}: {n} of {blk.nbytes} bytes"
    del blk
    if ((lo - LO) // a.chunk) % 4 == 0: log(f"  {hi:,}/{HI:,}")
fh.flush(); os.fsync(fh.fileno())


def read_rows(idx):
    """Read individual rows with ordinary read() syscalls."""
    out = np.empty((len(idx), G), np.float32)
    for k, r in enumerate(idx):
        fh.seek(DATA_OFF + int(r) * ROW)
        b = fh.read(ROW)
        assert len(b) == ROW, f"short read at row {r}"
        out[k] = np.frombuffer(b, np.float32)
    return out


# Sample uniformly across the whole shard to verify reconstruction sums up to the rigth total.
ns = min(a.check_rows, HI - LO)
ridx = np.linspace(LO, HI - 1, ns).astype(np.int64)
chk = read_rows(ridx).astype(np.float64)
ot = obs_tot[ridx]
rel = np.abs(chk.sum(1) - ot) / np.maximum(ot, 1e-12) # check total count matches with observed total

expect_zero = ot <= 0
got_zero = chk.sum(1) <= 0
nzero = int((got_zero & ~expect_zero).sum())
log(f"readback check: max rel err {np.nanmax(rel[~expect_zero]) if (~expect_zero).any() else 0:.3g} "
    f"over {ns:,} rows spanning [{LO:,},{HI:,}), {nzero} unexpectedly all-zero, "
    f"{int(expect_zero.sum())} zero by construction")
assert nzero == 0, f"{nzero} of {ns} sampled rows read back all-zero -- writes did not persist"
assert int((~got_zero & expect_zero).sum()) == 0, \
    "a row with zero target total came back non-empty"
assert np.nanmax(rel[~expect_zero], initial=0.0) < 1e-4, \
    "row totals do not match the target HVG totals"
fh.close()

json.dump(dict(shard=a.shard, lo=LO, hi=HI, clipped=clipped, nonzero=nonzero,
               frac_clipped=clipped / max(nonzero, 1),
               check_rows=ns, zero_by_construction=int(expect_zero.sum()),
               max_rel_err=float(np.nanmax(rel[~expect_zero], initial=0.0))),
          open(f"{a.out}.shard{a.shard:03d}.json", "w"), indent=1)
log(f"SHARD OK  clipped {clipped:,} of {nonzero:,} positive entries "
    f"({clipped/max(nonzero,1)*100:.2f}%)")
