#!/bin/bash
# Build the `scgpt` conda env.  Staged so a late failure does not discard
# earlier work: each stage writes a stamp file and is skipped if the stamp exists.
set -o pipefail

ENV_NAME=scgpt
CONDA_ROOT=/home/stat/nzh/team/emiliac/miniconda3
STAMP_DIR=/home/mnt/weka/nzh/nzhanglab/project/emiliac/FM_effective/scripts/scgpt/.stamps
mkdir -p "$STAMP_DIR"

source "$CONDA_ROOT/etc/profile.d/conda.sh"

# Serial builds only -- parallel compiles get OOM-killed under the cgroup cap.
export MAX_JOBS=1
export MAKEFLAGS="-j1"
export PIP_NO_INPUT=1

stage () {                       # stage <name> <command...>
  local name="$1"; shift
  if [ -f "$STAMP_DIR/$name" ]; then
    echo "### SKIP  $name (already done)"; return 0
  fi
  echo "### START $name  $(date)"
  "$@" || { echo "### FAIL  $name (exit $?)"; exit 1; }
  touch "$STAMP_DIR/$name"
  echo "### OK    $name  $(date)"
}

# ---------------------------------------------------------------- stage 1
mk_env () {
  conda create -y -n "$ENV_NAME" python=3.10 pip
}
stage 01_create_env mk_env

conda activate "$ENV_NAME"
PY="$CONDA_ROOT/envs/$ENV_NAME/bin/python"
echo "python = $($PY -V)  at  $PY"

# ---------------------------------------------------------------- stage 2
# torch 2.3.0 is the newest release torchtext still builds against
# (torchtext 0.18.0 hard-pins torch==2.3.0), and scgpt needs
# torchtext's C++ Vocab.  cu121 wheels run fine on CPU-only nodes.
mk_torch () {
  $PY -m pip install --no-cache-dir \
      torch==2.3.0 torchvision==0.18.0 \
      --index-url https://download.pytorch.org/whl/cu121 &&
  $PY -m pip install --no-cache-dir torchtext==0.18.0
}
stage 02_torch mk_torch

# ---------------------------------------------------------------- stage 3
# Pin torch/torchtext so scgpt's resolver cannot swap them out from under us.
mk_scgpt () {
  cat > /tmp/scgpt_constraints.txt <<'CONS'
torch==2.3.0
torchvision==0.18.0
torchtext==0.18.0
CONS
  $PY -m pip install --no-cache-dir -c /tmp/scgpt_constraints.txt "scgpt==0.2.4"
}
stage 03_scgpt mk_scgpt

# ---------------------------------------------------------------- stage 4
# Extras the tutorials use that are not in scgpt's dependency list.
mk_extras () {
  $PY -m pip install --no-cache-dir -c /tmp/scgpt_constraints.txt \
      ipykernel anndata h5py
}
stage 04_extras mk_extras

echo
echo "==================== pip check ===================="
$PY -m pip check
echo "(a non-zero exit above is reported, not fatal -- see verify step)"
echo
echo "==================== versions ===================="
$PY - <<'PY'
import importlib
for m in ["torch","torchtext","scgpt","scanpy","anndata","scvi","numpy","pandas","numba","datasets","scib"]:
    try:
        mod = importlib.import_module(m)
        print(f"  {m:12s} {getattr(mod,'__version__','?')}")
    except Exception as e:
        print(f"  {m:12s} IMPORT FAILED: {type(e).__name__}: {e}")
PY
echo "### ALL STAGES DONE $(date)"
