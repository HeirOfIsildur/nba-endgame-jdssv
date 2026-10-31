#!/bin/bash
# one Slurm array task: 20 bootstrap replicates, starting at 20 * task id. Submit 50 tasks (B = 1,000):
#   sbatch -A <account> -p ffa --array=0-49 -c 8 --mem=16G -t 1:00:00 --export=ALL,PY=<venv>/bin/python \
#          --chdir=<repo> scripts/chimera_boot_task.sh
# Before syncing the code to the cluster (it has no .git), record the commit: git rev-parse HEAD > COMMIT.
# Then copy results/boot/ and results/platform/ back, copy replicates 0, 517, 999 to results/platform/reference/
# (deposited, so that scripts/compare_platform.py can be rerun from the repository) and run
# `python -m nbaeg.bootstrap summary 1000`.
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMBA_CACHE_DIR=${TMPDIR:-/tmp}/numba-$SLURM_JOB_ID-$SLURM_ARRAY_TASK_ID
export PYTHONPATH=$PWD/src
PY=${PY:-python3}        # interpreter with requirements-lock.txt installed (venv on /home on Chimera)
first=$((SLURM_ARRAY_TASK_ID * 20))
$PY -m nbaeg.bootstrap run 20 ${SLURM_CPUS_PER_TASK} $first
mkdir -p results/platform
{ echo "host=$(hostname) job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID cores=$SLURM_CPUS_PER_TASK"
  lscpu | grep 'Model name'; cat /etc/redhat-release; echo "commit=$(cat COMMIT)"
  $PY -c 'import numpy,pandas,statsmodels,numba,sys;print(sys.version.split()[0],numpy.__version__,pandas.__version__,statsmodels.__version__,numba.__version__)'
} > results/platform/task_$SLURM_ARRAY_TASK_ID.txt
[ "$SLURM_ARRAY_TASK_ID" = 0 ] && $PY -c 'import importlib.metadata as m, sys; print("python", sys.version.split()[0]); [print(d.name + "==" + d.version) for d in sorted(m.distributions(), key=lambda d: d.name.lower())]' > results/platform/worker_environment.txt || true
