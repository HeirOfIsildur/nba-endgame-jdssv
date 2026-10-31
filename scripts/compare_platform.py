"""Recompute bootstrap replicates locally and compare them with the deposited (cluster) ones.

usage: python scripts/compare_platform.py [r ...]      (default replicates 0 517 999)
writes results/platform/local_vs_cluster.json: per replicate the fingerprints and the largest absolute
difference over all stored arrays. The cluster's replicates 0, 517 and 999 are deposited in
results/platform/reference/ (results/boot/ is not in the repository); other replicates are read from results/boot/.
Agreement is numerical (largest absolute difference), not byte equality of the files.
"""
import json
import platform
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

from nbaeg import bootstrap as b

reps = [int(x) for x in sys.argv[1:]] or [0, 517, 999]
deposited = b.BOOT
tmp = Path(tempfile.mkdtemp())
b.BOOT = tmp
b._init()
rows = []
for r in reps:
    b.one(r)
    ref = deposited.parent / 'platform' / 'reference' / f'rep_{r:03d}.npz'
    A, C = np.load(tmp / f'rep_{r:03d}.npz'), np.load(ref if ref.exists() else deposited / f'rep_{r:03d}.npz')
    d = max(float(np.nanmax(np.abs(A[k] - C[k]))) for k in A.files if k not in ('scalar_names', 'fingerprint'))
    rows.append(dict(replicate=r, fingerprint_local=str(A['fingerprint']), fingerprint_deposited=str(C['fingerprint']),
                     max_abs_diff=d))
    print(rows[-1])
shutil.rmtree(tmp)
out = dict(local_platform=platform.platform(), python=platform.python_version(), replicates=rows,
           max_abs_diff=max(r['max_abs_diff'] for r in rows))
json.dump(out, open(deposited.parent / 'platform' / 'local_vs_cluster.json', 'w'), indent=1)
