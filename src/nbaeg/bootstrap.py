"""Point maps (fitted, hand-set) and a game-clustered bootstrap of the whole pipeline.

Each replicate draws Poisson(1) weights per game, re-estimates every input (curves and scalars),
re-solves the game and stores the map summaries. Replicates are independent files, so a run resumes.

usage: python -m nbaeg.bootstrap point
       python -m nbaeg.bootstrap run B [workers] [first]
       python -m nbaeg.bootstrap summary
"""
import json
from pathlib import Path
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

from .dp import params_fitted, params_hand, solve
from .estimate import RES, TYPES, curve_sample, estimate, load, oreb_sample
from .maps import summarize

BOOT = RES / 'boot'
_G = {}


def point():
    for name, par in (('fitted', params_fitted()), ('hand', params_hand())):
        m = summarize(solve(par), par)
        np.savez(RES / f'maps_{name}.npz', **m)
        print('wrote maps', name)


SRC_FILES = ['estimate.py', 'dp.py', 'maps.py', 'bootstrap.py']


def fingerprint():
    """Hash of the code that generates a replicate and of its input tables; stored in every replicate."""
    import hashlib
    h = hashlib.sha256()
    here = Path(__file__).resolve().parent
    for f in SRC_FILES:
        h.update((here / f).read_bytes())
    from .build import DER
    for f in ('possessions.parquet', 'last_ft.parquet'):
        h.update((DER / f).read_bytes())
    return h.hexdigest()[:16]


def _init():
    P, F = load()
    _G['fp'] = fingerprint()
    _G.update(P=P, F=F, C=curve_sample(P), O=oreb_sample(P), games=np.sort(P.gameId.unique()))


def one(r):
    out = BOOT / f'rep_{r:03d}.npz'
    if out.exists():
        with np.load(out) as z:
            if 'fingerprint' in z.files and str(z['fingerprint']) == _G['fp']:
                return r                                # reuse only a replicate made by this code and data
    rng = np.random.default_rng(20261031 + r)
    g = _G['games']
    w = dict(zip(g, rng.poisson(1.0, len(g)).astype(float)))
    sc, cv = estimate(_G['P'], _G['F'], w, _G['C'], _G['O'])
    par = params_fitted(sc, cv)
    m = summarize(solve(par), par)
    for name in TYPES:
        p = np.minimum(cv[f'pshot_{name}_1'], 1 - sc['p_sfoul'] - sc['p_tov'])    # capped, as in the solver
        m[f'make3_{name}'] = p * cv[f'f3_{name}_1']
        m[f'make2_{name}'] = p * cv[f'f2_{name}_1']
    m['scalars'] = np.array([sc[k] for k in sorted(sc)])
    m['scalar_names'] = np.array(sorted(sc))
    m['fingerprint'] = np.array(_G['fp'])
    np.savez(out, **m)
    return r


def run(B, workers=4, first=0):
    BOOT.mkdir(parents=True, exist_ok=True)
    with Pool(workers, initializer=_init) as p:
        for r in p.imap_unordered(one, range(first, first + B)):
            print('rep', r, flush=True)


def summary(B=200):
    """Summarise exactly replicates 0..B-1 (fails if any is missing) and write a provenance manifest."""
    files = [BOOT / f'rep_{r:03d}.npz' for r in range(B)]
    missing = [f.name for f in files if not f.exists()]
    if missing:
        raise SystemExit(f'{len(missing)} of {B} replicates missing, e.g. {missing[:3]}; run `make boot` first')
    Z = [np.load(f) for f in files]
    fp = fingerprint()
    bad = [f.name for f, z in zip(files, Z) if 'fingerprint' not in z.files or str(z['fingerprint']) != fp]
    if bad:
        raise SystemExit(f'{len(bad)} replicates were made by other code or data (e.g. {bad[:3]}); rerun `make boot`')
    keys = [k for k in Z[0].files if k not in ('scalar_names', 'fingerprint')]
    out = {'B': np.array(len(Z))}
    for k in keys:
        A = np.stack([z[k] for z in Z])
        out[f'{k}_lo'] = np.nanpercentile(A, 5, axis=0)
        out[f'{k}_md'] = np.nanpercentile(A, 50, axis=0)
        out[f'{k}_hi'] = np.nanpercentile(A, 95, axis=0)
        if k in ('foul', 'ft', 'adv'):
            valid = ~np.isnan(A)                        # share of replicates, among those where the action exists
            with np.errstate(invalid='ignore', divide='ignore'):
                out[f'{k}_ppos'] = np.where(valid.sum(0) > 0, ((A > 0) & valid).sum(0) / valid.sum(0), np.nan)
    np.savez(RES / 'boot_summary.npz', **out)
    names = list(Z[0]['scalar_names'])
    A = np.stack([z['scalars'] for z in Z])
    est = json.load(open(RES / 'params_fitted.json'))
    tab = pd.DataFrame(dict(input=names, est=[est[n] for n in names], lo=np.percentile(A, 2.5, axis=0),
                            hi=np.percentile(A, 97.5, axis=0)))
    tab.to_csv(RES / 'inputs_ci.csv', index=False, float_format='%.6f')
    import hashlib
    import subprocess
    try:
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        commit = (RES.parent / 'COMMIT').read_text().strip() if (RES.parent / 'COMMIT').exists() else 'unknown'
    manifest = dict(B=B, summarised_with_commit=commit, fingerprint=fp,
                    replicates={f.name: hashlib.sha256(f.read_bytes()).hexdigest()[:16] for f in files})
    json.dump(manifest, open(RES / 'boot_manifest.json', 'w'), indent=0)
    print(f'B = {len(Z)}')
    print(tab.round(4).to_string(index=False))


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'point':
        point()
    elif cmd == 'run':
        run(int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 4, int(sys.argv[4]) if len(sys.argv) > 4 else 0)
    elif cmd == 'summary':
        summary(int(sys.argv[2]) if len(sys.argv) > 2 else 200)
