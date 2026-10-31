"""Export a short-horizon instance of the DP for the independent Mathematica re-solve (checks/dp_check.wl).

All inputs are exact thousandths (integers / 1000) taken from the fitted model, so Python (float64)
and Mathematica (exact rationals) solve the same game. Also exports, for the full fitted model, the
quantities in the last-FT break-even identity at a set of states.

usage: python -m nbaeg.checks       (writes checks/dp_small_inputs.json, checks/dp_small_python.json,
                                     checks/ft_identity_python.json)
"""
import json

import numpy as np

from .build import ROOT
from .dp import DM, ix, params_fitted, solve
from .maps import ft_mix

CHK = ROOT / 'checks'
NS = 14                                    # horizon of the small instance, in steps


def thousandths(x):
    return np.round(np.asarray(x, float) * 1000).astype(int)


def export_small():
    CHK.mkdir(exist_ok=True)
    full = params_fitted()
    inp = {}
    for k in ('M2', 'M3', 'X2'):
        inp[k] = thousandths(full[k][:, :NS])
    other = int(thousandths(full['p_sf']) + thousandths(full['p_tov']))
    shot = inp['M2'] + inp['X2']                            # P(shot got off), shared by k = 2, 3
    inp['X3'] = shot - inp['M3']
    inp['NONE'] = 1000 - other - shot
    for k in ('p_sf', 'p_tov', 'rho_fg', 'th_fg', 'eta_fg', 'rho_ft', 'rho_dm', 'th_ft', 'th_dm', 'eta_ft', 'eta_dm', 'q', 'p_sf3'):
        inp[k] = int(thousandths(full[k]))
    inp['p_pre'] = 150                                  # the main model has p_pre = 0: exercise the branch here
    inp['ALLOW2'] = full['ALLOW2'][:, :NS].astype(int)
    inp['FE'] = [3, 2, 2, 2, 2, 2]
    inp['C_TO'], inp['RT'] = 2, 2                       # C_TO = 2: FC14T keeps a 7-step shot clock
    inp['RTO'], inp['WO'] = [2], [1000]                 # ordinary FT misses: one delay
    inp['RTD'], inp['WD'] = [1, 2], [400, 600]          # deliberate misses: a two-point delay distribution
    inp['ot'] = 500
    inp['SC'] = 9                                       # shot-clock reset after OREB, exercised at n > 9
    assert (inp['NONE'] >= 0).all() and (inp['X3'] >= 0).all()
    js = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in inp.items()}
    js.update(NS=NS, DM=DM)
    json.dump(js, open(CHK / 'dp_small_inputs.json', 'w'))
    par = {}
    for k, v in inp.items():
        if k in ('ALLOW2',):
            par[k] = np.asarray(v, np.bool_)
        elif k == 'FE':
            par[k] = np.asarray(v, np.int64)
        elif k in ('C_TO', 'RT', 'SC'):
            par[k] = int(v)
        elif k in ('RTO', 'RTD'):
            par[k] = np.asarray(v, np.int64)
        elif k in ('WO', 'WD'):
            par[k] = np.asarray(v, float) / 1000
        elif isinstance(v, np.ndarray):
            par[k] = v.astype(float) / 1000
        else:
            par[k] = v / 1000
    S = solve(par)
    rows = []
    for n in (NS // 2, NS):
        for d in range(-4, 5):
            for s in range(6):
                for a in range(3):
                    for b in range(3):
                        for ga in range(2):
                            for gb in range(2):
                                rows.append([n, d, s, a, b, ga, gb, float(S['V'][n, d + DM, s, a, b, ga, gb])])
    ft = [[n, d, a, b, float(S['FMK'][n, d + DM, a, b, 0, 0]), float(S['FMS'][n, d + DM, a, b, 0, 0])]
          for n in (NS // 2, NS) for d in range(-4, 5) for a in range(3) for b in range(3)]
    json.dump(dict(V=rows, FT=ft), open(CHK / 'dp_small_python.json', 'w'))
    print('small instance:', len(rows), 'V states,', len(ft), 'FT states')


def export_ft_identity():
    """FMS - FMK and its parts (mk; kept / lost values of an ordinary and of a deliberate miss, each at its own
    rebound delay) in the fitted model, for the break-even identity."""
    par = params_fitted()
    S = solve(par)
    V = S['V']
    out = []
    for t in (1.0, 2.0, 3.0, 5.0, 8.0, 12.0):
        n = ix(t)
        for d in (-3, -2, -1, 1, 2):
            for a in range(3):
                mk = 1 - V[n, DM - (d + 1), 0, 0, a, 0, 0]
                kdm, ldm = (float(x) for x in ft_mix(V, n, par['RTD'], par['WD'], DM + d, a, 0, par['th_dm'], par['eta_dm'], par['SC']))
                kft, lft = (float(x) for x in ft_mix(V, n, par['RTO'], par['WO'], DM + d, a, 0, par['th_ft'], par['eta_ft'], par['SC']))
                out.append(dict(t=t, d=d, a=a, mk=float(mk), kdm=kdm, kft=kft, ldm=ldm, lft=lft,
                                fmk=float(S['FMK'][n, DM + d, a, 0, 0, 0]), fms=float(S['FMS'][n, DM + d, a, 0, 0, 0])))
    json.dump(dict(q=par['q'], rho_ft=par['rho_ft'], rho_dm=par['rho_dm'], rows=out),
              open(CHK / 'ft_identity_python.json', 'w'), indent=0)
    print('ft identity rows:', len(out))


if __name__ == '__main__':
    export_small()
    export_ft_identity()
