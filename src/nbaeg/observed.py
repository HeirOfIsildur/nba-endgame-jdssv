"""Observed coach behaviour in the states of the decision maps, by era, and a calibration check.

Eras: 'pre' = 2000-01..2016-17 (2006-07..2016-17 for timeout-dependent tables), 'post' = 2017-18..2025-26.
  obs_foul.csv   leader up 1-3, Q4, <= 24 s, trailer's phase (not after an OREB): share of phases in which
                 the leader fouls deliberately (non-shooting foul or side-out within 3 s of the decision point)
                 before the trailer acts;
                 decision time = the trailer's timeout if it called one (frontcourt, s = 1), else t0 (s = 0).
  obs_ft.csv     last FT of a 2-/3-shot trip, Q4, <= 24 s: miss rate and the expected miss rate
                 (1 - shooter's leave-one-out season FT%), by the shooter's lead and timeouts left.
  obs_shot.csv   trailer down 1-3, Q4, <= 24 s: share of its FGAs that are 3s, by the time of the shot.
  calib.csv      Q4 phases with lead -3..3 and <= 24 s: fitted-model win probability vs the outcome.
usage: python -m nbaeg.observed
"""
import numpy as np
import pandas as pd

from .dp import DM, ix, params_fitted, solve
from .estimate import RES, SHOOTING, load, wilson

TB = [0, 3, 6, 9, 12, 16, 20, 24]
TB_FT = [0, 1, 2, 3, 5, 8, 12, 24]


def era(season):
    return np.where(season >= 2017, 'post', 'pre')


def rate_rows(df, keys, y, bins, tcol):
    df = df.assign(tb=pd.cut(df[tcol], bins))
    rows = []
    for k, g in df.groupby(keys + ['tb'], observed=True):
        ph, lo, hi = wilson(float(g[y].sum()), len(g))
        rows.append(dict(zip(keys, k[:-1]), t_lo=k[-1].left, t_hi=k[-1].right, t_mid=g[tcol].median(),
                         n=len(g), rate=ph, lo=lo, hi=hi, won=g['won'].mean()))
    return pd.DataFrame(rows)


def main():
    P, F = load()
    Q = P[(P.period == 4) & (P.t0 <= 24) & (P.action != 'other')].copy()
    Q['era'] = era(Q.season)
    # the leader's deliberate foul
    # match the displayed states: leader without a foul to give; s = 0 is a dead-ball backcourt inbound
    # (after a made basket or free throw) with no trailer timeout, s = 1 a frontcourt inbound after one
    # (a) s = 0: dead-ball backcourt inbound, the trailer has no timeout left; (b) s = 1: the trailer calls its
    # last timeout and inbounds in the frontcourt. Timeout-dependent, so from 2006-07; pooled over the leader's
    # own timeouts (too few phases with none left before 2017-18)
    L = Q[Q.lead0.isin([-1, -2, -3]) & (Q.start != 'oreb') & (Q.ftg_D == 0) & (Q.season >= 2006)].copy()
    L['s'] = (L.to_O > 0).astype(int)
    L = L[((L.s == 0) & L.start.isin(['fg', 'ft']) & (L.tol_O == 0)) | ((L.s == 1) & (L.tol_O == 1))]
    L['t_dec'] = np.where(L.s == 1, L.t_to_O, L.t0)
    # deliberate = a non-shooting foul or side-out within 3 s of the decision point (the intentional-foul
    # window; later fouls in the possession are mostly incidental)
    t_foul = np.where(L.sideouts > 0, L.t_side, np.where((L.action == 'foul') & ~L.foul_type.isin(SHOOTING), L.t_act, np.nan))
    el = L.t_dec - t_foul
    L['fouled'] = ((el >= 0) & (el <= 3)).astype(int)
    L['lead'] = -L.lead0
    L['won'] = 1 - L.O_won                      # the leader won the game (after OT if any)
    rate_rows(L, ['era', 'lead', 's'], 'fouled', TB, 't_dec').to_csv(RES / 'obs_foul.csv', index=False)
    # the last free throw
    # timeout-dependent: seasons from 2006-07 (Section 2.3); pooled over the opponent's timeouts (too few
    # trips have an opponent without timeouts before 2017-18)
    G = F[(F.period == 4) & (F.t <= 24) & F.lead_b.isin([-3, -2, -1, 1, 2]) & (F.season >= 2006)].copy()
    G['era'] = era(G.season)
    G['miss'] = 1 - G.made
    G['tol'] = np.minimum(G.tol_S, 1)
    G['won'] = G.S_won
    ft = rate_rows(G, ['era', 'lead_b', 'tol'], 'miss', TB_FT, 't')
    ex = (1 - G.pct).groupby([G.era, G.lead_b, G.tol, pd.cut(G.t, TB_FT)], observed=True).mean()
    ft['expected'] = ex.values
    ft.to_csv(RES / 'obs_ft.csv', index=False)
    H = G[G.lead_b.isin([-2, -3]) & (G.t > 0.5) & (G.t <= 5)]
    H.groupby(['era', 'lead_b', 'tol', 'miss']).won.agg(['size', 'mean']).round(3).to_csv(RES / 'obs_ft_outcomes.csv')
    # the trailer's shot choice
    T = Q[Q.lead0.isin([-1, -2, -3]) & (Q.action == 'shot')].copy()
    T['three'] = T.fg3.astype(int)
    T['lead'] = T.lead0
    T['won'] = T.O_won
    rate_rows(T, ['era', 'lead'], 'three', TB, 't_act').to_csv(RES / 'obs_shot.csv', index=False)
    # calibration of the fitted game: predicted win probability of the team with the ball vs the outcome
    V = solve(params_fitted())['V']
    C = Q[Q.lead0.between(-3, 3) & (Q.t0 >= 0.1) & (Q.season >= 2006)].copy()   # timeout states from 2006-07
    C['s'] = C.start.map({'fg': 0, 'ft': 0, 'dreb': 2, 'tov': 2, 'oreb': 3})
    n = np.clip(np.round(C.t0 / 0.1).astype(int), 1, ix(24))
    C['pred'] = V[n, (DM + C.lead0).astype(int), C.s, np.minimum(C.tol_O, 2), np.minimum(C.tol_D, 2),
                  C.ftg_O, C.ftg_D]
    C['won'] = np.where(C.ot_after == 1, 0.5, C.O_won)    # OT valued at 1/2, as in the model
    C['pb'] = pd.cut(C.pred, np.linspace(0, 1, 11))
    cal = C.groupby(['era', 'pb'], observed=True).agg(n=('won', 'size'), pred=('pred', 'mean'),
                                                      obs=('won', 'mean')).reset_index()
    cal['pb'] = cal.pb.astype(str)
    cal.to_csv(RES / 'calib.csv', index=False)
    brier = float(np.mean((C.pred - C.won) ** 2))
    base = float(np.mean((C.won.mean() - C.won) ** 2))
    # score-and-time baseline: observed mean outcome by lead and 3-second bin (in-sample)
    cell = C.groupby([C.lead0, pd.cut(C.t0, np.arange(0, 25, 3))], observed=True).won.transform('mean')
    base_st = float(np.mean((cell - C.won) ** 2))
    pd.Series(dict(n=len(C), brier=brier, brier_const=base, brier_score_time=base_st, mean_pred=C.pred.mean(),
                   mean_obs=C.won.mean())
              ).to_csv(RES / 'calib_summary.csv')
    print(cal.round(3).to_string(index=False))
    print(f'Brier {brier:.4f} vs constant {base:.4f}, n={len(C)}')


if __name__ == '__main__':
    main()
