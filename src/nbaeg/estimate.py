"""Estimate the end-game model inputs from the possession phases and the last-FT table.

`estimate(P, F, w)` returns (scalars, curves) for game weights w (None = point estimate; Poisson(1)
weights per game = one bootstrap replicate). Curves are on the DP time grid (0.1 s, 0.1..24 s).

Start types of a possession phase (the `s` index of the DP):
  0 BC   dead-ball inbound in the backcourt (after a made FG/FT, no timeout)
  1 FC   frontcourt inbound after a timeout in the last 2:00 of Q4/OT (the "advance")
  2 LIVE after a defensive rebound or a turnover
  3 OREB after an offensive rebound

usage: python -m nbaeg.estimate     (writes results/params_fitted.json, results/curves_fitted.npz,
                                     results/curve_bins.csv)
"""
import json
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy import dmatrix, build_design_matrices

from .build import DER, ROOT

RES = ROOT / 'results'
DT, TMAX = 0.1, 24.0
GRID = np.round(np.arange(1, int(round(TMAX / DT)) + 1) * DT, 10)
TYPES = ['BC', 'FC', 'LIVE', 'OREB']
SHOOTING = {'Shooting', 'Shooting Block'}
FORCED = 1.0       # a shot with at most this many seconds left in the period was forced by the clock
warnings.filterwarnings('ignore')


# ----------------------------------------------------------------------------- samples
def curve_sample(P: pd.DataFrame, live_only=True) -> pd.DataFrame:
    """End-of-period phases for the rushed-shot curve.

    Q1-Q3: every phase with <= 24 s left (the last shot of a quarter is not score-driven).
    Q4/OT: phases of a side that is tied or trails by 1-3 (it must score): the 'pressure' sample.
    Phases ending in a non-shooting defensive foul (deliberate fouls) or with a side-out are dropped:
    the curve describes what the offence gets when the defence lets it play. Phases that start with a team
    rebound (a dead ball, restarted by an inbound) are dropped unless live_only=False: the OREB and LIVE curves
    describe live play after a rebound, as in the game.
    """
    x = P[(P.t0 <= TMAX) & (P.action != 'other') & (P.sideouts == 0)]
    if live_only:
        x = x[~x.reb_team]
    x = x[(x.period <= 3) | x.lead0.between(-3, 0)]
    x = x[~((x.action == 'foul') & ~x.foul_type.isin(SHOOTING))].copy()
    has_to = x.to_O > 0
    late2 = (x.period >= 4) & (x.t_to_O <= 120)
    x['s'] = np.select([has_to & late2, has_to, x.start.isin(['fg', 'ft']), x.start.isin(['dreb', 'tov'])],
                       [1, 0, 0, 2], 3)
    x['a'] = np.where(has_to, x.t_to_O, x.t0)
    x = x[(x.a > 0) & (x.a <= TMAX)]
    x['press'] = (x.period >= 4).astype(int)
    x['shot'] = (x.action == 'shot').astype(int)
    x['sfoul'] = (x.action == 'foul').astype(int)
    x['tov'] = (x.action == 'tov').astype(int)
    x['u'] = (x.a - x.t_act).clip(lower=DT)              # time used before the terminal action
    x['forced'] = (x.t_act <= FORCED).astype(int)        # the action came in the last second of the period
    return x[['gameId', 'season', 'period', 's', 'a', 'u', 't_act', 'forced', 'press', 'shot', 'sfoul', 'tov', 'fg3',
              'made', 'n_ft', 'lead0']]


def oreb_sample(P: pd.DataFrame) -> pd.DataFrame:
    """Missed FGs in Q4/OT with <= 24 s left and who got the ball next (same side = offensive rebound).
    Misses with <= 1 s left are dropped: the buzzer team rebound is credited to either side arbitrarily."""
    P = P.sort_values(['gameId', 'period', 'seq'])
    nxt = P.groupby(['gameId', 'period'])[['O', 'start', 't0']].shift(-1)
    m = (P.period >= 4) & (P.t0 <= TMAX) & (P.t_act > 1) & (P.action == 'shot') & (P.made == False) & nxt.start.isin(['oreb', 'dreb'])
    x = P.loc[m, ['gameId', 'season', 't_act']].copy()
    x['oreb'] = (nxt.loc[m, 'start'] == 'oreb').astype(int)
    x['team'] = P.groupby(['gameId', 'period'])['reb_team'].shift(-1).loc[m].astype(float)   # dead-ball team rebound
    x['reb_time'] = x.t_act - nxt.loc[m, 't0']
    return x


# ----------------------------------------------------------------------------- helpers
def wmean(y, w):
    y = np.asarray(y, float)
    w = np.ones_like(y) if w is None else np.asarray(w, float)
    ok = ~np.isnan(y)
    return float(np.sum(w[ok] * y[ok]) / np.sum(w[ok]))


SPL = "cr(np.log(a), df=4, constraints='center')"   # centred: no collinearity with the intercept


def fit_logit(d: pd.DataFrame, y: str, w, press_term=True):
    """Weighted logistic GLM y ~ natural cubic spline of log(a) (+ pressure); returns a predictor.
    (The identification variants pass a frame whose column `a` holds the time used.)"""
    form = SPL + (' + press' if press_term else '')
    X = dmatrix(form, d, return_type='dataframe')
    fw = None if w is None else np.asarray(w, float)
    res = sm.GLM(d[y].astype(float).values, X.values, family=sm.families.Binomial(), freq_weights=fw).fit()
    if not res.converged or np.linalg.matrix_rank(X.values) < X.shape[1]:
        raise RuntimeError(f'logit for {y} did not converge or is rank-deficient')
    info = X.design_info

    def pred(a, press):
        nd = pd.DataFrame({'a': a, 'press': press})
        return res.predict(np.asarray(build_design_matrices([info], nd)[0]))
    return pred


def wts(df, w):
    return None if w is None else df.gameId.map(w).fillna(0).values


# ----------------------------------------------------------------------------- estimation
def estimate(P: pd.DataFrame, F: pd.DataFrame, w: dict | None = None, C=None, O=None, curve='a'):
    """curve: 'a'      FG% of 2s and 3s as functions of the time available a (main analysis);
              'forced' FG% fitted on forced shots only (t_act <= FORCED, so the time used is about a);
              'used'   FG% as functions of the time used u = a - t_act, all shots.
    The attempt probability is always a function of a (identification check, Section 4.1)."""
    C = curve_sample(P) if C is None else C
    O = oreb_sample(P) if O is None else O
    sc, cv = {}, {}
    # rushed-shot curves by start type: P(shot attempted | a), FG% of 3s and of 2s given a
    for s, name in enumerate(TYPES):
        d = C[C.s == s]
        pt = name != 'FC'                      # FC phases are all Q4/OT: no pressure contrast
        sh = d[d.shot == 1]
        if curve == 'forced':
            sh = sh[sh.forced == 1]
        elif curve == 'used':
            sh = sh.assign(a=sh.u)
        s3, s2 = sh[sh.fg3 == True], sh[sh.fg3 == False]
        p_shot = fit_logit(d, 'shot', wts(d, w), pt)
        f3 = fit_logit(s3, 'made', wts(s3, w), pt)
        f2 = fit_logit(s2, 'made', wts(s2, w), pt)
        sh = d[d.shot == 1].assign(two=(d[d.shot == 1].fg3 == False).astype(int))   # always all shots, in a
        share2 = fit_logit(sh, 'two', wts(sh, w), pt)
        for press in (1, 0):
            pr = np.full(len(GRID), press)
            cv[f'pshot_{name}_{press}'] = p_shot(GRID, pr)
            cv[f'f3_{name}_{press}'] = f3(GRID, pr)
            cv[f'f2_{name}_{press}'] = f2(GRID, pr)
            cv[f'share2_{name}_{press}'] = share2(GRID, pr)
    pc = C[(C.press == 1) & (C.a >= 3)]
    sc['p_sfoul'] = wmean(pc.sfoul, wts(pc, w))
    sc['p_tov'] = wmean(pc.tov, wts(pc, w))
    sc['rho_fg'] = wmean(O.oreb, wts(O, w))
    sc['reb_time'] = wmean(O.reb_time.clip(0, 5), wts(O, w))
    # team rebounds (dead balls) among kept and lost missed FGs: a frontcourt inbound of the shooting team / a
    # backcourt inbound of the opponent instead of a live rebound
    kf, lf = O[O.oreb == 1], O[O.oreb == 0]
    sc['th_fg'] = wmean(kf.team, wts(kf, w))
    sc['eta_fg'] = wmean(lf.team, wts(lf, w))
    # missed last FTs: ordinary (>= 8 s left) and the deliberate-miss proxy (down 2-3 before, <= 3 s left)
    # rebounds logged at 0.0 (after the buzzer) are credited arbitrarily and carry no continuation: excluded
    fm = F[(F.made == 0) & F.oreb.notna() & (F.t_reb > 0)]
    ordn = fm[fm.t >= 8]
    dm = fm[fm.lead_b.isin([-2, -3]) & (fm.t <= 3) & fm.pct.notna()]   # same cohort as the mixture correction
    sc['rho_ft'] = wmean(ordn.oreb, wts(ordn, w))
    sc['rho_dm'] = wmean(dm.oreb, wts(dm, w))
    # kept rebounds that are team rebounds (dead ball out of bounds: the team inbounds in the frontcourt)
    for k, x in (('th_ft', ordn), ('th_dm', dm)):
        kept = x[x.oreb == 1]
        sc[k] = wmean(kept.reb_team, wts(kept, w))
    # lost rebounds that are team rebounds to the defence (out of bounds off the shooting team: the opponent
    # inbounds a dead ball in its backcourt)
    for k, x in (('eta_ft', ordn), ('eta_dm', dm)):
        lost = x[x.oreb == 0]
        sc[k] = wmean(lost.reb_team, wts(lost, w))
    sc['rho_dm_live'] = wmean(dm.oreb * (1 - dm.reb_team), wts(dm, w))     # live offensive rebounds only
    for k, m in (('rho_dm_down2', dm.lead_b == -2), ('rho_dm_down3', dm.lead_b == -3),
                 ('rho_dm_2000_16', dm.season <= 2016), ('rho_dm_2017_25', dm.season >= 2017)):
        sc[k] = wmean(dm.oreb[m], wts(dm[m], w))                          # transport across states and eras
    # the proxy mixes deliberate and accidental misses; with w the deliberate share among misses (excess miss
    # rate over the rate expected from season FT%), raw = w*rho_dm + (1-w)*rho_ft, so raw is a lower bound
    # for rho_dm (when rho_dm >= rho_ft) and the mixture-corrected rate is (raw - (1-w)*rho_ft) / w
    tr = F[F.lead_b.isin([-2, -3]) & (F.t <= 3) & F.pct.notna() & ~((F.made == 0) & ~(F.t_reb > 0))]
    tw = wts(tr, w)
    miss = 1 - wmean(tr.made, tw)
    w_dm = max(1e-6, (miss - wmean(1 - tr.pct, tw)) / miss)
    sc['w_dm'] = w_dm
    sc['rho_dm_mix'] = (sc['rho_dm'] - (1 - w_dm) * sc['rho_ft']) / w_dm
    sc['reb_time_ft'] = wmean((dm.t - dm.t_reb).clip(0, 3), wts(dm, w))           # deliberate misses (proxy)
    sc['reb_time_ft_ord'] = wmean((ordn.t - ordn.t_reb).clip(0, 3), wts(ordn, w))   # ordinary misses
    # FT% (leave-one-out season FT%) of the players who shoot late last FTs
    FL = F[F.t <= TMAX]
    sc['q'] = wmean(FL.pct, wts(FL, w))
    # deliberate fouls: Q4/OT phases with <= 24 s left where the defence fouls without a shot
    Q = P[(P.period >= 4) & (P.t0 <= TMAX) & (P.lead0 != 0)]
    fl = Q[((Q.action == 'foul') & ~Q.foul_type.isin(SHOOTING)) | (Q.sideouts > 0)].copy()
    fl['t_foul'] = np.where(fl.sideouts > 0, fl.t_side, fl.t_act)
    fl['frm'] = np.where(fl.to_O > 0, fl.t_to_O, fl.t0)
    fl = fl[fl.t_foul <= fl.frm]
    fl['el'] = fl.frm - fl.t_foul
    for name, m in [('BC', fl.start.isin(['fg', 'ft']) & (fl.to_O == 0)), ('FC', fl.to_O > 0),
                    ('LIVE', fl.start.isin(['dreb', 'tov']) & (fl.to_O == 0)),
                    ('OREB', (fl.start == 'oreb') & (fl.to_O == 0))]:
        x = fl[m]
        sc[f'foul_el_{name}'] = wmean(x.el.clip(0, 6), wts(x, w)) if len(x) > 20 else 1.0
    # an intentional foul on a side down 3 called as a shooting foul on a 3 (fouls within 2 s of the start)
    up3 = Q[(Q.lead0 == -3) & (Q.action == 'foul') & ((Q.t0 - Q.t_act) <= 2)]
    sc['p_sf3'] = wmean((up3.n_ft == 3).astype(float), wts(up3, w))
    # time a live-ball team needs to call a timeout (start to timeout), last 2:00 of Q4/OT
    lv = Q[Q.start.isin(['dreb', 'tov', 'oreb']) & (Q.to_O > 0) & (Q.t_to_O <= 120)]
    sc['to_cost_live'] = wmean((lv.t0 - lv.t_to_O).clip(0, 5), wts(lv, w))
    return sc, cv


# ----------------------------------------------------------------------------- outputs
def wilson(k, n, z=1.96):
    ph = k / n
    c = (ph + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return ph, c - h, c + h


def curve_bins(C: pd.DataFrame) -> pd.DataFrame:
    """Binned empirical curve points (for the figure), with Wilson intervals."""
    bins = [0, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 21, 24]
    C = C.assign(b=pd.cut(C.a, bins))
    rows = []
    for (s, press, b), d in C.groupby(['s', 'press', 'b'], observed=True):
        sh = d[d.shot == 1]
        for k, sub in (('pshot', d.shot), ('f3', sh[sh.fg3 == True].made), ('f2', sh[sh.fg3 == False].made)):
            n = len(sub)
            if n >= 15:
                ph, lo, hi = wilson(float(sub.sum()), n)
                rows.append(dict(s=TYPES[s], press=press, a_mid=d.a.median(), what=k, n=n, p=ph, lo=lo, hi=hi))
    return pd.DataFrame(rows)


def load():
    return pd.read_parquet(DER / 'possessions.parquet'), pd.read_parquet(DER / 'last_ft.parquet')


def main():
    RES.mkdir(exist_ok=True)
    P, F = load()
    C = curve_sample(P)
    sc, cv = estimate(P, F, None, C)
    json.dump(sc, open(RES / 'params_fitted.json', 'w'), indent=1)
    np.savez(RES / 'curves_fitted.npz', grid=GRID, **cv)
    curve_bins(C).to_csv(RES / 'curve_bins.csv', index=False)
    print('curve sample sizes (s, press):', C.groupby(['s', 'press']).size().to_dict())
    for k, v in sc.items():
        print(f'{k:14s} {v:.4f}')
    for name in TYPES:
        p = cv[f'pshot_{name}_1']
        idx = [int(round(a / DT)) - 1 for a in (1, 2, 4, 8, 16)]
        print(name, 'press=1 at a=1,2,4,8,16 s: P(made 3)=', np.round((p * cv[f'f3_{name}_1'])[idx], 3),
              ' P(made 2)=', np.round((p * cv[f'f2_{name}_1'])[idx], 3))


if __name__ == '__main__':
    main()
