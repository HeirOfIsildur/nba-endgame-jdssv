"""Sensitivity of the decisions to the modelling choices, and held-out calibration.

Each variant changes one input or modelling choice and re-solves the game (or re-estimates on a subset of
seasons). Outputs:
  results/sensitivity.csv    one row per variant: the key decision quantities
  results/sens_curves.npz    the two headline curves for every variant (for the robustness figure)
  results/ident_check.csv    identification check: FG% of unforced vs forced shots by time used
  results/calib_holdout.csv  season-split calibration: inputs estimated on even (odd) seasons only, scored on
                             odd (even) seasons, against an outcome table built on the training seasons
usage: python -m nbaeg.sensitivity
"""
import json

import numpy as np
import pandas as pd

from .dp import DM, ix, params_fitted, solve
from .estimate import GRID, RES, TYPES, curve_sample, estimate, load
from .maps import FT_LEADS, summarize


def key_numbers(m):
    fc, bc = m['foul'][:, 2, 1, 0, 0], m['foul'][:, 2, 0, 0, 0]
    ft = m['ft'][:, FT_LEADS.index(-2), 0, 0]
    i = lambda t: ix(t) - 1
    drop = np.flatnonzero((GRID >= 2 - 1e-9) & (ft < 1.0))  # last grid point of the >= 1 pp run that starts at 2 s
    fc_neg = GRID[(GRID > 3) & (fc < 0)]
    return dict(foul_up3_BC_3=bc[i(3)], foul_up3_BC_5=bc[i(5)], foul_up3_FC_3=fc[i(3)], foul_up3_FC_5=fc[i(5)],
                foul_up3_FC_negative_from=fc_neg.min() if len(fc_neg) else np.nan,
                ft_down2_2=ft[i(2)], ft_down2_5=ft[i(5)], ft_down2_8=ft[i(8)],
                ft_down2_gain_ge1_until=(GRID[drop[0] - 1] if len(drop) else 24.0) if ft[i(2)] >= 1 else np.nan,
                timeout_value_down1_max=m['adv'][:, 1, 0].max())


def variants(sc, ci):
    lo = lambda k: float(ci.loc[k, 'lo'])
    hi = lambda k: float(ci.loc[k, 'hi'])
    fe = {f'foul_el_{t}': sc[f'foul_el_{t}'] for t in TYPES}
    rd, ro = sc['reb_time_ft'], sc['reb_time_ft_ord']
    return [
        ('main estimate', {}),
        (f'deliberate-miss OREB at its lower 95% bound ({lo("rho_dm"):.2f})', dict(rho_dm=lo('rho_dm'))),
        (f'deliberate-miss OREB mixture-corrected ({sc["rho_dm_mix"]:.2f})', dict(rho_dm=sc['rho_dm_mix'])),
        (f'deliberate-miss OREB = ordinary ({sc["rho_ft"]:.2f})', dict(rho_dm=sc['rho_ft'])),
        ('fouls take 0.5 s longer', {k: v + 0.5 for k, v in fe.items()}),
        ('fouls take 0.5 s less', {k: max(0.3, v - 0.5) for k, v in fe.items()}),
        # the offence beats a committed foul (not identifiable: an attempted foul that fails looks like an ordinary
        # quick shot); in up-3 phases offences shoot within the fitted foul time in about 8-11% of phases
        ('offence beats a committed foul with probability 0.1', dict(p_pre=0.1)),
        ('offence beats a committed foul with probability 0.2', dict(p_pre=0.2)),
        ('offence always beats a committed foul (probability 1, an extreme)', dict(p_pre=1.0)),
        ('two-point threshold 15%', dict(share2_min=0.15)),
        ('two-point threshold 35%', dict(share2_min=0.35)),
        ('no shot-clock reset after OREB', dict(shot_clock=1000.0)),
        ('score-neutral shot curves (no pressure shift)', dict(press=0)),
        (f'3-FT risk at its lower bound ({lo("p_sf3"):.2f})', dict(p_sf3=lo('p_sf3'))),
        (f'3-FT risk at its upper bound ({hi("p_sf3"):.2f})', dict(p_sf3=hi('p_sf3'))),
        (f'OREB on missed FG at {hi("rho_fg") + 0.05:.2f}', dict(rho_fg=hi('rho_fg') + 0.05)),
        (f'deliberate-miss OREB: live rebounds only ({sc["rho_dm_live"]:.2f})', dict(rho_dm=sc['rho_dm_live'], th_dm=0.0)),
        ('FT team rebounds (both teams) as live rebounds', dict(th_dm=0.0, th_ft=0.0, eta_dm=0.0, eta_ft=0.0)),
        (f'deliberate-miss OREB from down-3 trips only ({sc["rho_dm_down3"]:.2f})', dict(rho_dm=sc['rho_dm_down3'])),
        (f'one FT rebound time for all misses, the deliberate one ({rd:.2f} s)', dict(reb_time_ft_ord=rd)),
        (f'one FT rebound time for all misses, the ordinary one ({ro:.2f} s)', dict(reb_time_ft=ro)),
    ]


def _delay_dist(x):
    d = np.round((x.t - x.t_reb).clip(0, 3).values, 1)
    u, c = np.unique(d, return_counts=True)
    return u, c / c.sum()


def dm_delays(F):
    """FT-to-rebound delays of the deliberate-miss proxy (the estimation cohort, clipped at 3 s)."""
    fm = F[(F.made == 0) & F.oreb.notna() & (F.t_reb > 0)]
    return _delay_dist(fm[fm.lead_b.isin([-2, -3]) & (fm.t <= 3) & fm.pct.notna()])


def ord_delays(F):
    """FT-to-rebound delays of ordinary misses (>= 8 s left, so not selected by the remaining clock)."""
    fm = F[(F.made == 0) & F.oreb.notna() & (F.t_reb > 0)]
    return _delay_dist(fm[fm.t >= 8])


def rebound_starts(P):
    """For phases that start with a rebound: whether it is a team rebound (a dead ball: the restart is an inbound,
    not live play) and whether it followed a missed field goal or a missed free throw. Index aligned with P."""
    from .build import DER
    L = pd.read_parquet(DER / 'late.parquet', columns=['gameId', 'period', 't', 'seq', 'personId', 'is_fga', 'is_ft'])
    L = L.sort_values(['gameId', 'period', 't', 'seq'], ascending=[True, True, False, True]).reset_index(drop=True)
    kind = pd.Series(np.select([L.is_fga, L.is_ft], ['FG', 'FT'], ''), index=L.index).replace('', np.nan).ffill()
    R = P[P.start.isin(['oreb', 'dreb'])]
    return pd.DataFrame(dict(team=(L.personId.values[R.i_start] > 1e9), origin=kind.values[R.i_start]), index=R.index)


def rebound_state_audit(P, C, rs):
    """Counts of live and team (dead-ball) rebound starts entering the OREB and LIVE curve samples, by origin."""
    rows = []
    x = rs.loc[rs.index.intersection(C.index)].join(P.start)
    for (st, org), d in x.groupby(['start', 'origin']):
        rows.append(dict(start=st, origin=org, n_curve_sample=len(d), n_team_rebound=int(d.team.sum()),
                         share_team=d.team.mean()))
    return pd.DataFrame(rows)


UBINS = ((0.5, 2), (2, 4), (4, 6), (6, 10))


def ident_check(C, B=1000, seed=2027):
    """Identification check: FG% of shots taken after u seconds when much more time was left (unforced: at least
    4 s left in the period at the shot, so a >= u + 4) against forced shots (at most 1 s left, so u ~ a), by start
    type, shot value and bin of u; difference unforced - forced with a 90% game-bootstrap interval."""
    sh = C[C.shot == 1].copy()
    sh['grp'] = np.select([sh.forced == 1, sh.t_act >= 4], ['forced', 'unforced'], 'mid')
    sh = sh[sh.grp != 'mid']
    games = np.sort(C.gameId.unique())
    gi = pd.Series(np.arange(len(games)), index=games)
    W = np.random.default_rng(seed).poisson(1.0, (B, len(games))).astype(np.float32)
    rows = []
    for s, name in enumerate(TYPES):
        for k in (2, 3):
            for lo, hi in UBINS:
                x = sh[(sh.s == s) & ((sh.fg3 == True) if k == 3 else (sh.fg3 == False)) & (sh.u > lo) & (sh.u <= hi)]
                r = dict(start=name, k=k, u=f'({lo},{hi}]')
                stats = {}
                for g in ('forced', 'unforced'):
                    y = x[x.grp == g]
                    r[f'n_{g}'] = len(y)
                    r[f'fg_{g}'] = y.made.mean() if len(y) else np.nan
                    idx = gi[y.gameId].values
                    nn = np.bincount(idx, minlength=len(games)).astype(np.float32)
                    mm = np.bincount(idx, weights=y.made.astype(float).values, minlength=len(games)).astype(np.float32)
                    with np.errstate(invalid='ignore', divide='ignore'):
                        stats[g] = (W @ mm) / (W @ nn)
                r['diff'] = r['fg_unforced'] - r['fg_forced']
                bd = stats['unforced'] - stats['forced']
                ok = min(r['n_forced'], r['n_unforced']) >= 20
                r['diff_lo'] = np.nanpercentile(bd, 5) if ok else np.nan
                r['diff_hi'] = np.nanpercentile(bd, 95) if ok else np.nan
                rows.append(r)
    return pd.DataFrame(rows)


def outcome_frame(P, seasons):
    Q = P[(P.period == 4) & (P.t0 <= 24) & (P.t0 >= 0.1) & (P.action != 'other') & P.lead0.between(-3, 3)
          & P.season.isin(seasons)].copy()
    Q['s'] = Q.start.map({'fg': 0, 'ft': 0, 'dreb': 2, 'tov': 2, 'oreb': 3})
    Q['won'] = np.where(Q.ot_after == 1, 0.5, Q.O_won)
    Q['tb'] = pd.cut(Q.t0, np.arange(0, 25, 3))
    return Q


def predict(V, Q):
    n = np.clip(np.round(Q.t0 / 0.1).astype(int), 1, ix(24))
    return V[n, (DM + Q.lead0).astype(int), Q.s, np.minimum(Q.tol_O, 2), np.minimum(Q.tol_D, 2), Q.ftg_O, Q.ftg_D]


def calib_holdout(P, F):
    rows = []
    seasons = np.arange(2006, 2026)
    for train_par in (0, 1):
        train_all = np.arange(2000, 2026)[np.arange(2000, 2026) % 2 == train_par]
        train = seasons[seasons % 2 == train_par]
        test = seasons[seasons % 2 != train_par]
        sc, cv = estimate(P[P.season.isin(train_all)], F[F.season.isin(train_all)])
        V = solve(params_fitted(sc, cv))['V']
        Qtr, Qte = outcome_frame(P, train), outcome_frame(P, test)
        Qte['pred'] = predict(V, Qte)
        table = Qtr.groupby(['lead0', 'tb'], observed=True).won.mean()
        Qte['table'] = [table.get((l, b), Qtr.won.mean()) for l, b in zip(Qte.lead0, Qte.tb)]
        rows.append(dict(train='even seasons' if train_par == 0 else 'odd seasons', n_test=len(Qte),
                         brier_model=np.mean((Qte.pred - Qte.won) ** 2),
                         brier_table=np.mean((Qte.table - Qte.won) ** 2),
                         brier_const=np.mean((Qtr.won.mean() - Qte.won) ** 2),
                         mean_pred=Qte.pred.mean(), mean_obs=Qte.won.mean()))
    return pd.DataFrame(rows)


def rho_dm_bins(F, B=1000, seed=2026):
    """Deliberate-miss rebound rate by time window (shooter down 2-3 before the last FT): raw OREB share of
    misses (a lower bound) and the mixture-corrected rate, with game-bootstrap 95% intervals."""
    rft_pool = F[(F.made == 0) & F.oreb.notna() & (F.t_reb > 0) & (F.t >= 8)]
    rows = []
    rng = np.random.default_rng(seed)
    for lo, hi in ((0, 1), (1, 2), (2, 3), (3, 5), (5, 8), (0, 3), (0, 5), (0, 8)):
        x = F[F.lead_b.isin([-2, -3]) & (F.t > lo) & (F.t <= hi) & F.pct.notna()
              & ~((F.made == 0) & ~(F.t_reb > 0))]
        games = np.union1d(x.gameId.unique(), rft_pool.gameId.unique())   # one weight per game for both samples

        def stat(wx, wr):
            m = (1 - x.made.values)
            miss = np.sum(wx * m) / np.sum(wx)
            exp = np.sum(wx * (1 - x.pct.values)) / np.sum(wx)
            w = max(1e-6, (miss - exp) / miss)
            mm = m == 1
            raw = np.sum(wx[mm] * x.oreb.values[mm]) / np.sum(wx[mm])
            rft = np.sum(wr * rft_pool.oreb.values) / np.sum(wr)
            return raw, w, (raw - (1 - w) * rft) / w
        est = stat(np.ones(len(x)), np.ones(len(rft_pool)))
        bs = []
        for _ in range(B):
            gw = pd.Series(rng.poisson(1.0, len(games)).astype(float), index=games)
            bs.append(stat(x.gameId.map(gw).values, rft_pool.gameId.map(gw).values))
        bs = np.array(bs)
        rows.append(dict(window=f'({lo},{hi}]', n_trips=len(x), n_misses=int((x.made == 0).sum()), raw=est[0],
                         raw_lo=np.nanpercentile(bs[:, 0], 2.5), raw_hi=np.nanpercentile(bs[:, 0], 97.5), w=est[1],
                         mix=est[2], mix_lo=np.nanpercentile(bs[:, 2], 2.5), mix_hi=np.nanpercentile(bs[:, 2], 97.5)))
    return pd.DataFrame(rows)


def main():
    P, F = load()
    sc = json.load(open(RES / 'params_fitted.json'))
    z = np.load(RES / 'curves_fitted.npz')
    cv = {k: z[k] for k in z.files}
    ci = pd.read_csv(RES / 'inputs_ci.csv').set_index('input')
    rows, curves = [], {}

    def add(lab, par):
        m = summarize(solve(par), par)
        rows.append(dict(variant=lab, **key_numbers(m)))
        curves[lab] = (m['foul'][:, 2, 1, 0, 0], m['ft'][:, FT_LEADS.index(-2), 0, 0])
        print('done:', lab, flush=True)

    for lab, over in variants(sc, ci):
        add(lab, params_fitted(sc, cv, **over))
    C = curve_sample(P)
    rs = rebound_starts(P)
    rebound_state_audit(P, curve_sample(P, live_only=False), rs).to_csv(RES / 'reb_state_audit.csv', index=False, float_format='%.4f')
    s2, c2 = estimate(P, F, None, curve_sample(P, live_only=False))
    add('rebound curves include team-rebound starts; FG team rebounds live (previous treatment)',
        params_fitted(s2, c2, th_fg=0.0, eta_fg=0.0))
    for lab, key, x in (('deliberate-miss FT rebound delay: empirical distribution of the proxy', 'rtd_dist', dm_delays(F)),
                        ('deliberate-miss FT rebound delay: distribution of ordinary misses (not clock-selected)',
                         'rtd_dist', ord_delays(F))):
        add(lab, params_fitted(sc, cv, **{key: x}))
    for lab, mode in (('FG% from forced shots only (identification)', 'forced'),
                      ('FG% by time used, all shots (identification)', 'used')):
        s2, c2 = estimate(P, F, None, C, curve=mode)
        add(lab, params_fitted(s2, c2))
    for lab, yrs in (('all inputs from 2000-01 to 2016-17 only', P.season <= 2016),
                     ('all inputs from 2017-18 to 2025-26 only', P.season >= 2017)):
        s2, c2 = estimate(P[yrs], F[F.season.isin(P[yrs].season.unique())])
        add(lab, params_fitted(s2, c2))
    tab = pd.DataFrame(rows)
    tab.to_csv(RES / 'sensitivity.csv', index=False, float_format='%.3f')
    labels = list(curves)
    np.savez(RES / 'sens_curves.npz', labels=np.array(labels), foul=np.stack([curves[k][0] for k in labels]),
             ft=np.stack([curves[k][1] for k in labels]))
    ic = ident_check(C)
    ic.to_csv(RES / 'ident_check.csv', index=False, float_format='%.4f')
    print(ic.round(3).to_string(index=False))
    rb = rho_dm_bins(F)
    rb.to_csv(RES / 'rho_dm_bins.csv', index=False, float_format='%.3f')
    print(rb.round(3).to_string(index=False))
    cal = calib_holdout(P, F)
    cal.to_csv(RES / 'calib_holdout.csv', index=False, float_format='%.4f')
    print(tab.round(2).to_string(index=False))
    print(cal.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
