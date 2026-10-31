"""Every number quoted in the paper, regenerated from results/ (writes results/headline.txt).

usage: python -m nbaeg.headline
"""
import json

import numpy as np
import pandas as pd

from .estimate import DT, RES
from .maps import FT_LEADS

L = []


def p(s=''):
    print(s)
    L.append(s)


def i(t):
    return int(round(t / DT)) - 1          # map arrays start at t = DT


def band(B, key, idx):
    if B is None:
        return ''
    return f' [{B[key + "_lo"][idx]:+.1f}, {B[key + "_hi"][idx]:+.1f}]'


def crossings(x, grid):
    s = np.sign(np.round(x, 6))
    return [round(float(grid[k + 1]), 1) for k in range(len(s) - 1) if s[k] != s[k + 1] and s[k + 1] != 0]


def ident_table(ic):
    """LaTeX rows of the identification-check table (Appendix), written to results/ident_table.tex."""
    lab = {'BC': 'backcourt inbound', 'FC': 'frontcourt after timeout', 'LIVE': 'live ball', 'OREB': 'offensive rebound'}
    bins = list(dict.fromkeys(ic.u))
    out = []
    for k in (3, 2):
        for st in ('BC', 'FC', 'LIVE', 'OREB'):
            x = ic[(ic.k == k) & (ic.start == st)].set_index('u')
            cells = []
            for b in bins:
                r = x.loc[b]
                if np.isnan(r.diff_lo):
                    cells.append('--')
                else:
                    f = lambda v: f'{int(round(100 * v)):+d}'.replace('-', '$-$')
                    cells.append(f'{f(r["diff"])} ({f(r.diff_lo)}, {f(r.diff_hi)})')
            out.append(f'{k} & {lab[st]} & ' + ' & '.join(cells) + r' \\')
    head = ['\\begin{tabular}{llcccc}', '\\toprule',
            '$k$ & Start & ' + ' & '.join(b.strip('()]').replace(',', '--') + ' s' for b in bins) + ' \\\\', '\\midrule']
    (RES / 'ident_table.tex').write_text('\n'.join(head + out + ['\\bottomrule', '\\end{tabular}']) + '\n')


def sens_table(sv):
    """LaTeX rows of the sensitivity table (Appendix B), written to results/sens_table.tex."""
    tex = lambda x: x.replace('%', r'\%').replace('>=', r'$\ge$').replace('_', r'\_')
    f = lambda v: f'{v:+.1f}'.replace('-', '$-$')
    out = ['\\begin{tabular}{p{7.4cm}rrrrrr}', '\\toprule',
           r'Variant & \multicolumn{2}{c}{Foul up 3, 3 s} & \multicolumn{2}{c}{Miss down 2} & $\ge$1 pp & Timeout \\',
           r' & \FC & \BC & 2 s & 5 s & until (s) & max \\', '\\midrule']
    for r in sv.itertuples():
        out.append(f'{tex(r.variant)} & {f(r.foul_up3_FC_3)} & {f(r.foul_up3_BC_3)} & {f(r.ft_down2_2)} & '
                   f'{f(r.ft_down2_5)} & {r.ft_down2_gain_ge1_until:.1f} & {r.timeout_value_down1_max:.1f} \\\\')
    (RES / 'sens_table.tex').write_text('\n'.join(out + ['\\bottomrule', '\\end{tabular}']) + '\n')


def runs_of(ok, grid):
    out, st = [], None
    for k, v in enumerate(ok):
        if v and st is None:
            st = k
        if (not v or k == len(ok) - 1) and st is not None:
            out.append((grid[st], grid[k - 1 if not v else k]))
            st = None
    return out


def main():
    F = dict(np.load(RES / 'maps_fitted.npz'))
    H = dict(np.load(RES / 'maps_hand.npz'))
    B = dict(np.load(RES / 'boot_summary.npz')) if (RES / 'boot_summary.npz').exists() else None
    cv = dict(np.load(RES / 'curves_fitted.npz'))
    grid = cv['grid']
    ci = pd.read_csv(RES / 'inputs_ci.csv').set_index('input')
    p(f'bootstrap replicates: {int(B["B"]) if B is not None else 0} (game-clustered Poisson; map bands are 5-95%)')
    p('\n== inputs (95% bootstrap CI) ==')
    for k in ci.index:
        p(f'{k:14s} {ci.loc[k, "est"]:.3f} [{ci.loc[k, "lo"]:.3f}, {ci.loc[k, "hi"]:.3f}]')
    for name in ('BC', 'FC', 'LIVE', 'OREB'):
        sc_ = json.load(open(RES / 'params_fitted.json'))
        cap = np.minimum(cv[f'pshot_{name}_1'], 1 - sc_['p_sfoul'] - sc_['p_tov'])   # as used by the solver
        m3 = cap * cv[f'f3_{name}_1']
        m2 = cap * cv[f'f2_{name}_1']
        p(f'P(made 3 | {name:4s}) at 2/4/8/16 s: ' + ', '.join(
            f'{m3[i(t)]:.3f}' + (f' [{B[f"make3_{name}_lo"][i(t)]:.3f}, {B[f"make3_{name}_hi"][i(t)]:.3f}]'
                                 if B is not None else '') for t in (2, 4, 8, 16))
          + f';  P(made 2) at 4/8 s: {m2[i(4)]:.3f}, {m2[i(8)]:.3f}')
    p('\n== leader up 3: gain from fouling at once (pp), fitted [5-95%] vs hand-set ==')
    for s, lab in ((0, 'backcourt inbound, trailer no TO'), (1, 'frontcourt after TO')):
        x = F['foul'][:, 2, s, 0, 0]
        p(f'{lab}: ' + '; '.join(f't={t}: {x[i(t)]:+.1f}{band(B, "foul", (i(t), 2, s, 0, 0))}' for t in (3, 5, 8, 12, 18)))
        pos = grid[x > 0]
        p(f'   fouling optimal (point estimate) for t in {pos.min() if len(pos) else None}..{pos.max() if len(pos) else None} s;'
          f' sign changes at {crossings(np.nan_to_num(x), grid)}')
    p('hand-set: ' + '; '.join(f't={t}: {H["foul"][i(t), 2, 0, 0, 0]:+.1f}' for t in (3, 5, 8, 12, 18)))
    for l in (0, 1):
        p(f'up {l + 1}, backcourt: worst / best gain from fouling over 3-24 s: '
          f'{np.nanmin(F["foul"][i(3):, l, 0, 0, 0]):+.1f} / {np.nanmax(F["foul"][i(3):, l, 0, 0, 0]):+.1f}')
    p('with a foul to give (side-out), up 3 backcourt: ' + '; '.join(
        f't={t}: {F["foul"][i(t), 2, 0, 0, 1]:+.1f}' for t in (3, 5, 8, 12, 18)))
    p('\n== last FT: gain from missing on purpose (pp), opponent no TO ==')
    for ld in (-2, -3, -1, 1):
        j = FT_LEADS.index(ld)
        for a in (0, 1):
            x = F['ft'][:, j, a, 0]
            p(f'lead {ld:+d}, shooter TOs {a}: ' + '; '.join(
                f't={t}: {x[i(t)]:+.1f}{band(B, "ft", (i(t), j, a, 0))}' for t in (1, 2, 3, 5, 8, 12))
              + f';  sign changes at {crossings(x, grid)}')
        p(f'   hand-set lead {ld:+d}: ' + '; '.join(f't={t}: {H["ft"][i(t), j, 0, 0]:+.1f}' for t in (1, 2, 3, 5, 8, 12)))
    p('break-even OREB rate rho*, down 2 / down 3, shooter no TO: ' + '; '.join(
        f't={t}: {F["rstar"][i(t), 1, 0, 0]:.2f} / {F["rstar"][i(t), 0, 0, 0]:.2f}' for t in (3.5, 5, 8, 12)))
    p('\n== value of holding one timeout (pp), dead ball, backcourt ==')
    for l in range(4):
        x = F['adv'][:, l, 0]
        p(f'down {l}: max {x.max():.1f} at t={grid[x.argmax()]:.1f}; at 2/5/10/20 s: '
          + ', '.join(f'{x[i(t)]:.1f}' for t in (2, 5, 10, 20)))
    p('\n== observed behaviour ==')
    of = pd.read_csv(RES / 'obs_foul.csv')
    for s in (0, 1):
        for e in ('pre', 'post'):
            x = of[(of.era == e) & (of.s == s) & (of['lead'] == 3) & (of.t_lo >= 3) & (of.t_hi <= 12)]
            p(f'up 3, s={s}, {e}: deliberate-foul rate at 3-12 s = {np.average(x.rate, weights=x.n):.3f} (n={x.n.sum()})')
        for e in ('pre', 'post'):
            x = of[(of.era == e) & (of.s == s) & (of['lead'] <= 2)]
            p(f'up 1-2, s={s}, {e}: foul rate (background) = {np.average(x.rate, weights=x.n):.3f} (n={x.n.sum()})')
    ft = pd.read_csv(RES / 'obs_ft.csv')
    for ld in (-2, -3):
        for e in ('pre', 'post'):
            x = ft[(ft.era == e) & (ft.lead_b == ld) & (ft.t_hi <= 5)]
            p(f'down {-ld} last FT <= 5 s, {e}: miss {np.average(x.rate, weights=x.n):.3f} vs expected '
              f'{np.average(x.expected, weights=x.n):.3f} (n={x.n.sum()})')
    o = pd.read_csv(RES / 'obs_ft_outcomes.csv')
    p('make vs miss outcome (down 2-3, 0.5 < t <= 5 s), by era and shooter timeouts:')
    p(o.to_string(index=False))
    P = pd.read_parquet(RES.parent / 'data' / 'derived' / 'possessions.parquet',
                        columns=['season', 'period', 't0', 'lead0', 'start', 'tol_O'])
    Q = P[(P.period == 4) & (P.t0 <= 24) & P.lead0.isin([-1, -2, -3]) & (P.start != 'oreb')]
    z = (Q.tol_O == 0).groupby(np.where(Q.season >= 2017, 'post', 'pre')).mean()
    p(f'trailer (down 1-3, <= 24 s) with no timeout left: pre {z["pre"]:.3f}, post {z["post"]:.3f}')
    e3 = np.select([Q.season <= 2005, Q.season <= 2016], ['2000-05', '2006-16'], '2017-25')
    z3 = (Q.tol_O == 0).groupby(e3).agg(['mean', 'size'])
    p('  by era (paper uses 2006-16 vs 2017-25): ' + '; '.join(f'{k} {r["mean"]:.3f} (n={int(r["size"])})' for k, r in z3.iterrows()))
    from .estimate import curve_sample, load, oreb_sample
    PP, FF = load()
    C = curve_sample(PP)
    fm = FF[(FF.made == 0) & FF.oreb.notna() & (FF.t_reb > 0)]
    g = pd.read_parquet(RES.parent / 'data' / 'derived' / 'games.parquet')
    p(f'\nsample sizes: games {len(g)}, phases {len(PP)}, last FTs {len(FF)}, curve sample Q1-3 {int((C.press == 0).sum())} '
      f'/ pressure {int((C.press == 1).sum())}, deliberate-miss proxy misses {len(fm[fm.lead_b.isin([-2, -3]) & (fm.t <= 3) & fm.pct.notna()])}, '
      f'ordinary misses {len(fm[fm.t >= 8])}')
    p('\n== facts quoted in the paper (referee round 31 Oct 2026) ==')
    for name in ('BC', 'LIVE', 'FC', 'OREB'):
        sh = cv[f'share2_{name}_0']
        p(f'two-point attempts allowed after {name}: from {grid[np.argmax(sh >= 0.25)]:.1f} s' if sh.max() >= 0.25
          else f'two-point attempts never allowed after {name}')
    late = pd.read_parquet(RES.parent / 'data' / 'derived' / 'late.parquet', columns=['gameId', 'period', 'season', 'seq', 'to_l3_H', 'to_l3_A'])
    q4 = late[late.period == 4].sort_values('seq').groupby('gameId').last()
    sh3 = (pd.concat([q4.to_l3_H, q4.to_l3_A]) > 2).groupby(pd.concat([q4.season, q4.season]).values).mean()
    p(f'team-games with > 2 timeouts in the last 3:00 of Q4: {100 * sh3[sh3.index < 2017].min():.1f}-'
      f'{100 * sh3[sh3.index < 2017].max():.1f}% per season before 2017-18, {100 * sh3[sh3.index >= 2017].min():.1f}-'
      f'{100 * sh3[sh3.index >= 2017].max():.1f}% since')
    l2 = pd.read_parquet(RES.parent / 'data' / 'derived' / 'late.parquet',
                         columns=['gameId', 'period', 'season', 'seq', 'to_l2full_H', 'to_l2full_A'])
    q2 = l2[(l2.period == 4) & (l2.season < 2017)].sort_values('seq').groupby('gameId').last()
    x2 = pd.concat([q2.to_l2full_H, q2.to_l2full_A])
    p(f'before 2017-18: team-games with 3 full timeouts in the last 2:00 of Q4: {int((x2 >= 3).sum())} of {len(x2)}')

    def win(x, lo=0, hi=24):
        m = (grid >= lo - 1e-9) & (grid <= hi + 1e-9)
        return np.nanmin(x[m]), grid[m][np.nanargmin(x[m])], np.nanmax(x[m]), grid[m][np.nanargmax(x[m])]
    for l in range(3):
        for s_, g_, lab in ((0, 0, 'BC'), (1, 0, 'FC'), (0, 1, 'BC + foul to give')):
            mn, tmn, mx, tmx = win(F['foul'][:, l, s_, 0, g_])
            p(f'up {l + 1}, {lab}: foul gain min {mn:+.2f} at {tmn:.1f} s, max {mx:+.2f} at {tmx:.1f} s')
    fmn, _, fmx, _ = win(F['foul'][:, 2, 1, 0, 0], 3, 5)
    p(f'up 3 at 3-5 s: fitted FC {fmn:+.2f} to {fmx:+.2f}')
    if B is not None:
        from .figures import band_includes_zero
        m = (grid >= 3 - 1e-9) & (grid <= 13 + 1e-9)
        unc = band_includes_zero(B['foul_lo'][:, 2, 0, 0, 0], B['foul_hi'][:, 2, 0, 0, 0])
        p(f'up 3, BC: share of 3-13 s where the 90% band of the foul gain includes 0 (hatched): {unc[m].mean():.2f}; '
          f'hatched runs ' + ', '.join(f'{a:.1f}-{b:.1f} s' for a, b in runs_of(unc, grid)))
    for l in range(4):
        x = F['adv'][:, l, 0]
        p(f'value of a timeout, down {l}: max beyond 10 s {x[grid > 10].max():.2f}')
    mn, _, mx, _ = win(F['foul'][:, 2, 0, 0, 0], 3, 5)
    hmn, _, hmx, _ = win(H['foul'][:, 2, 0, 0, 0], 3, 5)
    p(f'up 3 at 3-5 s: fitted BC {mn:+.2f} to {mx:+.2f}; hand-set {hmn:+.2f} to {hmx:+.2f}')
    for ld in (-2, 1):
        j = FT_LEADS.index(ld)
        for lab, arr in (('fitted', F['ft'][:, j, 0, 0]), ('hand-set', H['ft'][:, j, 0, 0])):
            pos = arr > 1e-9
            runs, st = [], None
            for k, v in enumerate(pos):
                if v and st is None:
                    st = k
                if (not v or k == len(pos) - 1) and st is not None:
                    runs.append((grid[st], grid[k - 1 if not v else k]))
                    st = None
            p(f'last FT lead {ld:+d}, {lab}, no timeouts: miss strictly better on ' + ', '.join(f'{a:.1f}-{b:.1f} s' for a, b in runs))
    for a in (0, 1):
        x = F['ft'][:, FT_LEADS.index(-2), a, 0]
        drop = np.flatnonzero((grid > 1) & (x < 1.0))
        p(f'last FT down 2, shooter TOs {a}: gain >= 1 pp up to {grid[drop[0] - 1]:.1f} s; last positive at '
          f'{grid[np.flatnonzero(x > 0).max()]:.1f} s')
    rs = F['rstar'][:, FT_LEADS.index(-2), 0, 0]
    from .dp import HAND
    for lab, r in (('raw rho_dm', ci.loc['rho_dm', 'est']), ('rho_dm lower 95% bound', ci.loc['rho_dm', 'lo']),
                   ('hand-set 0.13', HAND['oreb_ft'])):
        ok = (grid > 1) & (r > rs)
        p(f'down 2, no TO: {lab} ({r:.3f}) exceeds rho* on ' + ', '.join(
            f'{a:.1f}-{b:.1f} s' for a, b in runs_of(ok, grid)))
    from .dp import N as NT, ND, NS_, NA
    p(f'state space: {NT + 1} times x {ND} leads x {NS_} start types x {NA}^2 timeouts x 2^2 fouls to give')
    sc_ = json.load(open(RES / 'params_fitted.json'))
    p(f'shot clock after an OREB: 14.0 s; after a live timeout that follows it: {14.0 - sc_["to_cost_live"]:.1f} s '
      f'(on the grid {140 - int(round(sc_["to_cost_live"] / DT))} steps)')

    p('\n== sensitivity (results/sensitivity.csv) ==')
    sv = pd.read_csv(RES / 'sensitivity.csv')
    p(f'variants (excluding the main estimate): {len(sv) - 1}')
    ordn = sv.variant.str.startswith('deliberate-miss OREB = ordinary')
    for col in ('foul_up3_FC_3', 'foul_up3_BC_3', 'ft_down2_2', 'ft_down2_5', 'timeout_value_down1_max'):
        p(f'{col}: all variants {sv[col].min():+.2f} to {sv[col].max():+.2f}; excluding rho_dm = ordinary '
          f'{sv[col][~ordn].min():+.2f} to {sv[col][~ordn].max():+.2f}')
    used = sv.variant.str.contains('time used')
    p(f'timeout_value_down1_max excluding the time-used curves: {sv.timeout_value_down1_max[~used].min():.2f} to '
      f'{sv.timeout_value_down1_max[~used].max():.2f}')
    sc_curves = np.load(RES / 'sens_curves.npz')
    lab = [str(x) for x in sc_curves['labels']]
    k = [j for j, l in enumerate(lab) if l.startswith('deliberate-miss OREB = ordinary')][0]
    f_ord = sc_curves['ft'][k]
    p(f'rho_dm = ordinary: FT gain down 2 at 2 s {f_ord[i(2)]:+.2f}; last positive at {grid[np.flatnonzero(f_ord > 0).max()]:.1f} s')
    for _, r in sv.iterrows():
        p(f'  {r.variant}: foul FC3 {r.foul_up3_FC_3:+.2f}, BC3 {r.foul_up3_BC_3:+.2f}, FT2 {r.ft_down2_2:+.2f}, '
          f'FT5 {r.ft_down2_5:+.2f}, >=1pp until {r.ft_down2_gain_ge1_until:.1f}, TO {r.timeout_value_down1_max:.2f}')
    sens_table(sv)
    ch = pd.read_csv(RES / 'calib_holdout.csv')
    p('held-out calibration (Brier model / table): ' + '; '.join(
        f'train {r.train}: {r.brier_model:.3f} / {r.brier_table:.3f}' for r in ch.itertuples()))

    p('\n== identification check (results/ident_check.csv): FG% unforced (>= 4 s left) minus forced (<= 1 s left) ==')
    ic = pd.read_csv(RES / 'ident_check.csv')
    for k_ in (2, 3):
        x = ic[(ic.k == k_) & ic.diff_lo.notna()]
        p(f'{k_}-point: {len(x)} cells, difference {100 * x["diff"].min():+.0f} to {100 * x["diff"].max():+.0f} pp '
          f'(median {100 * x["diff"].median():+.0f}); cells with 90% interval above 0: {(x.diff_lo > 0).sum()}')
    p(f'forced shots {int(ic.n_forced.sum())}, unforced {int(ic.n_unforced.sum())}')
    ident_table(ic)

    p('\n== deliberate-miss proxy audit (results/dm_linkage.csv, dm_timing.csv, dm_transport.csv) ==')
    lk = pd.read_csv(RES / 'dm_linkage.csv')
    for r in lk.itertuples():
        p(f'  {r.category}: {r.n_all} (cohort {r.n_cohort}; linked rebound is the first event {r.linked_rebound_is_first_event};'
          f' after the buzzer {r.n_after_buzzer})')
    ra = pd.read_csv(RES / 'reb_state_audit.csv')
    for r in ra.itertuples():
        p(f'  rebound starts in the unfiltered curve sample, {r.start} after a missed {r.origin}: {r.n_curve_sample}, '
          f'team rebounds {r.n_team_rebound} ({100 * r.share_team:.1f}%; excluded from the curves)')
    tm = pd.read_csv(RES / 'dm_timing.csv')
    for r in tm.itertuples():
        p(f'  FT to rebound, {r.sample}: median {r.median:.1f} s, mean {r.mean:.2f}, <= 1 s {100 * r.share_le_1s:.0f}%, '
          f'> 2 s {100 * r.share_gt_2s:.0f}%')
    tr = pd.read_csv(RES / 'dm_transport.csv')
    for r in tr.itertuples():
        p(f'  {r.group} | {r.definition}: {r.rate:.3f} [{r.lo:.3f}, {r.hi:.3f}] (n={r.n})')
    cs = pd.read_csv(RES / 'calib_summary.csv', index_col=0).iloc[:, 0]
    p(f'\ncalibration: n={int(cs["n"])}, Brier {cs["brier"]:.4f} vs constant {cs["brier_const"]:.4f}; '
      f'score-and-time baseline {cs["brier_score_time"]:.4f}; mean predicted {cs["mean_pred"]:.3f} vs observed {cs["mean_obs"]:.3f}')
    chk = json.load(open(RES.parent / 'checks' / 'dp_check_out.json'))
    p(f'Mathematica ({chk["wolframVersion"]}): exact re-solve max |diff| V {chk["maxAbsDiffV"]:.1e} over '
      f'{chk["smallInstanceStates"]} states, FT {chk["maxAbsDiffFT"]:.1e}; break-even identity {chk["breakEvenIdentity"]}, '
      f'sign agreement {chk["signAgreementAll"]} on {chk["fittedRows"]} fitted states')
    (RES / 'headline.txt').write_text('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
