"""All paper figures, from results/ (and data/derived for the descriptive panels).

usage: python -m nbaeg.figures       (writes figures/fig*.pdf and .png)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D

from .build import DER, ROOT
from .dp import HAND
from .estimate import GRID, RES, TYPES, curve_sample, load
from .maps import FT_LEADS

FIG = ROOT / 'figures'
INK, INK2, MUTED, GRIDC, AXC = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
C_PRE, C_POST = '#2a78d6', '#eb6834'          # categorical slots 1-2 (validated palette)
C_TYPE = {'BC': '#2a78d6', 'FC': '#eb6834', 'LIVE': '#1baf7a', 'OREB': '#eda100'}
MK_ERA = {'pre': 'o', 'post': 's'}
MK_TYPE = {'BC': 'o', 'FC': 's', 'LIVE': '^', 'OREB': 'D'}   # start type by marker shape (Figure 2)   # era also by marker shape (greyscale)
LS_TYPE = {'BC': '-', 'FC': '--', 'LIVE': '-.', 'OREB': ':'}   # redundant with colour (greyscale, CVD)
LAB_TYPE = {'BC': 'backcourt inbound', 'FC': 'frontcourt inbound after timeout', 'LIVE': 'defensive rebound / steal',
            'OREB': 'offensive rebound'}
DIV = LinearSegmentedColormap.from_list('div', ['#8c1c1c', '#e34948', '#f2b1ad', '#f0efec', '#9ec5f4', '#2a78d6',
                                                '#104281'])
W = 6.3                                         # text width, inches

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8, 'legend.fontsize': 7,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.edgecolor': AXC, 'axes.labelcolor': INK,
    'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK, 'axes.grid': True, 'grid.color': GRIDC,
    'grid.linewidth': 0.5, 'axes.spines.top': False, 'axes.spines.right': False, 'lines.linewidth': 1.4,
    'legend.frameon': False, 'savefig.dpi': 300, 'savefig.bbox': 'tight', 'pdf.fonttype': 42,
    'hatch.color': '#52514e', 'hatch.linewidth': 0.4})


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f'{name}.pdf')
    fig.savefig(FIG / f'{name}.png')
    plt.close(fig)
    print('wrote', name)


def ld(name):
    z = np.load(RES / name)
    return {k: z[k] for k in z.files}


# ----------------------------------------------------------------------------- F1 rule check
def fig_timeouts(P):
    L = pd.read_parquet(DER / 'late.parquet', columns=['gameId', 'period', 'season', 'to_l3_H', 'to_l3_A', 'seq'])
    q4 = L[L.period == 4].sort_values('seq').groupby('gameId').last()
    x = pd.concat([q4.to_l3_H, q4.to_l3_A]) > 2
    s = pd.concat([q4.season, q4.season])
    share = x.groupby(s.values).mean()
    Q = P[(P.period == 4) & (P.t0 <= 24) & P.lead0.isin([-1, -2, -3]) & (P.start != 'oreb')]
    z = Q.groupby('season').tol_O.apply(lambda v: (v == 0).mean())
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.1))
    for a_, y, lab in ((ax[0], share, '(a) > 2 timeouts in the last 3:00 of Q4'),
                       (ax[1], z, '(b) phases of a trailer (down 1-3, ≤ 24 s) with no timeout')):
        yrs = y.index.values
        a_.axvspan(2016.5, yrs.max() + 0.5, color=C_POST, alpha=0.08, lw=0)
        a_.plot(yrs, 100 * y.values, color=INK2, marker='o', ms=3)
        a_.set_ylabel('%')
        a_.set_title(lab, loc='left', fontsize=7.5)
        a_.set_xticks([2000, 2005, 2010, 2017, 2021, 2025])
        a_.set_xticklabels(['00-01', '05-06', '10-11', '17-18', '21-22', '25-26'])
        a_.set_ylim(bottom=0)
    ax[0].text(2017.3, 100 * share.max() * 0.85, 'cap on late\ntimeouts', color=INK2, fontsize=7)
    save(fig, 'fig1_timeouts')


# ----------------------------------------------------------------------------- F2 rushed-shot curves
def fig_curves(P, B):
    C = curve_sample(P)
    C = C[C.press == 1]
    cv = ld('curves_fitted.npz')
    import json
    sc = json.load(open(RES / 'params_fitted.json'))
    cap = 1 - sc['p_sfoul'] - sc['p_tov']
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.6), sharey=True)
    bins = [0, 1, 2, 3, 4, 6, 8, 12, 16, 24]
    for j, k in enumerate((3, 2)):
        a_ = ax[j]
        for name in TYPES:
            col = C_TYPE[name]
            y = np.minimum(cv[f'pshot_{name}_1'], cap) * cv[f'f{k}_{name}_1']   # as used by the solver
            if k == 2:                                   # where the solver does not allow a two: thin and faint
                ok = cv[f'share2_{name}_0'] >= sc.get('share2_min', 0.25)
                a_.plot(GRID, np.where(ok, np.nan, y), color=col, ls=LS_TYPE[name], lw=0.6, alpha=0.45)
                y = np.where(ok, y, np.nan)
            a_.plot(GRID, y, color=col, ls=LS_TYPE[name], label=LAB_TYPE[name], marker=MK_TYPE[name], markevery=[-1],
                    ms=3.5)
            if B is not None:
                a_.fill_between(GRID, B[f'make{k}_{name}_lo'], B[f'make{k}_{name}_hi'], color=col, alpha=0.18, lw=0)
            d = C[C.s == TYPES.index(name)]
            # empirical P(made k): the shot-attempt share times the FG% of k-point attempts, per bin
            g = d.groupby(pd.cut(d.a, bins), observed=True)
            emp = []
            for _, x in g:
                sh = x[x.shot == 1]
                sk = sh[(sh.fg3 == True) == (k == 3)]
                if len(sk) >= 25:
                    emp.append((x.a.median(), x.shot.mean() * sk.made.astype(float).mean()))
            if emp:
                e = np.array(emp)
                a_.scatter(e[:, 0], e[:, 1], s=12, color=col, edgecolor='white', linewidth=0.6, zorder=3,
                           marker=MK_TYPE[name])          # start type also by marker shape (greyscale)
        eff = np.interp(GRID, [0, 1, 2, 3, 4, 5], [0, .15, .45, .75, .9, 1])
        hand = HAND[f'p{k}'] * (1 - HAND['tov'] - HAND['sfoul']) * eff
        if k == 2:
            hand = np.where(GRID > 2, hand, np.nan)
        a_.plot(GRID, hand, color=INK, ls='--', lw=1, label='hand-set prototype')
        a_.set_title(f'({"ab"[j]}) P(made {k}): tied or down 1-3, Q4/OT', loc='left')
        a_.set_xlabel('seconds available at the start of the play')
        a_.set_xlim(0, 24)
    ax[0].set_ylabel('probability')
    ax[0].set_ylim(0, None)
    fig.legend(*ax[0].get_legend_handles_labels(), loc='lower center', bbox_to_anchor=(0.5, 0.98), ncol=5, fontsize=6.5)
    save(fig, 'fig2_rushed_curves')


# ----------------------------------------------------------------------------- F3 inputs
def fig_inputs():
    t = pd.read_csv(RES / 'inputs_ci.csv').set_index('input')
    hand = dict(rho_fg=HAND['oreb'], rho_ft=HAND['oreb_ft'], rho_dm=HAND['oreb_ft'], p_sf3=HAND['p_sf_int'],
                p_sfoul=HAND['sfoul'], p_tov=HAND['tov'], q=HAND['ft'], foul_el_BC=HAND['foul_el'],
                foul_el_FC=HAND['foul_el'], foul_el_LIVE=HAND['foul_el'], reb_time=1.0, reb_time_ft=1.0,
                reb_time_ft_ord=1.0, th_dm=0.0, eta_dm=0.0)
    probs = [('rho_dm', 'OREB, deliberate FT miss'), ('rho_ft', 'OREB, ordinary FT miss'),
             ('th_dm', 'team rebound, kept deliberate miss'), ('eta_dm', 'team rebound, lost deliberate miss'),
             ('rho_fg', 'OREB, missed FG (late)'), ('p_sf3', 'intentional foul called a 3-FT foul'),
             ('p_sfoul', 'shooting foul per play'), ('p_tov', 'turnover per play'), ('q', 'FT% of fouled players')]
    secs = [('foul_el_BC', 'time to foul: backcourt inbound'), ('foul_el_FC', 'time to foul: after timeout'),
            ('foul_el_LIVE', 'time to foul: live ball'), ('to_cost_live', 'time to call timeout: live ball'),
            ('reb_time', 'rebound time: missed FG'), ('reb_time_ft', 'rebound time: deliberate FT miss'),
            ('reb_time_ft_ord', 'rebound time: ordinary FT miss')]
    fig, ax = plt.subplots(1, 2, figsize=(W, 3.0), gridspec_kw=dict(wspace=1.35))
    for a_, rows, xl in ((ax[0], probs, 'probability'), (ax[1], secs, 'seconds')):
        for i, (k, lab) in enumerate(rows):
            y = len(rows) - 1 - i
            a_.plot([t.loc[k, 'lo'], t.loc[k, 'hi']], [y, y], color=C_PRE, lw=2.2, solid_capstyle='round')
            a_.plot(t.loc[k, 'est'], y, 'o', color=C_PRE, ms=4.5, mec='white', mew=0.8)
            if k in hand:
                a_.plot(hand[k], y, 'D', color=INK, ms=3.5, mfc='white', mew=1)
        a_.set_yticks(range(len(rows)))
        a_.set_yticklabels([lab for _, lab in rows][::-1])
        a_.set_xlabel(xl)
        a_.grid(axis='y', visible=False)
    fig.legend(handles=[Line2D([], [], color=C_PRE, marker='o', lw=2, label='estimate, 95% bootstrap CI'),
                        Line2D([], [], color=INK, marker='D', mfc='white', lw=0, label='hand-set prototype')],
               loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=6.5)
    save(fig, 'fig3_inputs')


# ----------------------------------------------------------------------------- map helper
def band_includes_zero(lo, hi):
    """Hatching rule: the pointwise 90% bootstrap band [5th, 95th percentile] includes zero and is not exactly [0, 0]
    (a band of exactly zero is a tie: both actions equivalent in at least 90% of replicates)."""
    with np.errstate(invalid='ignore'):
        return (lo <= 0) & (hi >= 0) & ~((lo == 0) & (hi == 0))


def sign_labels(a_, z, r, tg, words, min_len=1.5, every=6.0):
    """Name the preferred action inside the strip (greyscale-safe): one label per run of cells with the same
    sign (|margin| >= 0.25 pp) that is at least min_len seconds long, repeated every `every` seconds; shorter runs
    of at least 0.3 s get a '+' (the action named first) or a minus sign."""
    sg = np.where(np.isnan(z) | (np.abs(z) < 0.25), 0, np.sign(z))
    k = 0
    while k < len(sg):
        if sg[k] == 0:
            k += 1
            continue
        e = k
        while e + 1 < len(sg) and sg[e + 1] == sg[k]:
            e += 1
        t0, t1 = tg[k], tg[e]
        if 0.3 <= t1 - t0 < min_len:
            a_.text(0.5 * (t0 + t1), r, '+' if sg[k] > 0 else '\u2212', ha='center', va='center', fontsize=6,
                    color=INK, zorder=6, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.05', fc='white', ec='none', alpha=0.85))
        if t1 - t0 >= min_len:
            m = max(1, int((t1 - t0) // every))
            for x in t0 + (t1 - t0) * (np.arange(m) + 0.5) / m:
                a_.text(x, r, words[0] if sg[k] > 0 else words[1], ha='center', va='center', fontsize=6.5,
                        color=INK, zorder=6, fontstyle='italic',
                        bbox=dict(boxstyle='round,pad=0.12', fc='white', ec='none', alpha=0.85))
        k = e + 1


def strip_map(a_, M, band, rows, labels, tmax, vmax, words=None):
    """Rows x time heat strip of a WP margin (pp); hatched where the pointwise 90% bootstrap band (lo, hi) includes
    zero (band_includes_zero); `words` = (positive, negative) action names written inside the strips."""
    i = GRID <= tmax + 1e-9
    tg = GRID[i]
    edges_t = np.r_[tg - 0.05, tg[-1] + 0.05]
    edges_r = np.arange(len(rows) + 1) - 0.5
    Z = M[i][:, rows].T
    a_.set_facecolor('#d9d8d2')                         # cells where the action is unavailable (NaN) stay grey
    pc = a_.pcolormesh(edges_t, edges_r, np.ma.masked_invalid(np.clip(Z, -vmax, vmax)), cmap=DIV,
                       norm=TwoSlopeNorm(0, -vmax, vmax), shading='flat', rasterized=True)
    for r in range(len(rows)):                          # decision boundary, drawn in black: a change of sign
        z = Z[r]                                        # between margins above 0.25 pp; near-zero cells are
        last, k_last = 0, None                          # treated as indifferent and do not create boundaries
        for k, v in enumerate(z):
            if np.isnan(v) or abs(v) < 0.25:
                continue
            sg = 1 if v > 0 else -1
            if last and sg != last:
                x = 0.5 * (edges_t[k_last + 1] + edges_t[k])
                a_.plot([x, x], [r - 0.5, r + 0.5], color=INK, lw=1.1, solid_capstyle='butt', zorder=3)
            last, k_last = sg, k
    if band is not None:
        unc = band_includes_zero(band[0][i][:, rows].T, band[1][i][:, rows].T).astype(float)
        for r in range(len(rows)):                       # hatch each uncertain run of cells
            on = np.r_[0, unc[r], 0]
            st, en = np.flatnonzero(np.diff(on) == 1), np.flatnonzero(np.diff(on) == -1)
            for a0, a1 in zip(st, en):
                a_.fill_between([edges_t[a0], edges_t[a1]], r - 0.5, r + 0.5, facecolor='none', hatch='////',
                                edgecolor=INK2, lw=0)
    if words is not None:
        for r in range(len(rows)):
            sign_labels(a_, Z[r], r, tg, words)
    for r in range(len(rows) + 1):
        a_.axhline(r - 0.5, color='white', lw=2)
    a_.set_yticks(range(len(rows)))
    a_.set_yticklabels(labels)
    a_.set_xlim(0, tmax)
    a_.set_ylim(-0.5, len(rows) - 0.5)
    a_.grid(False)
    return pc


def size_of(r):
    """Marker area proportional to the rate (no offset)."""
    return 150 * np.asarray(r)


# ----------------------------------------------------------------------------- F4 leader's foul map
def fig_foul_map(B):
    M = ld('maps_fitted.npz')['foul']
    obs = pd.read_csv(RES / 'obs_foul.csv')
    panels = [((0, 0, 0), 0, '(a) trailer inbounds in the backcourt, no timeout left'),
              ((1, 0, 0), 1, '(b) trailer inbounds in the frontcourt after its timeout'),
              ((0, 0, 1), None, '(c) as (a), but the leader has a foul to give')]
    fig, ax = plt.subplots(3, 1, figsize=(W, 4.8), sharex=True, gridspec_kw=dict(hspace=0.45))
    for a_, ((s, a, g), s_obs, title) in zip(ax, panels):
        pp = None if B is None else (B['foul_lo'][:, :, s, a, g], B['foul_hi'][:, :, s, a, g])
        pc = strip_map(a_, M[:, :, s, a, g], pp, [0, 1, 2], ['up 1', 'up 2', 'up 3'], 24, 8, ('foul', 'defend'))
        a_.set_title(title, loc='left')
        if s_obs is not None:
            for e, col, dy in (('pre', C_PRE, 0.25), ('post', C_POST, -0.25)):
                o = obs[(obs.era == e) & (obs.s == s_obs) & (obs.n >= 10)]
                z0 = o.rate == 0                           # observed, but nobody fouled: a small cross
                a_.scatter(o.t_mid[~z0], o['lead'][~z0] - 1 + dy, s=size_of(o.rate[~z0]), color=col, edgecolor='white',
                           linewidth=0.5, zorder=4, marker=MK_ERA[e])
                a_.scatter(o.t_mid[z0], o['lead'][z0] - 1 + dy, s=9, color=col, linewidth=0.8, zorder=4, marker='x')
    ax[-1].set_xlabel('seconds left when the trailer gets the ball (or calls its timeout)')
    cb = fig.colorbar(pc, ax=ax, shrink=0.6, pad=0.02, extend='both')
    cb.set_label("leader's gain from fouling at once (pp)")
    cb.ax.text(0.5, 1.07, 'foul\nbetter', transform=cb.ax.transAxes, ha='center', va='bottom', fontsize=6)
    cb.ax.text(0.5, -0.07, 'defend\nbetter', transform=cb.ax.transAxes, ha='center', va='top', fontsize=6)
    cb.outline.set_visible(False)
    hs = [Line2D([], [], marker=MK_ERA[e], lw=0, color=c, ms=np.sqrt(size_of(r)), mec='white',
                 label=f'{lab}, {int(100 * r)}% fouled') for e, c, lab, r in
          (('pre', C_PRE, '2006-17', 0.1), ('pre', C_PRE, '2006-17', 0.4), ('post', C_POST, '2017-26', 0.1),
           ('post', C_POST, '2017-26', 0.4))]
    hs.append(Line2D([], [], marker='x', lw=0, color=INK2, ms=3, label='observed, nobody fouled'))
    hs.append(plt.Rectangle((0, 0), 1, 1, facecolor='none', hatch='////', edgecolor=INK2, lw=0,
                            label='90% band includes 0'))
    hs.append(plt.Rectangle((0, 0), 1, 1, facecolor='#d9d8d2', lw=0, label='foul not possible in time'))
    hs.append(Line2D([], [], color=INK, lw=1.1, label='decision boundary'))
    ax[0].legend(handles=hs, ncol=4, loc='lower left', bbox_to_anchor=(0, 1.15), fontsize=6, handletextpad=0.3,
                 columnspacing=0.8)
    save(fig, 'fig4_foul_map')


# ----------------------------------------------------------------------------- F5 last-FT map
def fig_ft_map(B):
    M = ld('maps_fitted.npz')['ft']
    obs = pd.read_csv(RES / 'obs_ft.csv')
    # *: leading shooters use the deliberate-miss rebound rate estimated from trailing shooters (extrapolated)
    labels = [f'{"down" if l < 0 else "up"} {abs(l)}{"*" if l > 0 else ""}' for l in FT_LEADS]
    fig, ax = plt.subplots(2, 1, figsize=(W, 4.0), sharex=True, gridspec_kw=dict(hspace=0.4))
    for a_, a in zip(ax, (0, 1)):
        pp = None if B is None else (B['ft_lo'][:, :, a, 0], B['ft_hi'][:, :, a, 0])
        pc = strip_map(a_, M[:, :, a, 0], pp, list(range(len(FT_LEADS))), labels, 12, 6, ('miss', 'make'))
        a_.set_title(f'({"ab"[a]}) {"no timeout" if a == 0 else "one timeout"}', loc='left')
        for e, col, dy in (('pre', C_PRE, 0.25), ('post', C_POST, -0.25)):
            o = obs[(obs.era == e) & (obs.tol == a) & (obs.n >= 8) & (obs.t_lo < 12)]
            ex = (o.rate - o.expected).values
            y = np.array([FT_LEADS.index(l) + dy for l in o.lead_b])
            pos = ex >= 0                                  # filled: more misses than expected; open: fewer
            a_.scatter(o.t_mid[pos], y[pos], s=size_of(ex[pos]), color=col, edgecolor='white', linewidth=0.5, zorder=4,
                       marker=MK_ERA[e])
            a_.scatter(o.t_mid[~pos], y[~pos], s=size_of(-ex[~pos]), facecolor='none', edgecolor=col, linewidth=0.9,
                       zorder=4, marker=MK_ERA[e])
    ax[-1].set_xlabel('seconds left at the last free throw (shooter\'s lead before it on the left)')
    cb = fig.colorbar(pc, ax=ax, shrink=0.7, pad=0.02, extend='both')
    cb.set_label('gain from missing on purpose (pp)')
    cb.ax.text(0.5, 1.07, 'miss\nbetter', transform=cb.ax.transAxes, ha='center', va='bottom', fontsize=6)
    cb.ax.text(0.5, -0.07, 'make\nbetter', transform=cb.ax.transAxes, ha='center', va='top', fontsize=6)
    cb.outline.set_visible(False)
    hs = [Line2D([], [], marker=MK_ERA[e], lw=0, color=c, ms=np.sqrt(size_of(r)), mec='white',
                 label=f'{lab}, excess miss {int(100 * r)} pp') for e, c, lab, r in
          (('pre', C_PRE, '2006-17', 0.2), ('pre', C_PRE, '2006-17', 0.6), ('post', C_POST, '2017-26', 0.2),
           ('post', C_POST, '2017-26', 0.6))]
    hs.append(Line2D([], [], marker='o', lw=0, mfc='none', mec=INK2, ms=np.sqrt(size_of(0.2)),
                     label='open: fewer misses than expected'))
    ax[0].legend(handles=hs, ncol=3, loc='lower left', bbox_to_anchor=(0, 1.12), fontsize=6, handletextpad=0.3)
    for a_ in ax:              # raw rate estimated at <= 3 s; the corrected rate is checked up to 8 s
        a_.axvline(3, color=INK2, ls=(0, (1, 2)), lw=0.8, zorder=5)
        a_.axvline(8, color=INK2, ls=':', lw=1, zorder=5)
    ax[0].text(3.1, len(FT_LEADS) - 0.45, 'rebound rate estimated', fontsize=6, color=INK2, va='bottom')
    ax[0].text(8.1, len(FT_LEADS) - 0.45, 'corrected rate checked', fontsize=6, color=INK2, va='bottom')
    save(fig, 'fig5_ft_map')


# ----------------------------------------------------------------------------- F6 break-even rebound rate
def fig_breakeven(B):
    R = ld('maps_fitted.npz')['rstar']
    Rh = ld('maps_hand.npz')['rstar']
    t = pd.read_csv(RES / 'inputs_ci.csv').set_index('input')
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.5), sharey=True)
    for a_, ldv in zip(ax, (-2, -3)):
        j = FT_LEADS.index(ldv)
        a_.axvspan(8, 12, color=MUTED, alpha=0.12, lw=0)    # deliberate-miss rate extrapolated beyond 8 s
        a_.text(8.15, 0.95, 'rebound rate\nextrapolated', fontsize=6, color=INK2, va='top')
        a_.axhspan(t.loc['rho_dm', 'lo'], t.loc['rho_dm', 'hi'], color=MUTED, alpha=0.25, lw=0)
        a_.axhline(t.loc['rho_dm', 'est'], color=INK2, lw=1)
        a_.text(0.2, t.loc['rho_dm', 'hi'] + 0.01, 'estimated deliberate-miss OREB', ha='left', color=INK2, fontsize=6.5)
        a_.axhline(HAND['oreb_ft'], color=INK2, lw=1, ls=':')
        a_.text(11.9, HAND['oreb_ft'] - 0.015, 'hand-set OREB', ha='right', va='top', color=INK2, fontsize=6.5)
        for a, col in ((0, C_PRE), (1, C_POST)):
            a_.plot(GRID, R[:, j, a, 0], color=col, ls=('-', '-.')[a], label=f'fitted, shooter {"without" if a == 0 else "with"} a timeout')
            if B is not None:
                a_.fill_between(GRID, B['rstar_lo'][:, j, a, 0], B['rstar_hi'][:, j, a, 0], color=col, alpha=0.18, lw=0)
        a_.plot(GRID, Rh[:, j, 0, 0], color=INK, lw=1, ls='--', label='hand-set model')
        a_.set_xlim(0, 12)
        a_.set_ylim(0, 1)                                 # rho* outside [0, 1] (or undefined) is clipped
        for a in (0, 1):
            x = R[:, j, a, 0]
            out = (GRID <= 12) & (~np.isfinite(x) | (x > 1) | (x < 0)) & (GRID > 1)
            a_.plot(GRID[out], np.where(np.nan_to_num(x[out], nan=2.0) > 1, 0.99, 0.01), ls='none', marker='|',
                    color=(C_PRE, C_POST)[a], ms=3, mew=0.6)
        a_.set_title(f'down {-ldv} before the last FT', loc='left')
        a_.set_xlabel('seconds left at the last free throw')
    ax[0].set_ylabel('break-even OREB rate ρ*\n(miss on purpose if the rate exceeds it)')
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', bbox_to_anchor=(0.5, 0.98), ncol=3, fontsize=6.5)
    save(fig, 'fig6_breakeven')


# ----------------------------------------------------------------------------- F7 hand-set vs fitted
def fig_gap(B):
    F_, H_ = ld('maps_fitted.npz'), ld('maps_hand.npz')
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.5))
    a_ = ax[0]
    for s, col, lab in ((0, C_PRE, 'fitted, backcourt inbound'), (1, C_POST, 'fitted, after a timeout')):
        a_.plot(GRID, F_['foul'][:, 2, s, 0, 0], color=col, ls=('-', '-.')[s], label=lab)
        if B is not None:
            a_.fill_between(GRID, B['foul_lo'][:, 2, s, 0, 0], B['foul_hi'][:, 2, s, 0, 0], color=col, alpha=0.18, lw=0)
    a_.plot(GRID, H_['foul'][:, 2, 0, 0, 0], color=INK, ls='--', lw=1, label='hand-set')
    a_.axhline(0, color=AXC, lw=0.8)
    a_.set_title('(a) up 3: leader\'s gain from fouling (pp)', loc='left')
    a_.set_xlabel('seconds left')
    a_.set_xlim(0, 24)
    a_.legend(fontsize=6.5)
    a_ = ax[1]
    j = FT_LEADS.index(-2)
    for a, col, lab in ((0, C_PRE, 'fitted, no timeout'), (1, C_POST, 'fitted, timeout left')):
        a_.plot(GRID, F_['ft'][:, j, a, 0], color=col, ls=('-', '-.')[a], label=lab)
        if B is not None:
            a_.fill_between(GRID, B['ft_lo'][:, j, a, 0], B['ft_hi'][:, j, a, 0], color=col, alpha=0.18, lw=0)
    a_.plot(GRID, H_['ft'][:, j, 0, 0], color=INK, ls='--', lw=1, label='hand-set')
    a_.axhline(0, color=AXC, lw=0.8)
    a_.set_xlim(0, 12)
    a_.set_title('(b) down 2, last FT: gain from missing (pp)', loc='left')
    a_.set_xlabel('seconds left')
    a_.legend(fontsize=6.5)
    save(fig, 'fig7_hand_vs_fitted')


# ----------------------------------------------------------------------------- F8 calibration
def fig_calib():
    c = pd.read_csv(RES / 'calib.csv')
    s = pd.read_csv(RES / 'calib_summary.csv', index_col=0).iloc[:, 0]
    fig, a_ = plt.subplots(figsize=(3.1, 3.0))
    a_.plot([0, 1], [0, 1], color=AXC, lw=1)
    for e, col, lab in (('pre', C_PRE, '2006-17'), ('post', C_POST, '2017-26')):
        d = c[c.era == e]
        a_.scatter(d.pred, d.obs, s=d.n / 30, color=col, edgecolor='white', linewidth=0.6, label=lab, zorder=3,
                   marker=MK_ERA[e])
    a_.set_xlabel('model win probability (team with the ball)')
    a_.set_ylabel('observed (overtime counted as 1/2)')
    a_.set_title(f'Q4, ≤ 24 s, lead −3..3, n = {int(s["n"]):,}\nBrier {s["brier"]:.3f} (constant {s["brier_const"]:.3f})',
                 loc='left')
    a_.legend(loc='upper left')
    save(fig, 'fig8_calibration')


# ----------------------------------------------------------------------------- F9 value of the advance
def fig_advance(B):
    A = ld('maps_fitted.npz')['adv']
    fig, a_ = plt.subplots(figsize=(W * 0.6, 2.4))
    cols = ['#1baf7a', '#2a78d6', '#eb6834', '#eda100']
    for l in range(4):
        a_.plot(GRID, A[:, l, 0], color=cols[l], ls=('-', '--', '-.', ':')[l], label='tied' if l == 0 else f'down {l}')
        if B is not None:
            a_.fill_between(GRID, B['adv_lo'][:, l, 0], B['adv_hi'][:, l, 0], color=cols[l], alpha=0.18, lw=0)
    a_.set_xlabel('seconds left')
    a_.set_ylabel('win-probability gain (pp)')
    a_.set_title('value of holding a timeout (dead ball, backcourt)', loc='left')
    a_.legend(fontsize=6.5)
    a_.set_xlim(0, 24)
    save(fig, 'fig9_advance')


# ----------------------------------------------------------------------------- F10 sensitivity
def fig_sensitivity():
    rb = pd.read_csv(RES / 'rho_dm_bins.csv')
    rb = rb[rb.window.isin(['(0,1]', '(1,2]', '(2,3]', '(3,5]', '(5,8]'])]
    mid = np.array([0.5, 1.5, 2.5, 4.0, 6.5])
    Z = np.load(RES / 'sens_curves.npz')
    labels = list(Z['labels'])
    H = ld('maps_hand.npz')
    fig, ax = plt.subplots(1, 3, figsize=(W, 2.9), gridspec_kw=dict(width_ratios=[1, 1.15, 1.15], wspace=0.38))
    a_ = ax[0]
    a_.errorbar(mid - 0.12, rb.raw, yerr=[rb.raw - rb.raw_lo, rb.raw_hi - rb.raw], fmt='o', ms=3.5, color=INK2,
                mfc='white', lw=1, capsize=2, label='raw (lower bound under the mixture assumptions)')
    a_.errorbar(mid + 0.12, rb.mix, yerr=[rb.mix - rb.mix_lo, rb.mix_hi - rb.mix], fmt='o', ms=3.5, color=C_PRE,
                lw=1, capsize=2, label='corrected for accidental misses')
    est = pd.read_csv(RES / 'inputs_ci.csv').set_index('input')
    a_.axhline(est.loc['rho_ft', 'est'], color=INK2, ls=':', lw=1)
    a_.text(7.9, est.loc['rho_ft', 'est'] + 0.02, 'ordinary miss', ha='right', fontsize=6, color=INK2)
    a_.set_xlim(0, 8)
    a_.set_ylim(0, 1)
    a_.set_xlabel('seconds left at the free throw')
    a_.set_title('(a) deliberate-miss OREB', loc='left', fontsize=7.5)
    a_.legend(fontsize=5.5, loc='upper center', bbox_to_anchor=(0.5, -0.28), frameon=False)
    for j, (key, title, tmax) in enumerate((('foul', '(b) up 3, after timeout: foul', 24),
                                            ('ft', '(c) down 2, last FT: miss', 12))):
        a_ = ax[j + 1]
        nb = 0
        for k, lab in enumerate(labels):
            if lab.startswith('deliberate-miss OREB = ordinary'):
                a_.plot(GRID, Z[key][k], color=INK2, lw=0.9, ls=':', label='deliberate miss rebounded\nlike an ordinary one')
            elif 'beats a committed foul' in lab and key == 'foul':
                nb += 1
                a_.plot(GRID, Z[key][k], color=INK2, lw=0.9, ls='-.',
                        label='offence beats the foul\n($\\beta$ = 0.1, 0.2, 1)' if nb == 1 else None)
            elif lab != 'main estimate':
                a_.plot(GRID, Z[key][k], color=MUTED, lw=0.7, alpha=0.8)
        a_.plot(GRID, Z[key][labels.index('main estimate')], color=INK, lw=1.4, label='main estimate')
        hand = H['foul'][:, 2, 1, 0, 0] if key == 'foul' else H['ft'][:, FT_LEADS.index(-2), 0, 0]
        a_.plot(GRID, hand, color=INK, ls='--', lw=1, label='hand-set')
        a_.plot([], [], color=MUTED, lw=0.7, label=f'{len(labels) - 2 - nb} other variants')
        a_.axhline(0, color=AXC, lw=0.8)
        a_.set_xlim(0, tmax)
        a_.set_xlabel('seconds left')
        a_.set_title(title, loc='left', fontsize=7.5)
        if j == 0:
            a_.legend(fontsize=5.5, loc='upper center', bbox_to_anchor=(1.1, -0.28), ncol=3, frameon=False)
    ax[1].set_ylabel('gain (pp)')
    save(fig, 'fig10_sensitivity')


def main():
    P, _ = load()
    B = ld('boot_summary.npz') if (RES / 'boot_summary.npz').exists() else None
    fig_timeouts(P)
    fig_curves(P, B)
    fig_inputs()
    fig_foul_map(B)
    fig_ft_map(B)
    fig_breakeven(B)
    fig_gap(B)
    fig_calib()
    fig_advance(B)
    if (RES / 'sens_curves.npz').exists():
        fig_sensitivity()


if __name__ == '__main__':
    main()
