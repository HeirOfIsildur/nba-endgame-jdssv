"""Invariants of the rule encoding and of the solved game (run: pytest -q)."""
import numpy as np
import pytest

from nbaeg.dp import DM, params_hand, solve
from nbaeg.estimate import RES
from nbaeg.possessions import foul_to_give, timeouts_left


def test_timeouts_rules():
    # 2017-18 on: <= 2 after 3:00 of Q4, <= 4 in Q4, 7 per game
    assert timeouts_left(2020, 4, 100, 3, 0, 1, 1, 0, 0) == 2
    assert timeouts_left(2020, 4, 100, 5, 0, 1, 1, 1, 0) == 1
    assert timeouts_left(2020, 4, 100, 7, 0, 3, 3, 1, 0) == 0
    assert timeouts_left(2020, 4, 200, 2, 0, 0, 0, 0, 0) == 4
    assert timeouts_left(2020, 5, 10, 7, 0, 0, 1, 1, 0) == 1
    # before: 6 full (<= 3 in Q4) + one 20-second per half
    assert timeouts_left(2010, 4, 30, 4, 0, 1, 1, 1, 0) == 3
    assert timeouts_left(2010, 4, 30, 6, 1, 3, 3, 2, 1) == 0
    # before 2017-18: at most 2 full timeouts in the last 2:00 of Q4
    assert timeouts_left(2010, 4, 60, 4, 1, 2, 2, 2, 1, 2) == 0
    assert timeouts_left(2010, 4, 150, 4, 1, 2, 2, 2, 1, 2) == 1


def test_foul_to_give():
    assert foul_to_give(4, 100, 3, 0) == 1
    assert foul_to_give(4, 100, 3, 1) == 0
    assert foul_to_give(4, 150, 4, 0) == 0
    assert foul_to_give(5, 100, 3, 0) == 0


def check_game(par):
    tot = par['M2'] + par['X2'] + par['NONE'] + par['p_sf'] + par['p_tov']
    assert np.allclose(tot, 1) and (par['NONE'] >= -1e-12).all()
    assert np.allclose(par['M3'] + par['X3'], par['M2'] + par['X2'])
    S = solve(par)
    V = S['V']
    assert np.nanmin(V) >= -1e-12 and np.nanmax(V) <= 1 + 1e-12
    assert (V[0, :DM] == 0).all() and (V[0, DM + 1:] == 1).all()
    # in every start type: a bigger lead is never worse; a timeout in hand is never worse
    assert (np.diff(V[1:], axis=1) >= -1e-9).all()
    assert (V[:, :, :, 1:] - V[:, :, :, :-1] >= -1e-9).all()
    # the defence's foul option can only lower the offence's value (all start types except BC, where the
    # offence's timeout before the inbound comes after the defence's choice)
    assert (V[1:, :, 1:] <= S['NF'][1:, :, 1:] + 1e-12).all()


def test_hand_game():
    check_game(params_hand())


@pytest.mark.skipif(not (RES / 'params_fitted.json').exists(), reason='run the pipeline first')
def test_fitted_game():
    from nbaeg.dp import params_fitted
    check_game(params_fitted())


# ----------------------------------------------------------------------------- shot clock and FT rebounds
def _par(**over):
    par = params_hand()
    par.update(over)
    return par


def test_shot_clock_limits():
    """OREB and FC14 allow at most SC steps before the shot, FC14T (after a timeout) only SC - C_TO."""
    par = _par(C_TO=6)
    S = solve(par)
    SC, SCT = par['SC'], par['SC'] - par['C_TO']
    assert (S['BT'][SC + 1:, :, 3] <= SC).all() and (S['BT'][SC + 1:, :, 4] <= SC).all()
    assert (S['BT'][SCT + 1:, :, 5] <= SCT).all()
    # without the clock (FC) the best shot can come later than the FC14T limit
    assert (S['BT'][SC + 1:, :, 1] > SCT).any()


def test_timeout_after_oreb_keeps_remaining_clock():
    """A timeout after an offensive rebound leads to FC14T while the clock binds, else to FC."""
    par = _par(C_TO=6)
    S = solve(par)
    V, NF, SC, c = S['V'], S['NF'], par['SC'], par['C_TO']
    for n in (SC + 1, SC + 30, 200):
        assert (NF[n, :, 3, 1:] >= V[n - c, :, 5, :-1] - 1e-12).all()
    for n in (c + 1, SC):
        assert (NF[n, :, 3, 1:] >= V[n - c, :, 1, :-1] - 1e-12).all()
    # with a free timeout the remaining clock is a full SC: FC14T coincides with FC14
    S0 = solve(_par(C_TO=0))
    assert np.allclose(S0['V'][:, :, 5], S0['V'][:, :, 4])


def test_ft_rebound_split():
    """th_dm = 0 is the all-live model; th_dm = 1 makes every kept deliberate miss a frontcourt inbound."""
    par = _par(th_dm=1.0, rho_dm=0.4)
    S = solve(par)
    V, n, r = S['V'], 30, par['RTD'][0]
    n3 = n - r
    d = DM - 2
    lose = 1 - V[n3, DM + 2, 2, 0, 0, 0, 0]
    assert np.isclose(S['FMS'][n, d, 0, 0, 0, 0], 0.4 * V[n3, d, 1, 0, 0, 0, 0] + 0.6 * lose)


def test_ft_defensive_team_rebound():
    """eta_dm = 1: every lost deliberate miss is a dead ball inbounded in the opponent's backcourt."""
    par = _par(eta_dm=1.0, rho_dm=0.4)
    S = solve(par)
    V, n = S['V'], 30
    n3, d = n - par['RTD'][0], DM - 2
    keep = V[n3, d, 3, 0, 0, 0, 0]
    assert np.isclose(S['FMS'][n, d, 0, 0, 0, 0], 0.4 * keep + 0.6 * (1 - V[n3, DM + 2, 0, 0, 0, 0, 0]))
    # eta = 0 keeps the live defensive rebound
    S0 = solve(_par(rho_dm=0.4))
    V0 = S0['V']
    assert np.isclose(S0['FMS'][n, d, 0, 0, 0, 0], 0.4 * V0[n3, d, 3, 0, 0, 0, 0] + 0.6 * (1 - V0[n3, DM + 2, 2, 0, 0, 0, 0]))


def test_separate_ft_rebound_delays():
    """Ordinary misses are rebounded after RTO steps, deliberate misses after RTD steps."""
    par = _par(RTO=np.array([11]), RTD=np.array([7]))
    S = solve(par)
    V, n, d, q, rf = S['V'], 40, DM - 2, par['q'], par['rho_ft']
    mk = 1 - V[n, DM + 1, 0, 0, 0, 0, 0]
    no = n - 11
    fmk = q * mk + (1 - q) * (rf * V[no, d, 3, 0, 0, 0, 0] + (1 - rf) * (1 - V[no, DM + 2, 2, 0, 0, 0, 0]))
    assert np.isclose(S['FMK'][n, d, 0, 0, 0, 0], fmk)
    nd, rd = n - 7, par['rho_dm']
    assert np.isclose(S['FMS'][n, d, 0, 0, 0, 0], rd * V[nd, d, 3, 0, 0, 0, 0] + (1 - rd) * (1 - V[nd, DM + 2, 2, 0, 0, 0, 0]))


def _shot_value(S, par, n, d, s, a, b, ga, gb, k, tau):
    """Value of a k-point shot after tau steps, written from Equation (1) (no shot-clock states here)."""
    from nbaeg.dp import _ci
    V, F = S['V'], S['F']
    cs, t1, n2 = (1 if s >= 4 else s), tau - 1, n - tau
    lost = 1 - V[n2, _ci(-d), 2, b, a, gb, ga]
    mk = 1 - V[n2, _ci(-(d + k)), 0, b, a, gb, ga]
    n3 = n2 - par['RT']
    th, eta, sc = par['th_fg'], par['eta_fg'], (4 if n3 > par['SC'] else 1)
    keep = (1 - th) * V[n3, d + DM, 3, a, b, ga, gb] + th * V[n3, d + DM, sc, a, b, ga, gb]
    lose = (1 - eta) * (1 - V[n3, _ci(-d), 2, b, a, gb, ga]) + eta * (1 - V[n3, _ci(-d), 0, b, a, gb, ga])
    ms = par['rho_fg'] * keep + (1 - par['rho_fg']) * lose
    pm, px = (par['M2'][cs, t1], par['X2'][cs, t1]) if k == 2 else (par['M3'][cs, t1], par['X3'][cs, t1])
    return (par['p_tov'] + par['NONE'][cs, t1]) * lost + par['p_sf'] * F[k, n2, d + DM, a, b, ga, 0] + pm * mk + px * ms


def test_offence_beats_the_foul():
    """With p_pre the foul value is (1 - p_pre) Phi + p_pre max(Phi, best shot released before the foul); the shot
    value includes dead-ball team rebounds after a missed FG (th_fg, eta_fg)."""
    beta = 0.3
    par = _par(p_pre=beta, th_fg=0.3, eta_fg=0.2)
    S = solve(par)
    F, FE = S['F'], par['FE']
    for n, d, s in ((60, -3, 0), (45, -3, 1), (80, -2, 1), (35, -1, 2)):
        a = b = ga = gb = 0
        nf = n - FE[s]
        p3 = par['p_sf3'] if d <= -3 else 0.0
        phi = (1 - p3) * F[2, nf, d + DM, a, b, ga, 0] + p3 * F[3, nf, d + DM, a, b, ga, 0]
        early = max(_shot_value(S, par, n, d, s, a, b, ga, gb, k, tau) for tau in range(1, FE[s]) for k in (2, 3)
                    if k == 3 or par['ALLOW2'][1 if s >= 4 else s, tau - 1])
        assert np.isclose(S['FL'][n, d + DM, s, a, b, ga, gb], (1 - beta) * phi + beta * max(phi, early))
    # p_pre = 0 is the main model
    S0 = solve(_par(p_pre=0.0))
    assert np.allclose(S0['V'], solve(params_hand())['V'])


def test_break_even_identity_numeric():
    """miss - make = (K_DM - L_DM)(rho_DM - rho*) with the rho* stored by maps.summarize (separate delays and
    dead-ball team rebounds to both sides)."""
    from nbaeg.maps import FT_LEADS, ft_mix, summarize
    par = _par(th_dm=0.25, th_ft=0.15, eta_dm=0.3, eta_ft=0.05, rho_dm=0.4, RTO=np.array([11]),
               RTD=np.array([4, 7, 12]), WD=np.array([0.3, 0.5, 0.2]))
    S = solve(par)
    m = summarize(S, par)
    n = np.arange(1, S['V'].shape[0])
    for j, ld in enumerate(FT_LEADS):
        d = DM + ld
        kdm, ldm = ft_mix(S['V'], n, par['RTD'], par['WD'], d, 0, 0, 0.25, 0.3, par['SC'])
        ok = (n - par['RTD'].min() > 0) & (np.abs(kdm - ldm) > 1e-6)
        lhs = m['ft'][:, j, 0, 0] / 100
        rhs = (kdm - ldm) * (par['rho_dm'] - m['rstar'][:, j, 0, 0])
        assert np.allclose(lhs[ok], rhs[ok], atol=1e-10)


def test_rebound_delay_distribution():
    """A delay distribution averages the miss values over its delays (weights sum to one)."""
    from nbaeg.dp import delays
    dd = delays('RTD', 'WD', ([0.3, 0.7, 0.71, 1.2], [1, 1, 1, 1]), 0.69)
    assert list(dd['RTD']) == [3, 7, 12] and np.allclose(dd['WD'], [0.25, 0.5, 0.25])
    par = _par(**dd)
    S = solve(par)
    V, n, d, rd = S['V'], 40, DM - 2, par['rho_dm']
    want = sum(w * (rd * V[n - r, d, 3, 0, 0, 0, 0] + (1 - rd) * (1 - V[n - r, DM + 2, 2, 0, 0, 0, 0]))
               for r, w in zip(dd['RTD'], dd['WD']))
    assert np.isclose(S['FMS'][n, d, 0, 0, 0, 0], want)
