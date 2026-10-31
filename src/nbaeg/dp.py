"""Exact end-game model: a zero-sum stochastic game solved by backward induction on a 0.1 s grid.

State of a possession phase: V[n, d, s, a, b, ga, gb] = win probability of the team with the ball, where
  n  time left in steps of DT (t = n * DT, horizon 24 s; the shot clock binds only after an offensive rebound),
  d  lead of the team with the ball (index d + DM, clipped to [-DM, DM]),
  s  start type: 0 BC (dead ball, backcourt), 1 FC (frontcourt inbound after a timeout),
                 2 LIVE (defensive rebound / turnover), 3 OREB (offensive rebound),
                 4 FC14 (frontcourt inbound after a defensive side-out that followed an offensive rebound with
                   more than 14 s left: the shot clock is reset to max(remaining, 14 s) = 14 s, NBA Rule 7-IV-e),
                 5 FC14T (frontcourt inbound after a live-ball timeout that followed an offensive rebound with more
                   than 14 s left: the shot clock keeps what was left, 14 s - C_TO, Rule 5-VI-7),
  the remaining shot clock CL[s] binds only when it is shorter than the game clock (CL = 14 s in OREB and
  FC14, 14 s - C_TO in FC14T; off otherwise, because t <= 24 s and a backcourt possession starts with 24 s),
  a, b   timeouts left (0, 1, 2+) of the offence and of the defence,
  ga, gb foul to give (0/1) of the offence and of the defence.
Free-throw trips: F[j, n, d, a, b, ga, gb] = win probability of the shooting team with j FTs left.
Overtime is worth OT to both sides.

Moves, in order (each side maximises its own win probability):
  s = 0      the offence may call a timeout (no clock) and inbound in the frontcourt (s = 1);
  s = 0, 1   the defence may foul after FE[s] steps: with a foul to give it is a side-out
             (FC for FC/OREB, BC otherwise), else two FTs (three with prob p_sf3 if d <= -3);
             with probability p_pre the offence beats a committed foul: it may get a shot off after
             tau < FE[s] steps (its best such shot) or accept the foul (any s);
  s = 2, 3   the defence may foul (FE[s]); otherwise the offence may call a timeout after C_TO
             steps (-> s = 1, or s = 5 after an OREB whose shot clock binds) or play;
  s = 1,4,5  the defence may foul after FE[FC] steps; a side-out restarts FC (from FC), or FC14 / FC
             (from OREB, FC14, FC14T: the clock resets to 14 s, binding only if more than 14 s are left);
  play       hold (worth term(d); where the shot clock binds, a shot must come within CL[s] steps and holding
             is a violation: the opponent inbounds a dead ball in its backcourt) or a k-point shot after tau steps: turnover (p_tov, opponent live),
             shooting foul (p_sf, k FTs, defence loses its foul to give), made (M_k), missed
             (X_k; rebound after RT steps, kept with rho_fg as OREB, else opponent LIVE; a share th_fg of the
             kept and eta_fg of the lost rebounds are dead-ball team rebounds, restarted as for free throws),
             no shot got off (NONE; opponent LIVE);
  last FT    the shooter may miss on purpose: rebound after RTD steps, kept with rho_dm (an ordinary
             miss: after RTO steps, kept with rho_ft; RTD and RTO are delay distributions on the grid, with
             weights WD and WO: one delay with weight 1 in the main analysis); a share th_dm (th_ft) of the kept rebounds are dead-ball
             team rebounds (out of bounds off the defence): a frontcourt inbound (FC, or FC14 if > 14 s left)
             instead of a live offensive rebound; a share eta_dm (eta_ft) of the lost rebounds are dead-ball
             team rebounds to the defence (out of bounds off the shooting team): the opponent inbounds in its
             backcourt (BC) instead of a live defensive rebound.
"""
import json

import numpy as np
from numba import njit

from .estimate import DT, RES, TYPES

DM = 12
ND = 2 * DM + 1
NA = 3
NS_ = 6          # start types: 0 BC, 1 FC, 2 LIVE, 3 OREB, 4 FC14 (side-out, 14-s clock), 5 FC14T (timeout, clock 14 s - C_TO)
N = int(round(24.0 / DT))


@njit(cache=True)
def _term(d, ot):
    return 1.0 if d > 0 else (0.0 if d < 0 else ot)


@njit(cache=True)
def _ci(d):
    return min(max(d, -DM), DM) + DM


@njit(cache=True)
def _ft_miss(V, n3, di, d, a, b, ga, gb, rho, th, eta, SC, ot):
    """Value of a miss (last FT, or FG) rebounded at n3: kept with rho (live OREB, or with share th a frontcourt
    inbound), else lost (live defensive rebound, or with share eta the opponent's backcourt inbound)."""
    if n3 <= 0:
        return _term(d, ot)
    keep = (1.0 - th) * V[n3, di, 3, a, b, ga, gb] + th * V[n3, di, 4 if n3 > SC else 1, a, b, ga, gb]
    lose = (1.0 - eta) * (1.0 - V[n3, _ci(-d), 2, b, a, gb, ga]) + eta * (1.0 - V[n3, _ci(-d), 0, b, a, gb, ga])
    return rho * keep + (1.0 - rho) * lose


@njit(cache=True)
def _solve(M2, M3, X2, X3, NONE, ALLOW2, p_sf, p_tov, rho_fg, th_fg, eta_fg, rho_ft, rho_dm, th_ft, th_dm, eta_ft, eta_dm, q, p_sf3,
           p_pre, FE, C_TO, RT, RTO, WO, RTD, WD, ot, SC):
    N = M2.shape[1]                                        # horizon in steps
    SCT = max(1, SC - C_TO)                                # shot clock left after a live timeout following an OREB
    V = np.zeros((N + 1, ND, NS_, NA, NA, 2, 2))
    NF = np.full((N + 1, ND, NS_, NA, NA, 2, 2), np.nan)     # offence's best play (no defensive foul)
    FL = np.full((N + 1, ND, NS_, NA, NA, 2, 2), np.nan)     # value if the defence fouls (NaN = cannot)
    BK = np.zeros((N + 1, ND, NS_, NA, NA, 2, 2), np.int8)   # best shot: 0 hold, 2, 3
    BT = np.zeros((N + 1, ND, NS_, NA, NA, 2, 2), np.int16)  # its tau (steps)
    F = np.zeros((4, N + 1, ND, NA, NA, 2, 2))
    FMK = np.zeros((N + 1, ND, NA, NA, 2, 2))              # last FT: try to make it
    FMS = np.zeros((N + 1, ND, NA, NA, 2, 2))              # last FT: miss on purpose
    for di in range(ND):
        V[0, di] = _term(di - DM, ot)
    for n in range(N + 1):
        if n > 0:
            for so in range(NS_):
                s = (1, 4, 5, 0, 2, 3)[so]               # FC first: the BC timeout option reads it
                cs = 1 if s >= 4 else s                  # FC14 and FC14T use the frontcourt shot curves
                cl = SC if (s == 3 or s == 4) else (SCT if s == 5 else 0)     # remaining shot clock
                for di in range(ND):
                    d = di - DM
                    for a in range(NA):
                        for b in range(NA):
                            for ga in range(2):
                                for gb in range(2):
                                    # hold; where the shot clock (cl steps left) runs out before the game
                                    # clock, holding is a violation: the opponent's dead ball in its backcourt
                                    clock = cl > 0 and n > cl
                                    if clock:
                                        best = 1.0 - V[n - cl, _ci(-d), 0, b, a, gb, ga]
                                    else:
                                        best = _term(d, ot)
                                    bk, bt = 0, 0
                                    early = -1.0                 # best shot released before a foul arrives
                                    for tau in range(1, (cl if clock else n) + 1):
                                        n2 = n - tau
                                        t1 = tau - 1
                                        if n2 <= 0:
                                            v_lost = _term(d, ot)
                                        else:
                                            v_lost = 1.0 - V[n2, _ci(-d), 2, b, a, gb, ga]
                                        for k in (2, 3):
                                            if k == 2 and not ALLOW2[cs, t1]:
                                                continue
                                            if n2 <= 0:
                                                v_mk = _term(d + k, ot)
                                                v_ms = _term(d, ot)
                                                v_sf = F[k, 0, di, a, b, ga, 0]
                                            else:
                                                v_mk = 1.0 - V[n2, _ci(-(d + k)), 0, b, a, gb, ga]
                                                v_sf = F[k, n2, di, a, b, ga, 0]
                                                # kept (OREB, or a frontcourt inbound) or lost (opponent LIVE, or BC)
                                                v_ms = _ft_miss(V, n2 - RT, di, d, a, b, ga, gb, rho_fg, th_fg, eta_fg,
                                                                SC, ot)
                                            if k == 2:
                                                pm, px = M2[cs, t1], X2[cs, t1]
                                            else:
                                                pm, px = M3[cs, t1], X3[cs, t1]
                                            v = (p_tov + NONE[cs, t1]) * v_lost + p_sf * v_sf + pm * v_mk + px * v_ms
                                            if tau < FE[s] and v > early:
                                                early = v
                                            if v > best + 1e-13:
                                                best, bk, bt = v, k, tau
                                    play = best
                                    if (s == 2 or s == 3) and a > 0:         # timeout from a live ball
                                        nt = n - C_TO
                                        # after an OREB whose 14-s clock binds (n > SC), the timeout keeps the
                                        # shot clock as it was, 14 s - C_TO: frontcourt inbound in FC14T (s = 5)
                                        s_to = 5 if (s == 3 and n > SC) else 1
                                        vt = V[nt, di, s_to, a - 1, b, ga, gb] if nt > 0 else _term(d, ot)
                                        play = max(play, vt)
                                    NF[n, di, s, a, b, ga, gb] = play
                                    BK[n, di, s, a, b, ga, gb] = bk
                                    BT[n, di, s, a, b, ga, gb] = bt
                                    val = play
                                    nf = n - FE[s]
                                    if nf > 0:                               # the defence may foul
                                        if gb == 1:
                                            if s == 1:
                                                s2 = 1
                                            elif s >= 3:                     # frontcourt side-out: clock max(left, 14 s)
                                                s2 = 4 if nf > SC else 1
                                            else:
                                                s2 = 0
                                            vf = V[nf, di, s2, a, b, ga, 0]
                                        else:
                                            p3 = p_sf3 if d <= -3 else 0.0
                                            vf = (1.0 - p3) * F[2, nf, di, a, b, ga, 0] + p3 * F[3, nf, di, a, b, ga, 0]
                                        # with prob p_pre the offence beats the foul: an earlier shot, if better
                                        vf = (1.0 - p_pre) * vf + p_pre * max(vf, early)
                                        FL[n, di, s, a, b, ga, gb] = vf
                                        val = min(val, vf)
                                    if s == 0 and a > 0:                     # timeout before the inbound
                                        val = max(val, V[n, di, 1, a - 1, b, ga, gb])
                                    V[n, di, s, a, b, ga, gb] = val
        # free-throw trips at this time (they take no clock); they read V[n, ., 0] of the opponent
        for di in range(ND):
            d = di - DM
            for a in range(NA):
                for b in range(NA):
                    for ga in range(2):
                        for gb in range(2):
                            if n == 0:
                                mk = _term(d + 1, ot)
                                ms_o = _term(d, ot)
                                ms_d = ms_o
                            else:
                                mk = 1.0 - V[n, _ci(-(d + 1)), 0, b, a, gb, ga]
                                # ordinary and deliberate misses: own rebound delay, rates and team-rebound shares
                                ms_o = 0.0
                                for k in range(RTO.shape[0]):
                                    ms_o += WO[k] * _ft_miss(V, n - RTO[k], di, d, a, b, ga, gb, rho_ft, th_ft, eta_ft, SC, ot)
                                ms_d = 0.0
                                for k in range(RTD.shape[0]):
                                    ms_d += WD[k] * _ft_miss(V, n - RTD[k], di, d, a, b, ga, gb, rho_dm, th_dm, eta_dm, SC, ot)
                            vm = q * mk + (1.0 - q) * ms_o
                            FMK[n, di, a, b, ga, gb] = vm
                            FMS[n, di, a, b, ga, gb] = ms_d
                            F[1, n, di, a, b, ga, gb] = max(vm, ms_d)
        for j in (2, 3):
            for di in range(ND):
                dn = _ci(di - DM + 1)
                for a in range(NA):
                    for b in range(NA):
                        for ga in range(2):
                            for gb in range(2):
                                F[j, n, di, a, b, ga, gb] = q * F[j - 1, n, dn, a, b, ga, gb] + \
                                    (1.0 - q) * F[j - 1, n, di, a, b, ga, gb]
    return V, NF, FL, BK, BT, F, FMK, FMS


# ----------------------------------------------------------------------------- parameters
def curves_from_estimates(cv, sc, press=None):
    """Turn the estimated curves into the DP's per-(s, tau) outcome probabilities.

    p_sf and p_tov are constants; the probability of getting a shot off is capped at 1 - p_sf - p_tov
    so that made + missed + foul + turnover + no-shot sum to one.
    """
    press = int(sc.get('press', 1)) if press is None else press     # 1: pressure (Q4/OT) curves, 0: score-neutral
    thr = sc.get('share2_min', 0.25)                                 # two-point feasibility threshold
    M2, M3, X2, X3, NONE = (np.zeros((4, N)) for _ in range(5))
    ALLOW2 = np.zeros((4, N), np.bool_)
    other = sc['p_sfoul'] + sc['p_tov']
    for s, name in enumerate(TYPES):
        ps = np.minimum(cv[f'pshot_{name}_{press}'][:N], 1.0 - other)
        f2, f3 = cv[f'f2_{name}_{press}'][:N], cv[f'f3_{name}_{press}'][:N]
        M2[s], M3[s] = ps * f2, ps * f3
        X2[s], X3[s] = ps * (1 - f2), ps * (1 - f3)
        NONE[s] = 1.0 - ps - other
        ALLOW2[s] = cv[f'share2_{name}_0'][:N] >= thr       # 2s only where teams attempt them (score-neutral)
    return M2, M3, X2, X3, NONE, ALLOW2


def SC_FC(n, SC):
    """Start type of a dead-ball frontcourt inbound with n steps left: FC14 while the 14-s clock binds, else FC."""
    return 4 if n > SC else 1


def steps(x):
    return max(1, int(round(x / DT)))


def params_fitted(sc=None, cv=None, **over):
    sc = dict(json.load(open(RES / 'params_fitted.json'))) if sc is None else dict(sc)
    if cv is None:
        z = np.load(RES / 'curves_fitted.npz')
        cv = {k: z[k] for k in z.files}
    sc.update(over)
    M2, M3, X2, X3, NONE, ALLOW2 = curves_from_estimates(cv, sc)
    FE = np.array([steps(sc[f'foul_el_{t}']) for t in TYPES] + [steps(sc['foul_el_FC'])] * 2, np.int64)
    return dict(M2=M2, M3=M3, X2=X2, X3=X3, NONE=NONE, ALLOW2=ALLOW2, p_sf=sc['p_sfoul'],
                p_tov=sc['p_tov'], rho_fg=sc['rho_fg'], th_fg=sc.get('th_fg', 0.0), eta_fg=sc.get('eta_fg', 0.0),
                rho_ft=sc['rho_ft'], rho_dm=sc['rho_dm'],
                th_ft=sc.get('th_ft', 0.0), th_dm=sc.get('th_dm', 0.0), eta_ft=sc.get('eta_ft', 0.0),
                eta_dm=sc.get('eta_dm', 0.0), q=sc['q'], p_sf3=sc['p_sf3'], p_pre=sc.get('p_pre', 0.0), FE=FE,
                C_TO=steps(sc['to_cost_live']), RT=steps(sc['reb_time']),
                **delays('RTO', 'WO', sc.get('rto_dist'), sc.get('reb_time_ft_ord', sc['reb_time_ft'])),
                **delays('RTD', 'WD', sc.get('rtd_dist'), sc['reb_time_ft']), ot=0.5, SC=steps(sc.get('shot_clock', 14.0)))


def delays(kn, kw, dist, mean):
    """FT rebound delays on the grid: one delay (the estimated mean) or a distribution given as (seconds, weights)."""
    if dist is None:
        return {kn: np.array([steps(mean)], np.int64), kw: np.array([1.0])}
    sec, w = dist
    n = np.array([steps(x) for x in sec], np.int64)
    u = np.unique(n)
    wu = np.array([np.sum(np.asarray(w, float)[n == k]) for k in u])
    return {kn: u, kw: wu / wu.sum()}


HAND = dict(p2=0.545, p3=0.365, tov=0.13, sfoul=0.08, oreb=0.25, oreb_ft=0.13, foul_el=2.0, p_sf_int=0.05,
            ft=0.78)


def params_hand():
    """The hand-set prototype (research_hub audits/17 endgame2.P) on this state space: one shot curve
    for every start type (so the advance after a timeout is worth nothing), league FG%, and a
    deliberate miss rebounded as often as an ordinary one."""
    h = HAND
    tau = np.arange(1, N + 1) * DT
    eff = np.interp(tau, [0, 1, 2, 3, 4, 5], [0, .15, .45, .75, .9, 1.0])
    play = 1 - h['tov'] - h['sfoul']
    M2 = np.tile(play * h['p2'] * eff, (4, 1))
    M3 = np.tile(play * h['p3'] * eff, (4, 1))
    X2 = np.tile(play * (1 - h['p2'] * eff), (4, 1))
    X3 = np.tile(play * (1 - h['p3'] * eff), (4, 1))
    NONE = np.zeros((4, N))
    ALLOW2 = np.tile(tau > 2.0, (4, 1))
    FE = np.full(NS_, steps(h['foul_el']), np.int64)
    return dict(M2=M2, M3=M3, X2=X2, X3=X3, NONE=NONE, ALLOW2=ALLOW2, p_sf=h['sfoul'], p_tov=h['tov'],
                rho_fg=h['oreb'], th_fg=0.0, eta_fg=0.0, rho_ft=h['oreb_ft'], rho_dm=h['oreb_ft'], th_ft=0.0, th_dm=0.0, eta_ft=0.0, eta_dm=0.0,
                q=h['ft'], p_sf3=h['p_sf_int'], p_pre=0.0, FE=FE, C_TO=steps(1.0), RT=steps(1.0),
                **delays('RTO', 'WO', None, 1.0), **delays('RTD', 'WD', None, 1.0), ot=0.5, SC=steps(14.0))


ORDER = ['M2', 'M3', 'X2', 'X3', 'NONE', 'ALLOW2', 'p_sf', 'p_tov', 'rho_fg', 'th_fg', 'eta_fg', 'rho_ft', 'rho_dm', 'th_ft', 'th_dm', 'eta_ft',
         'eta_dm', 'q', 'p_sf3', 'p_pre', 'FE', 'C_TO', 'RT', 'RTO', 'WO', 'RTD', 'WD', 'ot', 'SC']


def solve(par):
    out = _solve(*[par[k] for k in ORDER])
    return dict(zip(['V', 'NF', 'FL', 'BK', 'BT', 'F', 'FMK', 'FMS'], out))


def ix(t):
    return int(round(t / DT))
