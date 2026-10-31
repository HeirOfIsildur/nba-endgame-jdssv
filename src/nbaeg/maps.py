"""Decision-map summaries of a solved game (time x score x timeouts), used for the point estimate, the
hand-set model and every bootstrap replicate.

  foul[i, l, s, a, g]  leader's gain (pp of win probability) from fouling at once rather than letting
                       the trailer play: trailer down l+1 (l = 0, 1, 2), at time GRID[i];
                       s = 0 backcourt inbound, 1 frontcourt inbound after a timeout;
                       a = trailer timeouts left (0, 1); g = leader's foul to give (0, 1).
  ft[i, j, a, b]       shooter's gain (pp) from missing the last FT on purpose; shooter's lead before
                       the FT = FT_LEADS[j]; a / b = shooter's / opponent's timeouts left (0, 1).
  rstar[i, j, a, b]    break-even deliberate-miss rebound rate (Mathematica-checked identity): miss - make =
                       (K_DM - L_DM)(rho_DM - rstar), with K_j / L_j the values of a kept / lost miss of type j
                       (FT ordinary, DM deliberate) at its own rebound delay (averaged over the delay distribution); kept rebounds are live or, with
                       share th_j, frontcourt inbounds; lost ones live or, with share eta_j, opponent BC inbounds.
  adv[i, l, k]         value (pp) of holding one more timeout (k -> k+1, k = 0, 1) for a team down l
                       (l = 0..3) inbounding a dead ball: the option to stop the clock and advance.
                       Never negative, because the option can be left unused.
  shot3[i, l, s]       1 if the trailer's best play is a 3 (down l+1, no timeouts, no foul to give).
  wp[i, j, s]          win probability of the team with the ball, lead -3..3, no timeouts, no fouls to give.
"""
import numpy as np

from .dp import DM, N

FT_LEADS = [-3, -2, -1, 1, 2]


def ft_parts(V, n3, d, a, b, th, eta, SC):
    """Values (shooter's view) of a kept and of a lost missed last FT rebounded at n3 (arrays; the terminal value
    where n3 <= 0, as in the solver). The shooting team has a timeouts, the opponent b, neither a foul to give."""
    n3 = np.asarray(n3)
    ok = n3 > 0
    m3 = np.maximum(n3, 0)
    keep = (1 - th) * V[m3, d, 3, a, b, 0, 0] + th * V[m3, d, np.where(m3 > SC, 4, 1), a, b, 0, 0]
    lose = (1 - eta) * (1 - V[m3, 2 * DM - d, 2, b, a, 0, 0]) + eta * (1 - V[m3, 2 * DM - d, 0, b, a, 0, 0])
    ld = d - DM
    term = 1.0 if ld > 0 else (0.0 if ld < 0 else 0.5)
    return np.where(ok, keep, term), np.where(ok, lose, term)


def ft_mix(V, n, R, W, d, a, b, th, eta, SC):
    """Kept and lost values averaged over the rebound-delay distribution (delays R, weights W)."""
    parts = [ft_parts(V, n - r, d, a, b, th, eta, SC) for r in R]
    return sum(w * p[0] for w, p in zip(W, parts)), sum(w * p[1] for w, p in zip(W, parts))


def summarize(S, par):
    V, NF, FL, BK = S['V'], S['NF'], S['FL'], S['BK']
    n = np.arange(1, N + 1)
    foul = np.full((N, 3, 2, 2, 2), np.nan)
    for l in range(3):
        d = DM - (l + 1)
        for s in (0, 1):
            for a in (0, 1):
                for g in (0, 1):
                    foul[:, l, s, a, g] = 100 * (NF[n, d, s, a, 0, 0, g] - FL[n, d, s, a, 0, 0, g])
    ft = np.zeros((N, len(FT_LEADS), 2, 2))
    rstar = np.full((N, len(FT_LEADS), 2, 2), np.nan)
    q, rft = par['q'], par['rho_ft']
    for j, ld in enumerate(FT_LEADS):
        d = DM + ld
        for a in (0, 1):
            for b in (0, 1):
                ft[:, j, a, b] = 100 * (S['FMS'][n, d, a, b, 0, 0] - S['FMK'][n, d, a, b, 0, 0])
                mk = 1 - V[n, DM - (ld + 1), 0, b, a, 0, 0]
                kdm, ldm = ft_mix(V, n, par['RTD'], par['WD'], d, a, b, par['th_dm'], par['eta_dm'], par['SC'])
                kft, lft = ft_mix(V, n, par['RTO'], par['WO'], d, a, b, par['th_ft'], par['eta_ft'], par['SC'])
                with np.errstate(invalid='ignore', divide='ignore'):
                    rstar[:, j, a, b] = (q * (mk - ldm) + (1 - q) * (rft * (kft - ldm) + (1 - rft) * (lft - ldm))) \
                        / (kdm - ldm)
                rstar[n - par['RTD'].min() <= 0, j, a, b] = np.nan        # every rebound comes after the buzzer
    adv = np.zeros((N, 4, 2))
    for l in range(4):
        for a in (0, 1):
            adv[:, l, a] = 100 * (V[n, DM - l, 0, a + 1, 0, 0, 0] - V[n, DM - l, 0, a, 0, 0, 0])
    shot3 = np.zeros((N, 3, 4))
    for l in range(3):
        for s in range(4):
            shot3[:, l, s] = (BK[n, DM - (l + 1), s, 0, 0, 0, 0] == 3).astype(float)
    wp = np.zeros((N, 7, 4))
    for j, ld in enumerate(range(-3, 4)):
        for s in range(4):
            wp[:, j, s] = V[n, DM + ld, s, 0, 0, 0, 0]
    return dict(foul=foul, ft=ft, rstar=rstar, adv=adv, shot3=shot3, wp=wp)
