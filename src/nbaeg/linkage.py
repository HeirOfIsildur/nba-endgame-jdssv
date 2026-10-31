"""Audit of the deliberate-miss proxy: event linkage, rebound type, timing and transport of the rate.

Proxy sample: last free throws of a 2- or 3-shot trip in Q4/OT, shooter's team down 2 or 3 before the FT,
at most 3 s left, missed, shooter with a season FT% (the cohort of estimate.py, before the exclusion of
rebounds logged after the buzzer). For each miss the first substantive event after the FT (substitutions,
replays, timeouts and ejections skipped) is classified; the rebound linked by `possessions.last_fts()` is
checked against it.

Outputs:
  results/dm_linkage.csv   counts by what followed the miss (all proxy misses / the estimation cohort)
  results/dm_timing.csv    seconds from the FT to the linked rebound, proxy vs ordinary misses
  results/dm_transport.csv OREB rate of misses by subgroup and definition, with 95% game-bootstrap intervals
usage: python -m nbaeg.linkage
"""
import numpy as np
import pandas as pd

from .build import DER
from .estimate import RES

SKIP = {'Substitution', 'Instant Replay', 'Timeout', 'Ejection'}


def first_events(F, L):
    """First substantive event after each FT in F (same game and period)."""
    L = L.sort_values(['gameId', 'seq']).reset_index(drop=True)        # `seq` restarts every season
    pos = pd.Series(np.arange(len(L)), index=pd.MultiIndex.from_arrays([L.gameId.values, L.seq.values]))
    out = []
    for r in F.itertuples():
        i = pos[(r.gameId, r.seq)] + 1
        ev = None
        while i < len(L) and L.gameId.iat[i] == r.gameId and L.period.iat[i] == r.period:
            if L.actionType.iat[i] not in SKIP:
                ev = L.iloc[i]
                break
            i += 1
        if ev is None or ev.actionType == 'period':
            cat = 'period ends before a rebound'
        elif ev.actionType == 'Rebound':
            team = ev.personId > 1e9
            same = ev.side == r.side
            cat = {(True, False): 'live offensive rebound', (True, True): 'team rebound to the shooter (dead ball)',
                   (False, False): 'defensive rebound', (False, True): 'team rebound to the opponent (dead ball)'}[(same, team)]
        elif ev.actionType in ('Violation', 'Turnover'):
            cat = 'violation or turnover before a rebound'
        elif ev.actionType in ('Missed Shot', 'Made Shot'):
            cat = 'tip or shot without a logged rebound'
        else:
            cat = 'other'
        out.append(dict(gameId=r.gameId, seq=r.seq, category=cat, ev_seq=(ev.seq if ev is not None else np.nan),
                        ev_t=(ev.t if ev is not None else np.nan)))
    return pd.DataFrame(out)


def boot_rate(x, num, den, B=1000, seed=7):
    """Ratio of sums with a 95% Poisson game-bootstrap interval."""
    games = x.gameId.values
    ug, gi = np.unique(games, return_inverse=True)
    W = np.random.default_rng(seed).poisson(1.0, (B, len(ug))).astype(float)
    n_ = np.bincount(gi, weights=den, minlength=len(ug))
    k_ = np.bincount(gi, weights=num, minlength=len(ug))
    with np.errstate(invalid='ignore', divide='ignore'):
        bs = (W @ k_) / (W @ n_)
    return num.sum() / den.sum(), np.nanpercentile(bs, 2.5), np.nanpercentile(bs, 97.5)


def main():
    F = pd.read_parquet(DER / 'last_ft.parquet')
    L = pd.read_parquet(DER / 'late.parquet', columns=['gameId', 'period', 'seq', 'actionType', 'side', 'personId', 't'])
    L = L[L.period >= 4]
    px = F[F.lead_b.isin([-2, -3]) & (F.t <= 3) & (F.made == 0) & F.pct.notna()].copy()
    ev = first_events(px, L)
    px = px.merge(ev, on=['gameId', 'seq'])
    # the rebound linked by last_fts(): first rebound after the FT in the same game and period
    R = L[L.actionType == 'Rebound'][['gameId', 'seq']].rename(columns={'seq': 'seq_reb'}).sort_values('seq_reb')
    px = pd.merge_asof(px.sort_values('seq'), R, left_on='seq', right_on='seq_reb', by='gameId', direction='forward')
    px['cohort'] = px.oreb.notna() & (px.t_reb > 0)              # estimation cohort (rebound before the buzzer)
    px['linked_is_first'] = px.seq_reb == px.ev_seq
    rows = []
    for cat, d in px.groupby('category'):
        rows.append(dict(category=cat, n_all=len(d), n_cohort=int(d.cohort.sum()),
                         linked_rebound_is_first_event=int((d.linked_is_first & d.cohort).sum())))
    tab = pd.DataFrame(rows).sort_values('n_all', ascending=False)
    tab.loc[len(tab)] = dict(category='total', n_all=len(px), n_cohort=int(px.cohort.sum()),
                             linked_rebound_is_first_event=int((px.linked_is_first & px.cohort).sum()))
    tab['n_after_buzzer'] = [int(((px.category == c) & (px.t_reb == 0)).sum()) if c != 'total'
                             else int((px.t_reb == 0).sum()) for c in tab.category]
    tab.to_csv(RES / 'dm_linkage.csv', index=False)
    print(tab.to_string(index=False))
    # timing
    fm = F[(F.made == 0) & F.oreb.notna() & (F.t_reb > 0)]
    dm = fm[fm.lead_b.isin([-2, -3]) & (fm.t <= 3) & fm.pct.notna()]
    ordn = fm[fm.t >= 8]
    tim = []
    for lab, x in (('deliberate-miss proxy', dm), ('ordinary misses (>= 8 s left)', ordn)):
        g = x.t - x.t_reb
        tim.append(dict(sample=lab, n=len(x), mean=g.mean(), q25=g.quantile(.25), median=g.median(), q75=g.quantile(.75),
                        share_le_1s=(g <= 1).mean(), share_gt_2s=(g > 2).mean(), max=g.max()))
    tim = pd.DataFrame(tim)
    tim.to_csv(RES / 'dm_timing.csv', index=False, float_format='%.3f')
    print(tim.round(3).to_string(index=False))
    # rates: definitions and subgroups
    rows = []

    def add(group, lab, x, num, den):
        est, lo, hi = boot_rate(x, np.asarray(num, float), np.asarray(den, float))
        rows.append(dict(group=group, definition=lab, n=int(np.sum(den)), rate=est, lo=lo, hi=hi))
    live = dm.oreb * (1 - dm.reb_team)
    add('proxy (down 2-3, <= 3 s)', 'kept, team rebounds counted (main)', dm, dm.oreb, np.ones(len(dm)))
    add('proxy (down 2-3, <= 3 s)', 'live offensive rebounds only', dm, live, np.ones(len(dm)))
    amb = px.loc[px.cohort & ~px.linked_is_first, ['gameId', 'seq']]          # linked rebound not the first event
    clean = dm[~dm.set_index(['gameId', 'seq']).index.isin(amb.set_index(['gameId', 'seq']).index)]
    add('proxy (down 2-3, <= 3 s)', 'excluding misses whose linked rebound is not the first event', clean, clean.oreb,
        np.ones(len(clean)))
    nt = dm[dm.reb_team == 0]
    add('proxy (down 2-3, <= 3 s)', 'team rebounds dropped from the sample', nt, nt.oreb, np.ones(len(nt)))
    add('proxy (down 2-3, <= 3 s)', 'share of kept rebounds that are team rebounds', dm[dm.oreb == 1],
        dm[dm.oreb == 1].reb_team, np.ones(int((dm.oreb == 1).sum())))
    for lab, m in (('down 2', dm.lead_b == -2), ('down 3', dm.lead_b == -3), ('2000-01 to 2016-17', dm.season <= 2016),
                   ('2017-18 to 2025-26', dm.season >= 2017)):
        x = dm[m]
        add(lab, 'kept, team rebounds counted (main)', x, x.oreb, np.ones(len(x)))
    up1 = fm[(fm.lead_b == 1) & (fm.t <= 3) & fm.pct.notna()]
    add('leading by 1 (<= 3 s), all misses', 'kept, team rebounds counted (main)', up1, up1.oreb, np.ones(len(up1)))
    add('ordinary (>= 8 s)', 'kept, team rebounds counted (main)', ordn, ordn.oreb, np.ones(len(ordn)))
    add('ordinary (>= 8 s)', 'share of kept rebounds that are team rebounds', ordn[ordn.oreb == 1],
        ordn[ordn.oreb == 1].reb_team, np.ones(int((ordn.oreb == 1).sum())))
    # how deliberate are misses when leading by 1? (excess miss rate, all trips)
    t1 = F[(F.lead_b == 1) & (F.t <= 3) & F.pct.notna()]
    add('leading by 1 (<= 3 s), all trips', 'miss rate', t1, 1 - t1.made, np.ones(len(t1)))
    add('leading by 1 (<= 3 s), all trips', 'expected miss rate from season FT%', t1, 1 - t1.pct, np.ones(len(t1)))
    # mixture-corrected rate for leaders up 1 (as in estimate.py: deliberate share w from the excess miss rate,
    # accidental misses rebounded at the ordinary rate), with shared game weights
    t1m = t1[~((t1.made == 0) & ~(t1.t_reb > 0))]
    games = np.union1d(t1m.gameId.unique(), ordn.gameId.unique())
    gi = pd.Series(np.arange(len(games)), index=games)

    def mix(wt, wo):
        miss = np.sum(wt * (1 - t1m.made.values)) / np.sum(wt)
        w = max(1e-6, (miss - np.sum(wt * (1 - t1m.pct.values)) / np.sum(wt)) / miss)
        mm = (t1m.made.values == 0) & t1m.oreb.notna().values
        raw = np.sum(wt[mm] * t1m.oreb.values[mm]) / np.sum(wt[mm])
        rft = np.sum(wo * ordn.oreb.values) / np.sum(wo)
        return (raw - (1 - w) * rft) / w
    est = mix(np.ones(len(t1m)), np.ones(len(ordn)))
    rng = np.random.default_rng(8)
    bs = []
    for _ in range(1000):
        g = rng.poisson(1.0, len(games)).astype(float)
        bs.append(mix(g[gi[t1m.gameId].values], g[gi[ordn.gameId].values]))
    rows.append(dict(group='leading by 1 (<= 3 s), all trips', definition='mixture-corrected deliberate-miss rate',
                     n=len(t1m), rate=est, lo=np.nanpercentile(bs, 2.5), hi=np.nanpercentile(bs, 97.5)))
    tr = pd.DataFrame(rows)
    tr.to_csv(RES / 'dm_transport.csv', index=False, float_format='%.4f')
    print(tr.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
