"""Possession phases in the last 60 s of every period, with the state at the start and the first
terminal action.

A phase starts when a side gains the ball:
  fg   after the opponent's made field goal (dead ball, backcourt inbound),
  ft   after the opponent's made last free throw (dead ball, backcourt inbound),
  dreb after a defensive rebound (live),
  oreb after an offensive rebound (live, frontcourt; a new phase of the same possession),
  tov  after an opponent turnover.
A rebound credited to a team rather than a player (reb_team) is a dead ball: the restart is an inbound, not live play.
It ends at the first of:
  shot   the offence attempts a field goal (fg3, made, t_act);
  tov    the offence turns the ball over;
  foul   the defence commits a team foul followed by free throws (n_ft; shooting or not);
  end    the period ends;
  other  inconsistent sequence (the defence acts as if it had the ball).
Fouls with no free throws (a foul to give) are counted in `sideouts` and the phase continues.

usage: python -m nbaeg.possessions      (reads data/derived/late.parquet, writes possessions.parquet)
"""
import numpy as np
import pandas as pd

from .build import DER

OTHER = {'H': 'A', 'A': 'H'}
OFFENSIVE = {'Offensive', 'Offensive Charge'}
STATE = ['to_full', 'to_short', 'to_perfull', 'to_per', 'to_l3', 'to_l2full', 'to_short_h2', 'tf_per', 'tf_l2']


def timeouts_left(season, period, t, full_game, short_game, per_full, per_all, l3_all, short_h2, l2_full=0):
    """Timeouts a team can still call, under the rules of its season (encoding described in the paper, Section 2).

    2017-18 on: 7 per game, at most 4 in Q4, at most 2 after 3:00 of Q4; 2 per overtime.
    2000-01..2016-17: 6 full per game, at most 3 full in Q4 and at most 2 full in its last 2:00 (the data agree:
    3 full timeouts in the last 2:00 in 30 of 40,978 team-games), plus one 20-second timeout per half;
    2 full and one 20-second timeout per overtime (2016-17 rule book; the Jul 2017 release counts this as
    3 overtime timeouts going to 2).
    """
    if season >= 2017:
        if period == 4:
            r = min(7 - full_game - short_game, 4 - per_all, (2 - l3_all) if t < 180 else 9)
        elif period > 4:
            r = 2 - per_all
        else:
            r = 7 - full_game - short_game
    else:
        short = max(0, 1 - short_h2) if period >= 3 else max(0, 1 - short_game)
        if period == 4:
            r = max(0, min(6 - full_game, 3 - per_full, (2 - l2_full) if t < 120 else 9)) + short
        elif period > 4:                          # 2 full + one 20-second timeout per overtime (2016-17 rules)
            r = max(0, 2 - per_full) + max(0, 1 - (per_all - per_full))
        else:
            r = max(0, 6 - full_game) + short
    return max(0, int(r))


def foul_to_give(period, t, tf_per, tf_l2):
    """1 if a foul by this team would not yet put the opponent on the line (NBA team-foul penalty)."""
    lim = 4 if period <= 4 else 3
    return int(tf_per < lim and not (t <= 120 and tf_l2 >= 1))


def scan(E, i0, i1):
    """Phases within one (game, period) slice E[i0:i1] (list of dicts)."""
    out = []
    last_miss = None
    for i in range(i0, i1):
        e = E[i]
        s = e['side']
        new, how = None, None
        if e['is_fga'] and s in OTHER:
            if e['fg_made']:
                # and-one: the shooter's FT comes at the same clock; the phase starts after it
                andone = any(E[j]['is_ft'] and E[j]['side'] == s and E[j]['t'] == e['t']
                             for j in range(i + 1, min(i + 8, i1)))
                if not andone:
                    new, how = OTHER[s], 'fg'
                last_miss = None
            else:
                last_miss = s
        elif e['is_ft'] and s in OTHER and not e['ft_tech'] and e['ft_k'] == e['ft_n']:
            if e['ft_made']:
                new, how = OTHER[s], 'ft'
                last_miss = None
            else:
                last_miss = s
        elif e['is_reb'] and s in OTHER and last_miss is not None:
            new, how = s, ('oreb' if s == last_miss else 'dreb')
            last_miss = None
        elif e['is_tov'] and s in OTHER:
            new, how = OTHER[s], 'tov'
        if new is not None and e['t'] > 0:
            out.append(classify(E, i, i1, new, how))
    return out


def classify(E, i, i1, O, how):
    e0 = E[i]
    D = OTHER[O]
    rec = dict(i_start=i, O=O, start=how, t0=e0['t'], lead0=(e0['margin'] if O == 'H' else -e0['margin']),
               reb_team=bool(e0['is_reb'] and e0['personId'] > 1e9),
               action='end', t_act=0.0, fg3=np.nan, made=np.nan, n_ft=0, foul_type='', sideouts=0,
               t_side=np.nan, to_O=0, t_to_O=np.nan, to_D=0)
    for j in range(i + 1, i1):
        e = E[j]
        s = e['side']
        if e['is_to']:
            if s == O:
                if not rec['to_O']:
                    rec['t_to_O'] = e['t']
                rec['to_O'] += 1
            elif s == D:
                rec['to_D'] += 1
            continue
        if e['is_fga'] and s == O:
            rec.update(action='shot', t_act=e['t'], fg3=bool(e['fg3']), made=bool(e['fg_made']))
            return rec
        if e['is_tov'] and s == O:
            rec.update(action='tov', t_act=e['t'])
            return rec
        if e['is_foul'] and s == D and e['team_foul']:
            n = 0
            for k in range(j + 1, min(j + 10, i1)):
                f = E[k]
                if f['t'] != e['t']:
                    break
                if f['is_ft'] and f['side'] == O and not f['ft_tech']:
                    n = int(f['ft_n']) if f['ft_n'] == f['ft_n'] else 1
                    break
            if n == 0:
                rec['sideouts'] += 1
                if np.isnan(rec['t_side']):
                    rec['t_side'] = e['t']
                continue
            rec.update(action='foul', t_act=e['t'], n_ft=n, foul_type=e['foul_type'])
            return rec
        if e['is_ft'] and s == O and not e['ft_tech']:      # FT trip with no recorded foul
            rec.update(action='foul', t_act=e['t'], n_ft=int(e['ft_n']) if e['ft_n'] == e['ft_n'] else 1, foul_type='?')
            return rec
        if (e['is_fga'] or e['is_tov']) and s == D:
            rec.update(action='other', t_act=e['t'])
            return rec
        if e['is_foul'] and s == O and e['foul_type'] in OFFENSIVE:
            rec.update(action='tov', t_act=e['t'])
            return rec
    return rec


def main():
    L = pd.read_parquet(DER / 'late.parquet').sort_values(['gameId', 'period', 't', 'seq'],
                                                          ascending=[True, True, False, True])
    L = L.reset_index(drop=True)
    E = L[['side', 't', 'margin', 'is_fga', 'fg_made', 'fg3', 'is_ft', 'ft_k', 'ft_n', 'ft_made', 'ft_tech',
           'is_reb', 'is_tov', 'is_foul', 'team_foul', 'foul_type', 'is_to', 'personId']].to_dict('records')
    key = (L.gameId.astype(np.int64) * 10 + L.period).to_numpy()
    cut = np.flatnonzero(np.diff(key)) + 1
    recs = []
    for a, b in zip(np.r_[0, cut], np.r_[cut, len(L)]):
        recs += scan(E, a, b)
    P = pd.DataFrame(recs)
    ctx = L.loc[P.i_start, ['season', 'gameId', 'period', 'seq', 'home_id', 'away_id'] +
                [f'{c}_{s}' for c in STATE for s in 'HA']].reset_index(drop=True)
    P = pd.concat([P, ctx], axis=1)
    for r in 'OD':
        sd = P['O'] if r == 'O' else P['O'].map(OTHER)
        for c in STATE:
            P[f'{c}_{r}'] = np.where(sd == 'H', P[f'{c}_H'], P[f'{c}_A'])
    P = P.drop(columns=[f'{c}_{s}' for c in STATE for s in 'HA'])
    for r in 'OD':
        P[f'tol_{r}'] = [timeouts_left(*a) for a in zip(
            P.season, P.period, P.t0, P[f'to_full_{r}'], P[f'to_short_{r}'], P[f'to_perfull_{r}'],
            P[f'to_per_{r}'], P[f'to_l3_{r}'], P[f'to_short_h2_{r}'], P[f'to_l2full_{r}'])]
        P[f'ftg_{r}'] = [foul_to_give(*a) for a in zip(P.period, P.t0, P[f'tf_per_{r}'], P[f'tf_l2_{r}'])]
    g = pd.read_parquet(DER / 'games.parquet').set_index('gameId')
    P = P.join(g[['home_final', 'away_final', 'last_period']], on='gameId')
    fm = P.home_final - P.away_final
    P['O_won'] = np.where(P.O == 'H', fm > 0, fm < 0).astype(int)
    P['ot_after'] = (P.last_period > P.period).astype(int)
    P['O_id'] = np.where(P.O == 'H', P.home_id, P.away_id)
    P['D_id'] = np.where(P.O == 'H', P.away_id, P.home_id)
    P['used'] = P.t0 - P.t_act
    P.to_parquet(DER / 'possessions.parquet')
    print(len(P), 'phases')
    print(P.groupby(['start', 'action']).size().unstack(fill_value=0))



def last_fts():
    """Last free throw of every 2- or 3-shot trip in Q4/OT, last 60 s, with what followed.

    Columns: shooter side S, lead before the FT (lead_b), made, t, the next rebound (oreb: the shooter's
    side got it; reb_team: a team rebound, i.e. a dead ball), timeouts left for the shooter's team (tol_S) and the opponent (tol_R), the opponent's
    foul to give (ftg_R), the shooter's season FT% (leave-one-out) and whether the shooter's team won.
    """
    L = pd.read_parquet(DER / 'late.parquet').sort_values(['gameId', 'period', 't', 'seq'],
                                                          ascending=[True, True, False, True]).reset_index(drop=True)
    L = L[L.period >= 4].reset_index(drop=True)
    ft = L[L.is_ft & ~L.ft_tech & (L.ft_k == L.ft_n) & (L.ft_n >= 2) & L.side.isin(['H', 'A'])].copy()
    # next rebound after a miss, within the same period
    reb = L[L.is_reb & L.side.isin(['H', 'A'])][['gameId', 'period', 'seq', 'side', 't', 'personId']]
    # `seq` restarts every season, so the forward match must be made within the game (by='gameId');
    # a rebound in a later period does not belong to this free throw
    # a team rebound (personId = team id) is a dead ball: out of bounds, the team inbounds
    reb = reb.assign(reb_team=(reb.personId > 1e9).astype(float)).drop(columns='personId')
    reb = reb.rename(columns={'side': 'reb_side', 't': 't_reb', 'seq': 'seq_reb', 'period': 'p2'})
    ft = pd.merge_asof(ft.sort_values('seq'), reb.sort_values('seq_reb'), left_on='seq', right_on='seq_reb',
                       by='gameId', direction='forward')
    ok = ft.p2 == ft.period
    ft.loc[~ok, ['reb_side', 't_reb', 'reb_team']] = [None, np.nan, np.nan]
    ft['oreb'] = np.where(ft.reb_side.isna(), np.nan, (ft.reb_side == ft.side).astype(float))
    sgn = np.where(ft.side == 'H', 1, -1)
    ft['lead_b'] = ft.margin_before * sgn
    ft['made'] = ft.ft_made.astype(int)
    S, R = ft.side, ft.side.map(OTHER)
    for r, sd in (('S', S), ('R', R)):
        for c in STATE:
            ft[f'{c}_{r}'] = np.where(sd == 'H', ft[f'{c}_H'], ft[f'{c}_A'])
        ft[f'tol_{r}'] = [timeouts_left(*a) for a in zip(
            ft.season, ft.period, ft.t, ft[f'to_full_{r}'], ft[f'to_short_{r}'], ft[f'to_perfull_{r}'],
            ft[f'to_per_{r}'], ft[f'to_l3_{r}'], ft[f'to_short_h2_{r}'], ft[f'to_l2full_{r}'])]
        ft[f'ftg_{r}'] = [foul_to_give(*a) for a in zip(ft.period, ft.t, ft[f'tf_per_{r}'], ft[f'tf_l2_{r}'])]
    pf = pd.read_parquet(DER / 'player_ft.parquet')
    ft = ft.merge(pf.rename(columns={'sum': 'ftm_s', 'count': 'fta_s'}), on=['personId', 'season'], how='left')
    ft['pct'] = ((ft.ftm_s - ft.made) / (ft.fta_s - 1)).where(ft.fta_s > 20)
    g = pd.read_parquet(DER / 'games.parquet').set_index('gameId')
    ft = ft.join(g[['home_final', 'away_final', 'last_period']], on='gameId')
    fm = ft.home_final - ft.away_final
    ft['S_won'] = np.where(ft.side == 'H', fm > 0, fm < 0).astype(int)
    keep = ['season', 'gameId', 'period', 'seq', 't', 'side', 'personId', 'lead_b', 'made', 'oreb', 't_reb', 'reb_team',
            'tol_S', 'tol_R', 'ftg_S', 'ftg_R', 'pct', 'S_won', 'last_period', 'ft_n']
    ft[keep].to_parquet(DER / 'last_ft.parquet')
    print(len(ft), 'last FTs')


if __name__ == '__main__':
    main()
    last_fts()
