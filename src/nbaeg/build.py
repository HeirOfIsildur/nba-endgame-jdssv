"""Parse stats.nba.com v3 play-by-play (shufinskiy/nba_data) into analysis tables.

usage: python -m nbaeg.build [first] [last]      (season = start year; default 2000..2025)

Writes to data/derived/:
  late.parquet        every event in the last 60 s of every period, with state columns
                      (score, timeouts used, team fouls) for both sides;
  games.parquet       one row per game: teams, final score, last period;
  team_season.parquet team-season FG2/FG3/FT makes and attempts (whole season);
  player_ft.parquet   player-season FT makes and attempts.
"""
import re
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW, DER = ROOT / 'data' / 'raw', ROOT / 'data' / 'derived'
LATE = 60.0
COLS = ['gameId', 'actionNumber', 'clock', 'period', 'teamId', 'personId', 'location', 'description',
        'actionType', 'subType', 'shotValue', 'scoreHome', 'scoreAway']
# foul subtypes that are team fouls (count toward the penalty); offensive fouls and technicals are not
TEAM_FOUL = {'Personal', 'Shooting', 'Loose Ball', 'Personal Take', 'Transition Take', 'Away From Play',
             'Inbound', 'Clear Path', 'Flagrant Type 1', 'Flagrant Type 2', 'Personal Block',
             'Shooting Block', 'Elbow', 'Punching'}
RX_TO = re.compile(r'\((?:Full|Reg\.)\s*(\d+)\s+Short\s+(\d+)\)')


def season_frame(y: int) -> pd.DataFrame:
    d = pd.read_csv(RAW / f'nbastatsv3_{y}.csv', usecols=lambda c: c in COLS, low_memory=False)
    d = d[d.actionType.notna() & (d.gameId.astype(str).str.zfill(10).str[2] == '2')].copy()
    m = d.clock.str.extract(r'PT(\d+)M([\d.]+)S').astype(float)
    d['t'] = m[0] * 60 + m[1]
    d = d.sort_values(['gameId', 'period', 't', 'actionNumber'], ascending=[True, True, False, True])
    d['seq'] = np.arange(len(d))
    d['side'] = d.location.map({'h': 'H', 'v': 'A'}).fillna('')
    d['desc'] = d.description.fillna('')
    zero = pd.to_numeric(d.scoreHome, errors='coerce').fillna(0).eq(0) & \
        pd.to_numeric(d.scoreAway, errors='coerce').fillna(0).eq(0)    # old seasons write 0-0 on non-scoring rows
    for c in ('scoreHome', 'scoreAway'):
        d[c] = pd.to_numeric(d[c], errors='coerce').mask(zero)
        d[c] = d.groupby('gameId')[c].ffill().fillna(0)
    d['margin'] = d.scoreHome - d.scoreAway                       # home lead after the event
    d['margin_before'] = d.groupby('gameId').margin.shift(1).fillna(0)
    d['season'] = y
    at, st = d.actionType, d.subType.fillna('')
    d['is_fga'] = at.isin(['Made Shot', 'Missed Shot', 'Heave'])
    d['fg_made'] = at.eq('Made Shot')
    sv = pd.to_numeric(d['shotValue'], errors='coerce') if 'shotValue' in d else pd.Series(np.nan, index=d.index)
    d['fg3'] = d.is_fga & (sv.eq(3) | d.desc.str.contains('3PT') | at.eq('Heave'))
    d['is_ft'] = at.eq('Free Throw')
    ftk = st.str.extract(r'(\d) of (\d)$').astype(float)
    d['ft_k'], d['ft_n'] = ftk[0], ftk[1]
    d['ft_tech'] = d.is_ft & st.str.contains('Technical')
    d['ft_made'] = d.is_ft & ~d.desc.str.startswith('MISS')
    d['is_reb'] = at.eq('Rebound')
    d['is_tov'] = at.eq('Turnover')
    d['is_foul'] = at.eq('Foul')
    d['foul_type'] = np.where(d.is_foul, st, '')
    d['team_foul'] = d.is_foul & st.isin(TEAM_FOUL)
    d['is_to'] = at.eq('Timeout') & st.isin(['Regular', 'Short']) & d.side.ne('')
    d['to_short'] = d.is_to & st.eq('Short')
    d['is_viol'] = at.eq('Violation')
    d['is_jump'] = at.eq('Jump Ball')
    return d


def add_state(d: pd.DataFrame) -> pd.DataFrame:
    """Cumulative timeouts and team fouls per side, as of each event (inclusive)."""
    g, gp = d.gameId, [d.gameId, d.period]
    for s in 'HA':
        mine = d.side.eq(s)
        to = d.is_to & mine
        sh = d.to_short & mine
        # timeouts: running count, overridden upward by the official counter in the description
        par = d.desc.where(to).str.extract(RX_TO).astype(float)
        cnt_full = (to & ~sh).astype(int).groupby(g).cumsum()
        cnt_short = sh.astype(int).groupby(g).cumsum()
        pf = par[0].groupby(g).ffill().fillna(0)
        ps = par[1].groupby(g).ffill().fillna(0)
        d[f'to_full_{s}'] = np.maximum(cnt_full, pf).groupby(g).cummax()
        d[f'to_short_{s}'] = np.maximum(cnt_short, ps).groupby(g).cummax()
        d[f'to_per_{s}'] = to.astype(int).groupby(gp).cumsum()
        d[f'to_perfull_{s}'] = (to & ~sh).astype(int).groupby(gp).cumsum()
        d[f'to_l3_{s}'] = (to & (d.t < 180)).astype(int).groupby(gp).cumsum()
        d[f'to_l2full_{s}'] = (to & ~sh & (d.t < 120)).astype(int).groupby(gp).cumsum()
        d[f'to_short_h2_{s}'] = (sh & (d.period >= 3)).astype(int).groupby(g).cumsum()
        tf = (d.team_foul & mine).astype(int)
        d[f'tf_per_{s}'] = tf.groupby(gp).cumsum()
        d[f'tf_l2_{s}'] = (tf * (d.t <= 120)).groupby(gp).cumsum()
    return d


def one(y: int):
    d = add_state(season_frame(y))
    tid = d[d.teamId.fillna(0) > 0].groupby(['gameId', 'side']).teamId.agg(lambda s: s.mode().iat[0]).unstack()
    games = d.groupby('gameId').agg(home_final=('scoreHome', 'last'), away_final=('scoreAway', 'last'),
                                    last_period=('period', 'max'))
    games['home_id'], games['away_id'], games['season'] = tid.get('H'), tid.get('A'), y
    sh = d[(d.is_fga | (d.is_ft & ~d.ft_tech)) & d.side.ne('')].join(games[['home_id', 'away_id']], on='gameId')
    sh['team'] = np.where(sh.side == 'H', sh.home_id, sh.away_id)
    sh['kind'] = np.where(sh.is_ft, 'ft', np.where(sh.fg3, 'fg3', 'fg2'))
    sh['made'] = np.where(sh.is_ft, sh.ft_made, sh.fg_made)
    ts = sh.groupby(['team', 'kind']).made.agg(['sum', 'count']).unstack()
    ts.columns = [f'{a}_{b}' for a, b in ts.columns]
    ts['season'] = y
    pft = sh[sh.is_ft].groupby('personId').made.agg(['sum', 'count'])
    pft['season'] = y
    late = d[d.t <= LATE].drop(columns=['clock', 'description', 'location', 'scoreHome', 'scoreAway'])
    late = late.join(games[['home_id', 'away_id']], on='gameId')
    print(y, len(d), len(late), len(games), flush=True)
    return late, games.reset_index(), ts.reset_index(), pft.reset_index()


def main():
    a = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    b = int(sys.argv[2]) if len(sys.argv) > 2 else 2025
    DER.mkdir(parents=True, exist_ok=True)
    with Pool(4) as p:
        res = p.map(one, range(a, b + 1))
    for i, name in enumerate(['late', 'games', 'team_season', 'player_ft']):
        pd.concat([r[i] for r in res], ignore_index=True).to_parquet(DER / f'{name}.parquet')


if __name__ == '__main__':
    main()
