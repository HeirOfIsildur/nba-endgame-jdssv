# NBA end game: decision maps from 26 seasons of play-by-play

Code and pipeline for a JDSSV paper (research_hub paper 20). It estimates the inputs of an exact
end-game model from public NBA play-by-play (2000-01 to 2025-26, regular season), solves the
zero-sum end-game game on a 0.1 s grid, and draws **decision maps**: time × score × timeouts, the
optimal choice and its win-probability margin, bootstrap uncertainty, and observed coach behaviour
before and after the 2017-18 cap on late timeouts.

## Reproduce

Tools: `uv` (Python 3.13), `curl`/network for the download, LaTeX (`latexmk`) for the paper, and Wolfram
(`wolframscript`) for `make check`. `make all` reproduces the statistical results; it reads, but does not rerun,
the committed Mathematica output and platform comparison (`make check`, `scripts/compare_platform.py`). The Makefile
is serial (`.NOTPARALLEL`).

```bash
make env          # .venv from requirements-lock.txt (Python 3.13, uv)
make all          # ~2 min download + build + estimates; bootstrap B=1000 (~1 h on 4 cores; ~10 min as a 50-task cluster array); writes results/ and figures/
make test         # rule-encoding and game invariants
make check        # Mathematica: exact-rational re-solve of the DP + break-even identity (wolframscript; not part of `make all`)
```

`make all` downloads `nbastatsv3_<season>.tar.xz` from
[shufinskiy/nba_data](https://github.com/shufinskiy/nba_data) (checksums in `data/raw/SHA256SUMS`;
about 230 MB compressed, 2.6 GB as CSV). Every number quoted in the paper is regenerated in
`results/headline.txt`.

The committed bootstrap (`results/boot_summary.npz`, `inputs_ci.csv`, `boot_manifest.json`) was computed on
the MFF Chimera cluster (Slurm job 38351867, 50 array tasks × 20 replicates = 1,000, `scripts/chimera_boot_task.sh`;
platform records in `results/platform/`). Replicates are seeded by their index and carry a fingerprint of the
code and input data; `make boot` regenerates any replicate whose fingerprint does not match and the summary
refuses mismatched ones. The same replicate computed on macOS and on Linux agrees numerically to 4.7e-13
(`python scripts/compare_platform.py` recomputes replicates 0, 517, 999 locally, compares them with the cluster's copies deposited in
`results/platform/reference/` and writes `results/platform/local_vs_cluster.json`; the cluster's Python packages are listed in
`results/platform/worker_environment.txt`).
`python scripts/check_numbers.py` lists numbers in the main text that are not found in the deposited results.

A fresh bootstrap from scratch: delete `results/boot/`, then `make boot` (or submit
`scripts/chimera_boot_task.sh` as a 50-task array as described in its header) and `make sensitivity figures headline`.

## Manuscript

`paper/main.tex` is the JDSSV submission (LaTeX v5 template from jdssv.org, unmodified class files);
`make paper` builds `paper/main.pdf`, and `make supplement` writes this repository as a zip archive to
`paper/submission/`.

## Pipeline (`src/nbaeg/`)

| step | module | output |
|---|---|---|
| download | `download.py` | `data/raw/nbastatsv3_<y>.csv` |
| parse; state columns (score, timeouts used, team fouls) | `build.py` | `data/derived/late.parquet`, `games`, `team_season`, `player_ft` |
| possession phases (start type, first terminal action, timeouts left, foul to give); last FTs | `possessions.py` | `possessions.parquet`, `last_ft.parquet` |
| inputs: rushed-shot curves by start type, OREB rates, 3-FT risk, foul and timeout times | `estimate.py` | `results/params_fitted.json`, `curves_fitted.npz` |
| exact game (numba backward induction) | `dp.py` | – |
| decision-map summaries; game-clustered bootstrap of the whole pipeline | `maps.py`, `bootstrap.py` | `maps_*.npz`, `boot/`, `boot_summary.npz`, `inputs_ci.csv` |
| observed behaviour by era; calibration | `observed.py` | `obs_*.csv`, `calib*.csv` |
| figures, headline numbers | `figures.py`, `headline.py` | `figures/fig1..9`, `results/headline.txt` |
| deliberate-miss proxy audit: event linkage, rebound type, timing, transport | `linkage.py` | `dm_linkage.csv`, `dm_timing.csv`, `dm_transport.csv` |
| sensitivity variants (27), rebound-state audit, identification check, held-out calibration | `sensitivity.py` | `sensitivity.csv` (table: `sens_table.tex`), `reb_state_audit.csv`, `ident_check.csv`, `calib_holdout.csv` |
| Mathematica audit inputs | `checks.py` + `checks/dp_check.wl` | `checks/dp_check_out.json` |

## Model in one paragraph

State: seconds left (0.1 s grid, ≤ 24 s), lead of the team with the ball, how its possession starts
(backcourt inbound, frontcourt inbound after a timeout, live rebound or steal, offensive rebound, and two
frontcourt inbounds with a running shot clock after an offensive rebound: FC14 after a side-out, clock reset to 14 s,
and FC14T after a live timeout, clock kept at 14 s minus the timeout's cost),
timeouts left for both teams (0/1/2+), and a foul to give for both teams. The defence may foul (a
side-out with a foul to give, else two FTs, three with an estimated probability when the offence trails
by 3); with probability `p_pre` (0 in the main analysis, 0.1/0.2/1 as sensitivity variants: not identifiable from
play-by-play) the offence beats a committed foul with a shot released before it arrives. The offence may call a timeout and advance, hold, or shoot a 2 or a 3 after any number of tenths.
A missed field goal is kept with an estimated rate; an estimated share of kept (lost) rebounds are dead-ball team
rebounds, restarted as a frontcourt inbound (the opponent's backcourt inbound), and the rebound-start shot curves are
estimated from live rebounds only. A free-throw shooter may miss the last FT on purpose; ordinary and deliberate misses have their own estimated rebound
delays (1.08 s and 0.69 s; delay distributions are supported and used as sensitivity variants); a kept miss is a live offensive rebound or, with an estimated share, a dead-ball team rebound
(frontcourt inbound), and a lost miss is a live defensive rebound or, with an estimated share, a dead-ball team rebound
to the defence (the opponent inbounds in its backcourt). Overtime is worth ½. All inputs are estimated; the
hand-set prototype (research_hub `audits/17-foul-up-three/python/endgame2.py`) is solved on the same
state space for comparison.
