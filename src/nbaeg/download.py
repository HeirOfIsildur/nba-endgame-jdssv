"""Download shufinskiy/nba_data stats.nba.com v3 play-by-play (regular season) from a pinned commit and
verify every archive against the committed checksum manifest.

usage: python -m nbaeg.download [first_season] [last_season]   (season = start year, 2000 -> 2000-01)
       python -m nbaeg.download --record [first] [last]        (rewrite data/raw/SHA256SUMS; maintainers only)
Files go to data/raw/nbastatsv3_<y>.csv. A checksum mismatch stops the pipeline.
"""
import hashlib
import subprocess
import sys
import tarfile
from pathlib import Path

COMMIT = 'e829d4678be1e075f99e5d41a1c5f97089be446b'   # shufinskiy/nba_data main, committed 17 Jul 2026
URL = 'https://github.com/shufinskiy/nba_data/raw/{commit}/datasets/nbastatsv3_{y}.tar.xz'
RAW = Path(__file__).resolve().parents[2] / 'data' / 'raw'
SUMS = RAW / 'SHA256SUMS'


def expected():
    out = {}
    for line in SUMS.read_text().splitlines():
        h, name = line.split()
        out[name] = h
    return out


def fetch(y: int, record: bool) -> str:
    RAW.mkdir(parents=True, exist_ok=True)
    xz = RAW / f'nbastatsv3_{y}.tar.xz'
    csv = RAW / f'nbastatsv3_{y}.csv'
    if not xz.exists():
        subprocess.run(['curl', '-sSfL', '--retry', '3', '-o', str(xz), URL.format(commit=COMMIT, y=y)], check=True)
    h = hashlib.sha256(xz.read_bytes()).hexdigest()
    if not record:
        want = expected().get(xz.name)
        if want != h:
            raise SystemExit(f'checksum mismatch for {xz.name}: expected {want}, got {h}')
    if not csv.exists():
        with tarfile.open(xz) as t:
            t.extractall(RAW, filter='data')
    return f'{h}  {xz.name}'


def main():
    args = [a for a in sys.argv[1:] if a != '--record']
    record = '--record' in sys.argv
    a = int(args[0]) if args else 2000
    b = int(args[1]) if len(args) > 1 else 2025
    sums = [fetch(y, record) for y in range(a, b + 1)]
    if record:
        SUMS.write_text('\n'.join(sums) + '\n')
    print(f'{len(sums)} archives verified against {SUMS.name} (upstream commit {COMMIT[:10]})')


if __name__ == '__main__':
    main()
