"""Colour-vision check of the figure palette: simulates protanopia, deuteranopia and tritanopia (Machado,
Oliveira & Fernandes 2009, severity 1) and reports the OKLab distance (x100) between colours that must be told
apart, and the contrast ratio of each colour against the white page.
usage: python scripts/check_palette.py      (writes results/palette_check.txt; exits 1 if a pair falls below 8)"""
import itertools
import sys
from pathlib import Path

import numpy as np

PAIRS = {  # colours that must be distinguishable, as used in figures.py
    'eras (blue / orange)': ('#2a78d6', '#eb6834'),
    'start types BC / FC': ('#2a78d6', '#eb6834'), 'start types FC / LIVE': ('#eb6834', '#1baf7a'),
    'start types LIVE / OREB': ('#1baf7a', '#eda100'), 'start types BC / OREB': ('#2a78d6', '#eda100'),
    'start types BC / LIVE': ('#2a78d6', '#1baf7a'), 'start types FC / OREB': ('#eb6834', '#eda100'),
    'diverging poles': ('#104281', '#8c1c1c'), 'diverging mid-arms': ('#2a78d6', '#e34948'),
}
CVD = {
    'normal': np.eye(3),
    'protanopia': np.array([[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998]]),
    'deuteranopia': np.array([[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.011820, 0.042940, 0.968881]]),
    'tritanopia': np.array([[1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602], [0.004733, 0.691367, 0.303900]]),
}


def lin(hexc):
    c = np.array([int(hexc[i:i + 2], 16) for i in (1, 3, 5)]) / 255
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def oklab(rgb):
    M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929], [0.2119034982, 0.6806995451, 0.1073969566],
                   [0.0883024619, 0.2817188376, 0.6299787005]])
    M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468], [1.9779984951, -2.4285922050, 0.4505937099],
                   [0.0259040371, 0.7827717662, -0.8086757660]])
    return M2 @ np.cbrt(np.clip(M1 @ rgb, 0, None))


def contrast(hexc):
    L = 0.2126 * lin(hexc)[0] + 0.7152 * lin(hexc)[1] + 0.0722 * lin(hexc)[2]
    return 1.05 / (L + 0.05)


def main():
    out, worst = [], 1e9
    for name, (a, b) in PAIRS.items():
        row = []
        for v, M in CVD.items():
            dE = 100 * np.linalg.norm(oklab(np.clip(M @ lin(a), 0, 1)) - oklab(np.clip(M @ lin(b), 0, 1)))
            row.append(f'{v} {dE:5.1f}')
            if v != 'normal':
                worst = min(worst, dE)
        out.append(f'{name:28s} ' + ' | '.join(row))
    cols = sorted({c for p in PAIRS.values() for c in p})
    out.append('contrast vs white: ' + ', '.join(f'{c} {contrast(c):.2f}' for c in cols))
    out.append(f'worst simulated-CVD distance: {worst:.1f} (target >= 8; identity is also carried by line style, '
               f'position or labels in every figure)')
    Path('results/palette_check.txt').write_text('\n'.join(out) + '\n')
    print('\n'.join(out))
    sys.exit(0 if worst >= 8 else 1)


if __name__ == '__main__':
    main()
