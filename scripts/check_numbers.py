"""Numbers audit: every decimal number (and every integer >= 30 written with a thousands separator or
above 30) in the main text of paper/main.tex must appear, at the printed precision, in a deposited result
(results/headline.txt, the summary CSVs listed in STRICT, checks/dp_check_out.json, results/palette_check.txt,
results/platform/local_vs_cluster.json).
Prints the numbers that are not found; they need a manual check.

usage: python scripts/check_numbers.py
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
tex = (ROOT / 'paper' / 'main.tex').read_text()
body = tex[tex.index('\\begin{document}'):tex.index('\\section*{Computational details}')]
abstract = tex[tex.index('\\Abstract{'):tex.index('\\Keywords')]
text = abstract + body
text = re.sub(r'(?<!\\)%.*', '', text)
text = re.sub(r'\\(cite[pt]?|ref|label|eqref|includegraphics|path|code|url)\{[^}]*\}', ' ', text)
text = re.sub(r'\[[^\]]*width[^\]]*\]', ' ', text)
text = text.replace('{,}', '')

src = ''
STRICT = ['inputs_ci.csv', 'sensitivity.csv', 'rho_dm_bins.csv', 'dm_linkage.csv', 'dm_timing.csv', 'dm_transport.csv',
          'calib_holdout.csv', 'calib_summary.csv', 'ident_check.csv']
for f in [ROOT / 'results' / c for c in STRICT] + [ROOT / 'results' / 'headline.txt', ROOT / 'results' / 'palette_check.txt',
                                                     ROOT / 'checks' / 'dp_check_out.json',
                                                     ROOT / 'results' / 'platform' / 'local_vs_cluster.json']:
    src += f.read_text() + '\n'
vals = [float(x) for x in re.findall(r'-?\d+\.\d+(?:e-?\d+)?|-?\d+', src)]


def found(tok):
    dec = len(tok.split('.')[1]) if '.' in tok else 0
    x = abs(float(tok))
    for v in vals:
        for y in (v, 100 * v):                    # shares printed as percentages
            if round(abs(y), dec) == x or (dec == 0 and abs(abs(y) - x) <= 0.5):
                return True
    return False


skip = set()
miss = []
for m in re.finditer(r'(?<![\w.^_{])(\d+\.\d+|\d{2,})(?![\w-]*\d{2}\b)', text):
    tok = m.group(1)
    if '.' not in tok and (int(tok) < 30 or 1990 <= int(tok) <= 2030):
        continue                                  # small integers (seconds, counts of variants) and years
    if not found(tok):
        ctx = text[max(0, m.start() - 50):m.end() + 30].replace('\n', ' ')
        miss.append((tok, ctx))
for tok, ctx in miss:
    print(f'{tok:>10s}  ...{ctx}...')
print(f'{len(miss)} numbers not found in deposited results')
