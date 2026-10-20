"""
generate_fractal_table.py

Run windowed_fractal_analysis for all 4 electrodes × 2 methods (Higuchi, R/S)
× 2 signals (dR, dX), save results/fractal_results.npz, and print a LaTeX
booktabs table.

Column structure:
    Signal (δR / δX) → Algorithm (Higuchi / R/S) → Sub (Single / Fast / Slow)

Format: compact $1.87 (3)$ — std's leading significant digit in parentheses
at the same decimal precision as the mean.
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'experimental_signal_analysis'))
from fractal_dim import windowed_fractal_analysis

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load experimental data
npz = np.load(os.path.join(RESULTS_DIR, 'phase1_experimental.npz'), allow_pickle=False)


def get_arr(npz, stem, key):
    for k in npz.files:
        if k == f'{stem}_{key}' or k == f'{stem}/{key}':
            return npz[k]
    return None


FOCUS_STEMS = ['eb11', 'eb04', 'eb13', 'eb15']
METHODS     = ['higuchi', 'rs']
SIGNALS     = ['dR', 'dX']
ROW_ORDER   = ['eb04', 'eb11', 'eb13', 'eb15']

# ── Run analysis ───────────────────────────────────────────────────────────────
results = {}

for stem in FOCUS_STEMS:
    t = get_arr(npz, stem, 't')
    if t is None:
        print(f'[WARN] {stem}: time array not found — skipping.')
        continue
    dt = float(np.median(np.diff(t)))

    for sig_key in SIGNALS:
        sig = get_arr(npz, stem, sig_key)
        if sig is None:
            print(f'[WARN] {stem}/{sig_key}: not found — skipping.')
            continue
        for meth in METHODS:
            print(f'  Running {stem}/{sig_key}/{meth}...', end=' ', flush=True)
            r = windowed_fractal_analysis(sig, dt, meth)
            results[(stem, meth, sig_key)] = {
                'df_single_mean': r['df_single_mean'],
                'df_single_std':  r['df_single_std'],
                'df_fast_mean':   r['df_fast_mean'],
                'df_fast_std':    r['df_fast_std'],
                'df_slow_mean':   r['df_slow_mean'],
                'df_slow_std':    r['df_slow_std'],
            }
            print(f"df_single={r['df_single_mean']:.4f}+/-{r['df_single_std']:.4f}")

# ── Save npz ───────────────────────────────────────────────────────────────────
save_dict = {}
for (stem, meth, sig_key), v in results.items():
    for qty, val in v.items():
        save_dict[f'{stem}_{meth}_{sig_key}_{qty}'] = np.array(val)
np.savez(os.path.join(RESULTS_DIR, 'fractal_results.npz'), **save_dict)
print(f'\nSaved: {os.path.join(RESULTS_DIR, "fractal_results.npz")}')

# ── LaTeX table ────────────────────────────────────────────────────────────────
# Each method: (key, label, [sub-column names])
# Higuchi: Single/Fast/Slow  — R/S: Single only (no breakpoint analysis)
METH_INFO = [
    ('higuchi', 'Higuchi', ['Single', 'Fast', 'Slow']),
    ('rs',      'R/S',     ['Single']),
]
SIG_INFO  = [('dR', r'$\delta R$'), ('dX', r'$\delta X$')]
ROW_LABELS = {
    'eb04': 'Seeding',
    'eb11': 'Confluent',
    'eb13': 'Wound healing',
    'eb15': 'Post-healing',
}

cols_per_sig = sum(len(subs) for _, _, subs in METH_INFO)  # 3 + 1 = 4
n_data       = len(SIG_INFO) * cols_per_sig                 # 2 × 4 = 8

NL = '\n'


def fmt_cell(mean, std):
    """Compact LaTeX: $1.87 (3)$ — std's leading digit in parens."""
    if not np.isfinite(mean) or not np.isfinite(std):
        return '---'
    if std == 0:
        return f'${mean:.3f}$'
    dec       = -int(math.floor(math.log10(abs(std))))
    mean_r    = round(mean, dec)
    std_r     = round(std,  dec)
    std_digit = int(round(std_r * 10 ** dec))
    return f'${mean_r:.{dec}f} ({std_digit})$'


lines = []
lines.append(r'\begin{table}[htbp]')
lines.append(r'\centering')
lines.append(
    r'\caption{Fractal dimension estimates ($\delta R$ and $\delta X$). '
    r'Values are mean with std (1 sig.\ digit) in parentheses across analysis windows. '
    r'Higuchi Single/Fast/Slow: full-range and bi-segment slopes at forced breakpoint. '
    r'R/S: single-slope only (no breakpoint analysis).}'
)
lines.append(r'\label{tab:fractal_dim}')
lines.append(r'\scriptsize')
lines.append(r'\begin{tabular}{l' + 'c' * n_data + '}')
lines.append(r'\toprule')

# Row 1: signal groupings
parts = ['']
for _, sig_lbl in SIG_INFO:
    parts.append(r'\multicolumn{' + str(cols_per_sig) + r'}{c}{' + sig_lbl + '}')
lines.append(' & '.join(parts) + r' \\')

# cmidrule for signal groups
sig_cmidrules = []
for si in range(len(SIG_INFO)):
    start = 2 + si * cols_per_sig
    end   = start + cols_per_sig - 1
    sig_cmidrules.append(r'\cmidrule(lr){' + f'{start}-{end}' + '}')
lines.append(' '.join(sig_cmidrules))

# Row 2: algorithm groupings (asymmetric spans)
parts2 = ['']
for _ in SIG_INFO:
    for _, meth_lbl, subs in METH_INFO:
        n = len(subs)
        if n == 1:
            parts2.append(meth_lbl)
        else:
            parts2.append(r'\multicolumn{' + str(n) + r'}{c}{' + meth_lbl + '}')
lines.append(' & '.join(parts2) + r' \\')

# cmidrule for algorithm groups (only multi-column ones)
alg_cmidrules = []
col_cursor = 2
for _ in SIG_INFO:
    for _, _, subs in METH_INFO:
        n = len(subs)
        if n > 1:
            alg_cmidrules.append(
                r'\cmidrule(lr){' + f'{col_cursor}-{col_cursor + n - 1}' + '}')
        col_cursor += n
if alg_cmidrules:
    lines.append(' '.join(alg_cmidrules))

# Row 3: individual sub-column labels
parts3 = ['']
for _ in SIG_INFO:
    for _, _, subs in METH_INFO:
        parts3 += subs
lines.append(' & '.join(parts3) + r' \\')
lines.append(r'\midrule')

# Data rows
for stem in ROW_ORDER:
    row_parts = [ROW_LABELS[stem]]
    for sig_key, _ in SIG_INFO:
        for meth_key, _, subs in METH_INFO:
            key = (stem, meth_key, sig_key)
            if key not in results:
                row_parts += ['---'] * len(subs)
                continue
            v = results[key]
            for sub in subs:
                if sub == 'Single':
                    row_parts.append(fmt_cell(v['df_single_mean'], v['df_single_std']))
                elif sub == 'Fast':
                    row_parts.append(fmt_cell(v['df_fast_mean'],   v['df_fast_std']))
                elif sub == 'Slow':
                    row_parts.append(fmt_cell(v['df_slow_mean'],   v['df_slow_std']))
    lines.append(' & '.join(row_parts) + r' \\')

lines.append(r'\bottomrule')
lines.append(r'\end{tabular}')
lines.append(r'\end{table}')

latex = NL.join(lines)
print()
print('=' * 70)
print('LaTeX table:')
print('=' * 70)
print(latex)
