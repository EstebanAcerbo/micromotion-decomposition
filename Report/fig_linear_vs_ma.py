"""
fig_linear_vs_ma.py — Response to R1.1
======================================
Generates a 1×2 figure comparing linear detrending vs moving-average
detrending for the two non-stationary recordings (wound-healing eb13
and seeding eb04).

Left panel:  raw R(t) with regression line and MA(300 s) overlaid.
Right panel: residuals after each method.

Output: fig_linear_vs_ma.png (in the same directory as this script)
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

# ── Add experimental analysis module to path ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, os.path.join(PROJ_ROOT, 'experimental_signal_analysis'))

from load_experimental import load_dat, _detrend_linear, _detrend_moving_avg

# ── Configuration ──
STEMS = ['eb13', 'eb04']
LABELS = {'eb13': 'Wound healing', 'eb04': 'Seeding'}
COLORS = {'eb13': '#FF9800', 'eb04': '#4CAF50'}
W_SEC = 300  # MA window for both (healing/seeding group)

OUT_PATH = os.path.join(SCRIPT_DIR, 'fig_linear_vs_ma.png')


def _moving_avg_trend(sig, dt, window_sec):
    """Return the MA trend (not the residual)."""
    n_win = max(3, int(round(window_sec / dt)))
    if n_win % 2 == 0:
        n_win += 1
    return uniform_filter1d(sig.astype(float), size=n_win, mode='nearest')


def _linear_trend(sig, t):
    """Return the linear-fit trend."""
    t_n = (t - t.mean()) / (t[-1] - t[0] + 1e-12)
    coeffs = np.polyfit(t_n, sig, 1)
    return np.polyval(coeffs, t_n)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax_raw = axes[0]
    ax_res = axes[1]

    for stem in STEMS:
        t, R, X, C, f_meas = load_dat(stem)
        dt = float(np.median(np.diff(t)))
        t_h = (t - t[0]) / 3600.0  # hours

        # Trends
        trend_lin = _linear_trend(R, t)
        trend_ma = _moving_avg_trend(R, dt, W_SEC)

        # Residuals
        res_lin = R - trend_lin
        res_ma = R - trend_ma

        color = COLORS[stem]
        label = LABELS[stem]

        # Left panel: raw R(t) + trends
        ax_raw.plot(t_h, R, color=color, alpha=0.4, lw=0.5)
        ax_raw.plot(t_h, trend_lin, color=color, ls='--', lw=1.5,
                    label=f'{label} — linear fit')
        ax_raw.plot(t_h, trend_ma, color=color, ls='-', lw=1.5,
                    label=f'{label} — MA({W_SEC} s)')

        # Right panel: residuals
        ax_res.plot(t_h, res_lin, color=color, alpha=0.5, lw=0.4,
                    label=f'{label} — linear')
        ax_res.plot(t_h, res_ma, color=color, alpha=0.8, lw=0.4,
                    ls='-')

    # ── Formatting ──
    ax_raw.set_xlabel('Time (h)')
    ax_raw.set_ylabel(r'$R\;(\Omega)$')
    ax_raw.set_title('(a)  Raw signal and trend estimates')
    ax_raw.legend(fontsize=7, loc='lower right')

    ax_res.set_xlabel('Time (h)')
    ax_res.set_ylabel(r'$\delta R\;(\Omega)$')
    ax_res.set_title('(b)  Detrended residuals')
    from matplotlib.lines import Line2D
    handles = []
    for stem in STEMS:
        handles.append(Line2D([0], [0], color=COLORS[stem], ls='--', lw=1,
                              label=f'{LABELS[stem]} — linear'))
        handles.append(Line2D([0], [0], color=COLORS[stem], ls='-', lw=1,
                              label=f'{LABELS[stem]} — MA({W_SEC} s)'))
    ax_res.legend(handles=handles, fontsize=7, loc='upper right')

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=200, bbox_inches='tight')
    print(f'Saved: {OUT_PATH}')
    plt.show()


if __name__ == '__main__':
    main()
