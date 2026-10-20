"""
fig_window_comparison.py — Response to R1.2
============================================
Overlay detrended dR(t) for eb11 (confluent) at W=60s, W=300s, W=900s
on a single panel. Smaller windows suppress slow content, keeping the
signal closer to zero; larger windows preserve minute-regime fluctuations.

Output: fig_window_comparison.png
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, os.path.join(PROJ_ROOT, 'experimental_signal_analysis'))

from load_experimental import load_dat, _detrend_moving_avg

STEM = 'eb11'
WINDOWS = [60, 300, 900]
WINDOW_COLORS = {60: '#E53935', 300: '#FB8C00', 900: '#1E88E5'}

OUT_PATH = os.path.join(SCRIPT_DIR, 'fig_window_comparison.png')


def main():
    t, R, X, C, f_meas = load_dat(STEM)
    dt = float(np.median(np.diff(t)))
    t_h = (t - t[0]) / 3600.0

    # Show a ~30 min window so minute-regime structure is visible
    t_center = 2.0  # hours (internal, not shown)
    t_half = 0.25   # +/- 15 min
    mask = (t_h >= t_center - t_half) & (t_h <= t_center + t_half)
    t_plot = t_h[mask] - t_h[mask][0]  # normalize to start at 0

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))

    # Raw signal
    ax.plot(t_plot, R[mask], color='k', lw=0.5, alpha=0.4, label='Raw $R(t)$')

    # Moving-average trends at different window sizes
    from scipy.ndimage import uniform_filter1d
    for w in WINDOWS:
        n_win = max(3, int(round(float(w) / dt)))
        if n_win % 2 == 0:
            n_win += 1
        trend = uniform_filter1d(R.astype(float), size=n_win, mode='nearest')
        ax.plot(t_plot, trend[mask], color=WINDOW_COLORS[w], lw=2,
                alpha=0.9, label=f'MA($W = {w}$ s)')

    ax.set_xlabel('Time (h)')
    ax.set_ylabel(r'$R\;(\Omega)$')
    ax.set_title('Confluent recording — moving-average trends at different window sizes')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=200, bbox_inches='tight')
    print(f'Saved: {OUT_PATH}')
    plt.show()


if __name__ == '__main__':
    main()
