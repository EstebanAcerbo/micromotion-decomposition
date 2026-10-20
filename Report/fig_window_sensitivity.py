"""
fig_window_sensitivity.py — Response to R1.2
=============================================
Generates a figure showing std(dR) as a function of moving-average window
length for all four electrodes, demonstrating that variance plateaus above
the chosen W values.

Output: fig_window_sensitivity.png
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, os.path.join(PROJ_ROOT, 'experimental_signal_analysis'))

from load_experimental import load_dat, window_sensitivity

STEMS = ['eb04', 'eb11', 'eb13', 'eb15']
LABELS = {
    'eb04': 'Seeding',
    'eb11': 'Confluent',
    'eb13': 'Wound healing',
    'eb15': 'Post-healing',
}
COLORS = {
    'eb04': '#4CAF50',
    'eb11': '#2196F3',
    'eb13': '#FF9800',
    'eb15': '#9C27B0',
}
# Group-adaptive W used in the manuscript
W_USED = {
    'eb04': 300,
    'eb11': 900,
    'eb13': 300,
    'eb15': 900,
}

WINDOWS = [15, 30, 60, 120, 180, 300, 600, 900, 1800, 3600]

OUT_PATH = os.path.join(SCRIPT_DIR, 'fig_window_sensitivity.png')


def main():
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))

    for stem in STEMS:
        t, R, X, C, f_meas = load_dat(stem)
        ws = window_sensitivity(R, t, windows_sec=WINDOWS)

        w_arr = np.array([float(w) for w in ws['windows']])
        s_arr = ws['stds']

        ax.plot(w_arr, s_arr, 'o-', color=COLORS[stem], lw=1.5, ms=5,
                label=LABELS[stem])

        # Mark the chosen W with a vertical line
        w_chosen = W_USED[stem]
        idx = np.argmin(np.abs(w_arr - w_chosen))
        ax.plot(w_arr[idx], s_arr[idx], 's', color=COLORS[stem], ms=10,
                mec='k', mew=1.2, zorder=5)

    ax.set_xscale('log')
    ax.set_xlabel('Moving-average window $W$ (s)')
    ax.set_ylabel(r'std($\delta R$)  ($\Omega$)')
    ax.set_title('Detrended signal variance vs.\ window length')

    # Add annotations for the chosen windows
    from matplotlib.lines import Line2D
    legend_handles = ax.get_legend_handles_labels()[0]
    legend_labels = ax.get_legend_handles_labels()[1]
    legend_handles.append(Line2D([0], [0], marker='s', color='gray', ls='',
                                 ms=8, mec='k', mew=1.2))
    legend_labels.append('Chosen $W$')
    ax.legend(legend_handles, legend_labels, fontsize=8)

    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=200, bbox_inches='tight')
    print(f'Saved: {OUT_PATH}')
    plt.show()


if __name__ == '__main__':
    main()
