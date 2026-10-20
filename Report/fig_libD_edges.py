"""
fig_libD_edges.py — Response to R1.13
======================================
Alternative Library D phase-space plot showing only the 12 edges of the
perturbation cube: for each edge, two parameters are fixed at their
extreme values (+/-1%) and the third sweeps the full range.

Output: fig_libD_edges.png (does NOT overwrite rv2_lib_D.png)
"""

import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
RESULTS = os.path.join(PROJ_ROOT, 'results')

TARGET_FREQ = 1000.0
PCT = 0.01  # 1%
TOL = 1e-6  # tolerance for matching extreme values

OUT_PATH = os.path.join(SCRIPT_DIR, 'fig_libD_edges.png')

PARAM_COLORS = {'alpha': '#c62828', 'Rb': '#1565c0', 'Cm': '#2e7d32'}
PARAM_LABELS = {'alpha': r'$\alpha$', 'Rb': r'$R_b$', 'Cm': r'$C_m$'}


def main():
    p = os.path.join(RESULTS, 'library_D_rb.npz')
    npz = np.load(p, allow_pickle=False)

    d_alpha = npz['delta_alpha']
    d_Rb = npz['delta_Rb']
    d_Cm = npz['delta_Cm']
    delta_vals = npz['delta_vals']
    freqs = npz['freqs']
    Z_lib = npz['Z'].astype(complex)
    Z_base = npz['Z_baseline'].astype(complex)

    i_tgt = int(np.argmin(np.abs(freqs - TARGET_FREQ)))
    dZ = Z_lib[:, i_tgt] - Z_base[i_tgt]
    dR = dZ.real
    dX = dZ.imag

    # Extreme values: the min and max of delta_vals
    d_min = float(delta_vals.min())
    d_max = float(delta_vals.max())

    params = {
        'alpha': d_alpha,
        'Rb': d_Rb,
        'Cm': d_Cm,
    }
    param_names = ['alpha', 'Rb', 'Cm']

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))

    # For each edge: fix two parameters at extreme, sweep the third
    # 3 sweep parameters x 4 corners of the other two = 12 edges
    for sweep_param in param_names:
        other_params = [p for p in param_names if p != sweep_param]
        color = PARAM_COLORS[sweep_param]
        label_base = PARAM_LABELS[sweep_param]

        for i, (sign_a, sign_b) in enumerate([(d_min, d_min), (d_min, d_max),
                                               (d_max, d_min), (d_max, d_max)]):
            # Mask: other two params at their extreme values
            mask = (np.abs(params[other_params[0]] - sign_a) < TOL) & \
                   (np.abs(params[other_params[1]] - sign_b) < TOL)

            if mask.sum() == 0:
                continue

            sweep_vals = params[sweep_param][mask]
            order = np.argsort(sweep_vals)

            label = None
            if i == 0:
                label = f'{label_base} sweep'

            ax.plot(dR[mask][order], dX[mask][order],
                    color=color, lw=1.2, alpha=0.8,
                    marker='o', ms=3, markerfacecolor='white',
                    markeredgecolor=color, markeredgewidth=0.6,
                    label=label)

    # Also plot the 8 corners as larger markers
    corners_mask = np.zeros(len(d_alpha), dtype=bool)
    for da in [d_min, d_max]:
        for db in [d_min, d_max]:
            for dc in [d_min, d_max]:
                m = (np.abs(d_alpha - da) < TOL) & \
                    (np.abs(d_Rb - db) < TOL) & \
                    (np.abs(d_Cm - dc) < TOL)
                corners_mask |= m

    ax.scatter(dR[corners_mask], dX[corners_mask],
               s=40, c='k', marker='D', zorder=5,
               label=r'Cube corners ($\pm 1\%$ all)')

    ax.axhline(0, color='#aaaaaa', lw=0.5)
    ax.axvline(0, color='#aaaaaa', lw=0.5)
    ax.scatter([0], [0], color='k', s=30, zorder=6, marker='x')

    ax.set_xlabel(r'$\delta R$ [$\Omega$]', fontsize=12)
    ax.set_ylabel(r'$\delta X$ [$\Omega$]', fontsize=12)
    ax.set_title(f'Library D edges at {TARGET_FREQ:.0f} Hz', fontsize=12)
    ax.legend(fontsize=9, loc='lower left', facecolor='none', edgecolor='gray')
    ax.grid(True, alpha=0.2)
    ax.set_aspect('equal', adjustable='datalim')

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=200, bbox_inches='tight')
    print(f'Saved: {OUT_PATH}')
    if 'Agg' not in matplotlib.get_backend():
        plt.show()


if __name__ == '__main__':
    main()
