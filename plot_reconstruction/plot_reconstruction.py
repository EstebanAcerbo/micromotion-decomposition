"""
plot_reconstruction.py — Compact reconstruction comparison figure.

Usage
-----
    python plot_reconstruction.py eb11
    python plot_reconstruction.py eb11 --output results/my_fig.png --show

Layout  (1 row × 2 cols)
-------------------------
  Left   δR — exp (black/solid, z=1), Greedy (red/dashed, z=2),
               Viterbi (blue/dotted, z=3)
         Inset (axins): δR residuals — Greedy + Viterbi, zero line
  Right  δX — same styling and same inset logic

Inset position is set via INSET_RECT (x0, y0, w, h) in axes-fraction coords.
Viterbi is silently skipped when reconstruct_v4_{stem}.npz is absent.
"""

import os
import sys
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS    = os.path.join(SCRIPT_DIR, '..', 'results')

# ── Visual style ───────────────────────────────────────────────────────────
C_EXP     = 'black'
C_GREEDY  = 'tab:red'
C_VITERBI = 'tab:blue'

LS_EXP     = '-'
LS_GREEDY  = '--'
LS_VITERBI = ':'

LW_EXP     = 0.7
LW_REC     = 0.8
LW_RESID   = 0.6

ALPHA_EXP   = 0.75
ALPHA_GRE   = 0.5
ALPHA_VIT   = 0.4

# Inset position: (x0, y0, width, height) in axes-fraction coordinates.
# Expanded to fill from current left edge to the top-right corner.
# Fine-tune these values as needed:
#   x0=0.42  left edge of inset (42% from left)
#   y0=0.50  bottom edge of inset (50% from bottom)
#   w=0.56   right edge reaches ~98% of axes width
#   h=0.47   top edge reaches ~97% of axes height
INSET_RECT = (0.62, 0.65, 0.36, 0.32)


# ── Helpers ────────────────────────────────────────────────────────────────

def _load_npz(stem, version):
    path = os.path.join(RESULTS, f'reconstruct_{version}_{stem}.npz')
    if not os.path.isfile(path):
        print(f'[WARN] {path} not found — {version} will be skipped.')
        return None
    return np.load(path, allow_pickle=False)


def _scalar(npz, key):
    if npz is None or key not in npz.files:
        return float('nan')
    return float(np.asarray(npz[key]).flat[0])


def _rmse_label(npz, key):
    v = _scalar(npz, key)
    return f'  RMSE={v:.2f} Ω' if np.isfinite(v) else ''


def _add_inset(ax, t_min, resid_v3, resid_v4):
    """
    Add a residuals inset to *ax* at INSET_RECT.
    resid_v3 / resid_v4 may be None when the corresponding NPZ is absent.
    """
    axins = ax.inset_axes(INSET_RECT)

    if resid_v3 is not None:
        axins.plot(t_min, resid_v3,
                   color=C_GREEDY, lw=LW_RESID, ls=LS_GREEDY,
                   alpha=ALPHA_GRE+.1, label='Greedy resid.')
    if resid_v4 is not None:
        axins.plot(t_min, resid_v4,
                   color=C_VITERBI, lw=LW_RESID, ls=LS_VITERBI,
                   alpha=ALPHA_VIT+.1, label='Viterbi resid.')

    axins.axhline(0, color='#aaaaaa', lw=0.5, zorder=0)
    axins.tick_params(labelsize=8)
    axins.grid(True, alpha=0.15)
    axins.set_title('Residuals', fontsize=8, pad=2)
    axins.legend(fontsize=7.5, loc='upper right', framealpha=0.80)

    return axins


# ── Main plot ──────────────────────────────────────────────────────────────

def plot_reconstruction(stem, output=None, show=False):
    v3 = _load_npz(stem, 'v3')
    v4 = _load_npz(stem, 'v4')

    if v3 is None and v4 is None:
        print(f'[ERROR] No reconstruction files found for stem {stem!r}. Aborting.')
        sys.exit(1)

    ref    = v3 if v3 is not None else v4
    t_min  = ref['t'] / 60.0
    t_min  = t_min - t_min[0]  # normalize to start at zero
    dR_exp = ref['dR_exp'].astype(float)
    dX_exp = ref['dX_exp'].astype(float)

    # 3-column layout: δR | δX | residuals (top δR, bottom δX)
    fig = plt.figure(figsize=(16, 5))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.45],
                          hspace=0.35, wspace=0.30)

    ax_dR    = fig.add_subplot(gs[:, 0])   # full height
    ax_dX    = fig.add_subplot(gs[:, 1])   # full height
    ax_res_R = fig.add_subplot(gs[0, 2])   # top-right
    ax_res_X = fig.add_subplot(gs[1, 2])   # bottom-right

    # ── δR panel ───────────────────────────────────────────────────────────
    ax_dR.plot(t_min, dR_exp,
               color=C_EXP, lw=LW_EXP, ls=LS_EXP, alpha=ALPHA_EXP,
               label='Experimental', zorder=1)
    if v3 is not None:
        ax_dR.plot(t_min, v3['dR_rec'].astype(float),
                   color=C_GREEDY, lw=LW_REC, ls=LS_GREEDY, alpha=ALPHA_GRE,
                   label='Greedy' + _rmse_label(v3, 'rmse_R'), zorder=2)
    if v4 is not None:
        ax_dR.plot(t_min, v4['dR_rec'].astype(float),
                   color=C_VITERBI, lw=LW_REC, ls=LS_VITERBI, alpha=ALPHA_VIT,
                   label='Viterbi' + _rmse_label(v4, 'rmse_R'), zorder=3)

    ax_dR.set_xlabel('Time [min]', fontsize=11)
    ax_dR.set_ylabel(r'$\delta R$ [Ω]', fontsize=11)
    ax_dR.legend(fontsize=9, loc='best', ncol=1, framealpha=0.80)
    ax_dR.tick_params(labelsize=9)
    ax_dR.grid(True, alpha=0.18)

    # ── δX panel ───────────────────────────────────────────────────────────
    ax_dX.plot(t_min, dX_exp,
               color=C_EXP, lw=LW_EXP, ls=LS_EXP, alpha=ALPHA_EXP,
               label='Experimental', zorder=1)
    if v3 is not None:
        ax_dX.plot(t_min, v3['dX_rec'].astype(float),
                   color=C_GREEDY, lw=LW_REC, ls=LS_GREEDY, alpha=ALPHA_GRE,
                   label='Greedy' + _rmse_label(v3, 'rmse_X'), zorder=2)
    if v4 is not None:
        ax_dX.plot(t_min, v4['dX_rec'].astype(float),
                   color=C_VITERBI, lw=LW_REC, ls=LS_VITERBI, alpha=ALPHA_VIT,
                   label='Viterbi' + _rmse_label(v4, 'rmse_X'), zorder=3)

    ax_dX.set_xlabel('Time [min]', fontsize=11)
    ax_dX.set_ylabel(r'$\delta X$ [Ω]', fontsize=11)
    ax_dX.legend(fontsize=9, loc='best', ncol=1, framealpha=0.80)
    ax_dX.tick_params(labelsize=9)
    ax_dX.grid(True, alpha=0.18)

    # ── Residuals panels (right column) ────────────────────────────────────
    for ax_res, comp, ylabel, resid_key in [
            (ax_res_R, r'$\delta R$', r'$\delta R$ [$\Omega$]', 'dR_resid'),
            (ax_res_X, r'$\delta X$', r'$\delta X$ [$\Omega$]', 'dX_resid')]:
        if v3 is not None:
            ax_res.plot(t_min, v3[resid_key].astype(float),
                        color=C_GREEDY, lw=LW_RESID, ls=LS_GREEDY,
                        alpha=ALPHA_GRE+.1, label='Greedy resid.')
        if v4 is not None:
            ax_res.plot(t_min, v4[resid_key].astype(float),
                        color=C_VITERBI, lw=LW_RESID, ls=LS_VITERBI,
                        alpha=ALPHA_VIT+.1, label='Viterbi resid.')
        ax_res.axhline(0, color='#aaaaaa', lw=0.5, zorder=0)
        ax_res.set_ylabel(ylabel, fontsize=9)
        ax_res.tick_params(labelsize=8)
        ax_res.grid(True, alpha=0.15)
        ax_res.set_title(f'{comp} residuals', fontsize=9, pad=2)
        ax_res.legend(fontsize=7.5, loc='upper right', framealpha=0.80)

    ax_res_X.set_xlabel('Time [min]', fontsize=9)

    # ── Suptitle ───────────────────────────────────────────────────────────
    rmse_parts = []
    if v3 is not None:
        rmse_parts.append(f"Greedy total RMSE={_scalar(v3, 'rmse_total'):.2f} Ω")
    if v4 is not None:
        rmse_parts.append(f"Viterbi total RMSE={_scalar(v4, 'rmse_total'):.2f} Ω")

    fig.suptitle(
        ' | '.join(rmse_parts) if rmse_parts else '',
        fontsize=13,
    )

    # ── Save ───────────────────────────────────────────────────────────────
    if output is None:
        output = os.path.join(RESULTS, f'reconstruction_comparison_{stem}.png')

    os.makedirs(os.path.dirname(output), exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches='tight')
    print(f'Saved: {output}')

    if show:
        matplotlib.use('TkAgg')
        plt.show()
    plt.close(fig)


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Compact reconstruction comparison — exp vs Greedy vs Viterbi.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('stem', help='e.g. eb11, eb04, eb13, eb15')
    parser.add_argument('--output', default=None,
                        help='Output PNG (default: results/reconstruction_comparison_{stem}.png)')
    parser.add_argument('--show', action='store_true',
                        help='Display figure interactively after saving.')
    args = parser.parse_args()
    plot_reconstruction(args.stem, output=args.output, show=args.show)


if __name__ == '__main__':
    main()
