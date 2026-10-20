"""
plot_mode_trajectories.py
=========================
State-sequence figures for ECIS micromotion mode analysis.

Per stem (auto-discovered from results/reconstruct_v3_*.npz):
  mode_states_{stem}_v4.png  -- distribution + zoom (4 rows x 3 columns)

Columns: distribution (full recording) | Greedy zoom | Viterbi zoom
Rows:    A_shift, B_rb, D_rb, C_brickwall
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

# ────────────────────────────── CONFIG ───────────────────────────────────────

RESULTS_DIR = Path("results")
ZOOM_HALF_S = 2.5 * 60.0     # 5-min total window, half-width in seconds
DPI         = 200

# D_rb parameters: (array key, color, alpha)
_D_PARAMS = [
    ("da_seq",  "steelblue", 0.85),
    ("dRb_seq", "red",       0.6),
    ("dCm_seq", "green",     0.3),
]

_STATE_ROWS = [
    ("B_rb",        "B_rb: K_Rb and lw"),
    ("D_rb",        "D_rb: param. perturbations"),
    ("A_shift",     "A_shift: displacement"),
    ("C_brickwall", "C_brickwall: brick-wall offset"),
]


# ────────────────────────────── ZOOM LOGIC ───────────────────────────────────

def zoom_window(t_s: np.ndarray, sig: np.ndarray) -> tuple:
    """ZOOM_HALF_S window centred on peak |sig|, clamped to array bounds."""
    i_peak = int(np.argmax(np.abs(sig)))
    t_peak = t_s[i_peak]
    i_lo   = max(0,             int(np.searchsorted(t_s, t_peak - ZOOM_HALF_S)))
    i_hi   = min(len(t_s) - 1, int(np.searchsorted(t_s, t_peak + ZOOM_HALF_S)))
    return i_lo, i_hi


# ───────────────────────── ZOOM PANEL HELPERS ────────────────────────────────

def _zoom_A(ax, t_min, rec, i_lo, i_hi):
    t_z = t_min[i_lo:i_hi+1]
    ax.plot(t_z, rec["dx_um"][i_lo:i_hi+1], color="steelblue", lw=1.0,
            alpha=0.85, label="dx")
    ax.plot(t_z, rec["dy_um"][i_lo:i_hi+1], color="coral", lw=1.0,
            alpha=0.85, label="dy")
    ax.set_ylabel(r"displacement [$\mu$m]", fontsize=11)
    ax.legend(fontsize=10, loc="upper right", framealpha=0.7)


def _zoom_B(ax, t_min, rec, i_lo, i_hi):
    t_z = t_min[i_lo:i_hi+1]
    ax.step(t_z, rec["k_rb_seq"][i_lo:i_hi+1], color="red", lw=1.0,
            where="mid", alpha=0.85)
    ax.set_ylabel(r"$K_{Rb}$", fontsize=14, color="red")
    ax.tick_params(axis="y", labelsize=12, colors="red")
    ax.spines["left"].set_edgecolor("red")
    ax_tw = ax.twinx()
    ax_tw.step(t_z, rec["lw_seq"][i_lo:i_hi+1].astype(float),
               color="navy", lw=1.0, where="mid", alpha=0.85)
    ax_tw.set_ylabel("lw [px]", fontsize=12, color="navy")
    ax_tw.tick_params(axis="y", labelsize=11, colors="navy")
    ax_tw.spines["right"].set_edgecolor("navy")


def _zoom_D(ax, t_min, rec, i_lo, i_hi):
    t_z = t_min[i_lo:i_hi+1]
    ax.axhline(0, color="0.65", lw=0.35)
    for arr, color, alpha in _D_PARAMS:
        ax.step(t_z, rec[arr][i_lo:i_hi+1] * 100, color=color, lw=1.0,
                where="mid", alpha=alpha)
    ax.set_ylabel("Perturbation [%]", fontsize=11)


def _zoom_C(ax, t_min, rec, i_lo, i_hi):
    t_z = t_min[i_lo:i_hi+1]
    ax.step(t_z, rec["c_idx_seq"][i_lo:i_hi+1], color="purple", lw=1.0,
            where="mid", alpha=0.85)
    ax.set_ylabel("offset [px]", fontsize=11)


_ZOOM_FN = {
    "A_shift":     _zoom_A,
    "B_rb":        _zoom_B,
    "D_rb":        _zoom_D,
    "C_brickwall": _zoom_C,
}


# ──────────────────────── DISTRIBUTION HELPERS ───────────────────────────────

def _dist_A(ax, v3, v4):
    """Library A: histogram of displacement magnitude, greedy vs viterbi."""
    disp_g = np.sqrt(v3["dx_um"]**2 + v3["dy_um"]**2)
    disp_v = np.sqrt(v4["dx_um"]**2 + v4["dy_um"]**2)
    bins = np.linspace(0, max(disp_g.max(), disp_v.max()) * 1.05, 40)
    ax.hist(disp_g, bins=bins, color="steelblue", alpha=0.4, density=True,
            label="Greedy", edgecolor="steelblue", linewidth=0.5)
    ax.hist(disp_v, bins=bins, color="coral", alpha=0.4, density=True,
            label="Viterbi", edgecolor="coral", linewidth=0.5,
            histtype="stepfilled", hatch="///")
    ax.set_xlabel(r"|displacement| [$\mu$m]", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.legend(fontsize=10, loc="upper right", framealpha=0.7)
    ax.tick_params(labelsize=10)


def _dist_B(ax, v3, v4):
    """Library B: 2D scatter of (K_Rb, lw) occupancy, greedy vs viterbi."""
    ax.scatter(v3["k_rb_seq"], v3["lw_seq"].astype(float),
               s=12, alpha=0.15, c="red", rasterized=True, edgecolors="none")
    ax.scatter(v4["k_rb_seq"], v4["lw_seq"].astype(float),
               s=18, alpha=0.15, c="navy", rasterized=True, marker="x",
               linewidths=0.8)
    ax.set_xlabel(r"$K_{Rb}$", fontsize=11)
    ax.set_ylabel("lw [px]", fontsize=11)
    h_g = Line2D([0], [0], marker="o", color="red", ls="", ms=6, alpha=0.7,
                 label="Greedy")
    h_v = Line2D([0], [0], marker="x", color="navy", ls="", ms=6, alpha=0.7,
                 label="Viterbi")
    ax.legend(handles=[h_g, h_v], fontsize=10, loc="upper right",
              framealpha=0.7)
    ax.tick_params(labelsize=10)


def _dist_D(ax, v3, v4):
    """
    Library D: grouped bar chart of delta_alpha, delta_Rb, delta_Cm.
    For each discrete grid value, two side-by-side bars:
      Greedy (dots hatch), Viterbi (stripes hatch).
    Parameters are distinguished by color.
    """
    param_info = [
        ("da_seq",  r"$\delta\alpha$",  "steelblue"),
        ("dRb_seq", r"$\delta R_b$",    "red"),
        ("dCm_seq", r"$\delta C_m$",    "green"),
    ]

    # Convert fractional -> percent, round to 1-decimal grid (0.1% step)
    def to_pct_grid(arr):
        return np.round(np.asarray(arr, dtype=float) * 100.0, 1)

    # Discrete grid shared across parameters and algorithms
    # Explicit grid from library design: -1.0% to +1.0% in 0.1% steps
    grid_step = 0.1
    grid = np.round(np.arange(-1.0, 1.0 + grid_step / 2, grid_step), 1)
    # Slot layout: at each grid value, 3 parameters * 2 algorithms = 6 bars.
    # Cluster width < grid_step so clusters do not overlap between grid values.
    cluster_w = grid_step * 0.85
    n_params  = len(param_info)                      # 3
    pair_w    = cluster_w / n_params                 # width per (param,algo) pair
    bar_w     = pair_w * 0.45                        # each bar is 45% of its pair slot
    # Offsets: for param p in {0,1,2}, pair centre relative to cluster centre
    pair_offsets = (np.arange(n_params) - (n_params - 1) / 2.0) * pair_w
    # Within a pair: greedy at -bar_w/2, viterbi at +bar_w/2
    algo_offset  = bar_w / 2.0

    def occupancy(arr_pct):
        """Fraction of time spent at each grid value."""
        counts = np.zeros_like(grid, dtype=float)
        # Bin each sample into its closest grid cell (handles FP noise)
        idx = np.clip(
            np.round((arr_pct - grid[0]) / grid_step).astype(int),
            0, len(grid) - 1,
        )
        np.add.at(counts, idx, 1)
        return counts / counts.sum() if counts.sum() > 0 else counts

    for p_idx, (key, label, color) in enumerate(param_info):
        pct_g = to_pct_grid(v3[key])
        pct_v = to_pct_grid(v4[key])
        frac_g = occupancy(pct_g)
        frac_v = occupancy(pct_v)

        xc = grid + pair_offsets[p_idx]
        # Greedy: dots hatch
        ax.bar(xc - algo_offset, frac_g, bar_w,
               color=color, alpha=0.55, edgecolor="black", linewidth=0.1,
               hatch="...", label=f"{label} (G)")
        # Viterbi: stripes hatch
        ax.bar(xc + algo_offset, frac_v, bar_w,
               color=color, alpha=0.55, edgecolor="black", linewidth=0.1,
               hatch="///", label=f"{label} (V)")

    ax.set_xlabel("Perturbation [%]", fontsize=11)
    ax.set_ylabel("Fraction of time", fontsize=11)
    # Sparse ticks on the grid for readability
    
    tick_vals = grid[::2]
    ax.set_xticks(tick_vals)
    
    ax.set_xticklabels([f"{v:.1f}" for v in tick_vals], rotation=-45, ha="left")
    
    # Custom legend: two rows (one per algorithm) listing all three params
    from matplotlib.patches import Patch
    legend_handles = []
    for _, label, color in param_info:
        legend_handles.append(Patch(facecolor=color, alpha=0.55,
                                    edgecolor="black", hatch="...",
                                    label=f"{label} (Greedy)"))
    for _, label, color in param_info:
        legend_handles.append(Patch(facecolor=color, alpha=0.55,
                                    edgecolor="black", hatch="///",
                                    label=f"{label} (Viterbi)"))
    ax.legend(handles=legend_handles, fontsize=8.5, loc="upper right",
              framealpha=0.7, ncol=2, handlelength=1.5)
    ax.tick_params(labelsize=10)


def _dist_C(ax, v3, v4):
    """Library C: bar chart of fractional occupancy per offset index."""
    c_g = v3["c_idx_seq"].astype(int)
    c_v = v4["c_idx_seq"].astype(int)
    max_idx = max(c_g.max(), c_v.max())
    indices = np.arange(0, max_idx + 1)
    width = 0.38

    counts_g = np.bincount(c_g, minlength=max_idx + 1).astype(float)
    counts_v = np.bincount(c_v, minlength=max_idx + 1).astype(float)
    frac_g = counts_g / counts_g.sum()
    frac_v = counts_v / counts_v.sum()

    ax.bar(indices - width / 2, frac_g, width, color="purple", alpha=0.5,
           label="Greedy", edgecolor="purple", linewidth=0.4)
    ax.bar(indices + width / 2, frac_v, width, color="orchid", alpha=0.5,
           label="Viterbi", edgecolor="orchid", linewidth=0.4)

    ax.set_xlabel("Brick-wall offset [px]", fontsize=11)
    ax.set_ylabel("Fraction of time", fontsize=11)
    ax.legend(fontsize=10, loc="upper right", framealpha=0.7)
    ax.tick_params(labelsize=10)


_DIST_FN = {
    "A_shift":     _dist_A,
    "B_rb":        _dist_B,
    "D_rb":        _dist_D,
    "C_brickwall": _dist_C,
}


# ──────────────────────────── MAIN FIGURE ────────────────────────────────────

def plot_states_figure_v4(stem: str, save: bool = True):
    """
    4 rows x 3 columns:
      Col 0: Statistical distribution (full recording, greedy vs viterbi)
      Col 1: Greedy zoom (15-min window)
      Col 2: Viterbi zoom (15-min window)

    Row order: A_shift, B_rb, D_rb, C_brickwall
    """
    v3_path = RESULTS_DIR / f"reconstruct_v3_{stem}.npz"
    v4_path = RESULTS_DIR / f"reconstruct_v4_{stem}.npz"
    missing = [str(p) for p in (v3_path, v4_path) if not p.exists()]
    if missing:
        warnings.warn(f"[{stem}] missing: {missing} -- skipped (states v4)")
        return None

    v3 = np.load(v3_path, allow_pickle=True)
    v4 = np.load(v4_path, allow_pickle=True)

    t_s   = v3["t"]
    t_min = t_s / 60.0
    i_lo, i_hi = zoom_window(t_s, v3["dR_exp"])
    t_zoom = t_min[i_lo:i_hi+1]

    fig, axes = plt.subplots(
        4, 3, figsize=(15, 13),
        gridspec_kw={"width_ratios": [1.0, 0.8, 0.8]},
    )
    fig.subplots_adjust(left=0.07, right=0.93, top=0.94, bottom=0.06,
                        hspace=0.40, wspace=0.35)

    axes[0, 0].set_title("Distribution (full recording)",
                         fontsize=12, fontweight="bold", pad=14)
    axes[0, 1].set_title("Greedy zoom", fontsize=12, fontweight="bold", pad=14)
    axes[0, 2].set_title("Viterbi zoom", fontsize=12, fontweight="bold", pad=14)

    for row, (lib_key, subtitle) in enumerate(_STATE_ROWS):
        # Column 0: distribution
        _DIST_FN[lib_key](axes[row, 0], v3, v4)

        # Column 1: greedy zoom
        _ZOOM_FN[lib_key](axes[row, 1], t_min, v3, i_lo, i_hi)

        # Column 2: viterbi zoom
        _ZOOM_FN[lib_key](axes[row, 2], t_min, v4, i_lo, i_hi)

        # Row subtitle on distribution column
        if row == 0:
            axes[row, 0].text(
                0.0, 1.18, subtitle, transform=axes[row, 0].transAxes,
                fontsize=11, fontweight="bold", ha="left", va="bottom")
        else:
            axes[row, 0].set_title(
                subtitle, fontsize=11, fontweight="bold", loc="left", pad=8)

        # Zoom columns: fix xlim and tick size
        for c in (1, 2):
            axes[row, c].set_xlim(t_zoom[0], t_zoom[-1])
            axes[row, c].tick_params(labelsize=10)

        # x-axis labels: bottom row only for zoom columns
        if row < 3:
            for c in (1, 2):
                axes[row, c].tick_params(labelbottom=False)
        else:
            for c in (1, 2):
                axes[row, c].set_xlabel("Time [min]", fontsize=11)

    fig.suptitle(
        f"{stem} -- State sequences (distribution + zoom)",
        fontsize=13, fontweight="bold",
    )

    if save:
        out = RESULTS_DIR / f"mode_states_{stem}_v4.png"
        fig.savefig(out, dpi=DPI, bbox_inches="tight")
        print(f"  Saved {out}")

    return fig


# ──────────────────────────────── MAIN ───────────────────────────────────────

def main():
    v3_files = sorted(RESULTS_DIR.glob("reconstruct_v3_*.npz"))
    if not v3_files:
        raise FileNotFoundError(f"No reconstruct_v3_*.npz files in {RESULTS_DIR}/")

    stems = [p.stem.replace("reconstruct_v3_", "") for p in v3_files]
    print(f"Stems found: {stems}")

    for stem in stems:
        print(f"\n--- {stem} ---")
        fig = plot_states_figure_v4(stem, save=True)
        if fig is not None:
            plt.close(fig)

    print("\nDone.")


if __name__ == "__main__":
    main()