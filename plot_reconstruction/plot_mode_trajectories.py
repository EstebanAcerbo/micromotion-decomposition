"""
plot_mode_trajectories.py
=========================
Publication-quality figures for ECIS micromotion mode trajectory analysis.

Per stem (auto-discovered from results/reconstruct_v3_*.npz):
  mode_trajectories_{stem}.png      -- dR waterfall (Strategy B)
  mode_trajectories_{stem}_dX.png   -- dX waterfall (same logic, imag component)
  mode_states_{stem}.png            -- state sequences, overlaid zoom (3 cols)
  mode_states_{stem}_v2.png         -- state sequences, separated zoom (4 cols)

Combined:
  mode_trajectories_all.png         -- 2x2 greedy-only dR waterfall grid
"""

import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import numpy as np

# ────────────────────────────── CONFIG ───────────────────────────────────────

RESULTS_DIR       = Path(os.path.dirname(os.path.abspath(__file__))) / '..' / 'results'
FREQ_TARGET       = 1000.0
TWO_ELECTRODE_FAC = 2.0
ZOOM_HALF_S       = 2.5 * 60.0     # 5-min total window, half-width in seconds
DPI               = 200

LIB_ORDER = ["B_rb", "D_rb", "A_shift", "C_brickwall"]

LIB_LABELS = {
    "B_rb":        "Lib B  (Rb)",
    "D_rb":        "Lib D  (alpha, Rb, Cm)",
    "A_shift":     "Lib A  (shift)",
    "C_brickwall": "Lib C  (brick-wall)",
}
LIB_COLORS = {
    "B_rb":        "red",
    "D_rb":        "teal",
    "A_shift":     "steelblue",
    "C_brickwall": "purple",
}
LIB_FILES = {
    "B_rb":        "library_B_rb.npz",
    "D_rb":        "library_D_rb.npz",
    "A_shift":     "library_A_shift.npz",
    "C_brickwall": "library_C_brickwall.npz",
}
LIB_IDX_KEY = {
    "B_rb":        "idx_B",
    "D_rb":        "idx_D",
    "A_shift":     "idx_A",
    "C_brickwall": "idx_C",
}

# Line-style convention: Greedy = dashed, Viterbi = dotted
LS_GREEDY  = "--"
LS_VITERBI = ":"

# Shared legend handles for state figures
_ALG_HANDLES = [
    Line2D([0], [0], color="k", ls=LS_GREEDY,  lw=1.5, label="Greedy"),
    Line2D([0], [0], color="k", ls=LS_VITERBI, lw=1.5, label="Viterbi"),
]
_ROW_PARAM_HANDLES = {
    "A_shift": [
        Line2D([0], [0], color="steelblue", lw=1.5, label="dx"),
        Line2D([0], [0], color="coral",     lw=1.5, label="dy"),
    ],
    "B_rb": [
        Line2D([0], [0], color="red",  lw=1.5, label=r"$K_{Rb}$"),
        Line2D([0], [0], color="navy", lw=1.5, label="lw"),
    ],
    "D_rb": [
        Line2D([0], [0], color="steelblue", lw=1.5, label=r"$\delta\alpha$"),
        Line2D([0], [0], color="red",       lw=1.5, label=r"$\delta R_b$"),
        Line2D([0], [0], color="green",     lw=1.5, label=r"$\delta C_m$"),
    ],
    "C_brickwall": [
        Line2D([0], [0], color="purple", lw=1.5, label="offset"),
    ],
}

# D_rb parameters: (array key, color, alpha)
_D_PARAMS = [
    ("da_seq",  "steelblue", 0.85),
    ("dRb_seq", "red",       0.6),
    ("dCm_seq", "green",     0.3),
]


# ───────────────────────────── LIBRARY I/O ───────────────────────────────────

def load_all_lib_dZ(component: str = "R") -> dict:
    """
    Load dR (component='R') or dX (component='X') arrays at ~1 kHz,
    two-electrode factor applied.  Raw library size, no null prepend.
    """
    out = {}
    for lib_key in LIB_ORDER:
        lib = np.load(RESULTS_DIR / LIB_FILES[lib_key], allow_pickle=True)
        i_f = int(np.argmin(np.abs(lib["freqs"] - FREQ_TARGET)))
        dZ  = (lib["Z"][:, i_f] - lib["Z_baseline"][i_f]) * TWO_ELECTRODE_FAC
        out[lib_key] = dZ.real if component == "R" else dZ.imag
    return out


def make_v4_lib_dZ(lib_raw: dict) -> dict:
    """
    Null-mode-prepended arrays for v4 index mapping.
    B_rb, D_rb, C_brickwall get a zero prepended.  A_shift mode 0 is already null.
    """
    return {
        "A_shift":     lib_raw["A_shift"],
        "B_rb":        np.concatenate([[0.0], lib_raw["B_rb"]]),
        "D_rb":        np.concatenate([[0.0], lib_raw["D_rb"]]),
        "C_brickwall": np.concatenate([[0.0], lib_raw["C_brickwall"]]),
    }


# ──────────────────────────── CALIBRATION ────────────────────────────────────

def detect_cal_factor(rec, lib_dZ: dict, component: str = "R") -> float:
    """
    Recover calibration factor from a reconstruct npz.
    d{component}_rec[t] = cal * sum_k( lib_dZ[k][idx_k[t]] ).
    Pass null-prepended lib_dZ for v4.
    """
    rec_key = f"d{component}_rec"
    sizes   = {k: len(v) for k, v in lib_dZ.items()}
    T       = len(rec["t"])
    step    = max(1, T // 500)
    ratios  = []
    for t in range(0, T, step):
        if any(int(rec[LIB_IDX_KEY[k]][t]) >= sizes[k] for k in LIB_ORDER):
            continue
        s = sum(lib_dZ[k][int(rec[LIB_IDX_KEY[k]][t])] for k in LIB_ORDER)
        if abs(s) > 1.0:
            ratios.append(float(rec[rec_key][t]) / s)
    if len(ratios) < 5:
        warnings.warn("cal detection: too few valid samples, defaulting to 1.0")
        return 1.0
    med = float(np.median(ratios))
    return 1.0 if abs(med - 1.0) < 1e-3 else med


# ──────────────────────────── CONTRIBUTIONS ──────────────────────────────────

def get_contributions(rec, lib_dZ: dict, cal: float) -> dict:
    """Per-library dZ time series with calibration.  OOB indices -> NaN."""
    contribs = {}
    for lib_key in LIB_ORDER:
        idx    = rec[LIB_IDX_KEY[lib_key]].copy()
        arr    = lib_dZ[lib_key]
        valid  = idx < len(arr)
        out    = np.full(len(idx), np.nan, dtype=np.float64)
        out[valid] = arr[idx[valid]] * cal
        if not np.all(valid):
            warnings.warn(f"  {lib_key}: {(~valid).sum()} OOB indices -> NaN")
        contribs[lib_key] = out
    return contribs


# ────────────────────────────── ZOOM LOGIC ───────────────────────────────────

def zoom_window(t_s: np.ndarray, sig: np.ndarray) -> tuple:
    """ZOOM_HALF_S window centred on peak |sig|, clamped to array bounds."""
    i_peak = int(np.argmax(np.abs(sig)))
    t_peak = t_s[i_peak]
    i_lo   = max(0,             int(np.searchsorted(t_s, t_peak - ZOOM_HALF_S)))
    i_hi   = min(len(t_s) - 1, int(np.searchsorted(t_s, t_peak + ZOOM_HALF_S)))
    return i_lo, i_hi


# ──────────────── FIGURE 1: dR / dX WATERFALL (Strategy B) ──────────────────

def plot_waterfall_figure(stem: str,
                          lib_dZ_v3: dict, lib_dZ_v4: dict,
                          component: str = "R",
                          save: bool = True):
    """
    Two-column (Greedy | Viterbi) x four-row (B, D, A, C) waterfall.
    component='R' -> dR contributions; component='X' -> dX contributions.

    Strategy B: full trace faded + zoom segment at full visibility on same axes.
    Shared y-limits per row across both columns.
    """
    v3_path = RESULTS_DIR / f"reconstruct_v3_{stem}.npz"
    v4_path = RESULTS_DIR / f"reconstruct_v4_{stem}.npz"
    missing = [str(p) for p in (v3_path, v4_path) if not p.exists()]
    if missing:
        warnings.warn(f"[{stem}] missing: {missing} -- skipped")
        return None, None, None, None

    v3 = np.load(v3_path, allow_pickle=True)
    v4 = np.load(v4_path, allow_pickle=True)

    t_s  = v3["t"]
    t_min = t_s / 60.0
    exp  = v3[f"d{component}_exp"]

    i_lo, i_hi = zoom_window(t_s, exp)

    cal_v3 = detect_cal_factor(v3, lib_dZ_v3, component)
    cal_v4 = detect_cal_factor(v4, lib_dZ_v4, component)
    print(f"  [{stem}] d{component}  cal_v3={cal_v3:.4f}  cal_v4={cal_v4:.4f}")

    c3 = get_contributions(v3, lib_dZ_v3, cal_v3)
    c4 = get_contributions(v4, lib_dZ_v4, cal_v4)

    fig, axes = plt.subplots(4, 2, figsize=(14, 10), sharex="col")
    fig.subplots_adjust(left=0.09, right=0.97, top=0.92, bottom=0.07,
                        hspace=0.18, wspace=0.25)

    for col, (rec, contribs, algo_lbl) in enumerate([
        (v3, c3, "Greedy"),
        (v4, c4, "Viterbi"),
    ]):
        rmse_R = float(rec["rmse_R"])
        rmse_X = float(rec["rmse_X"])

        for row, lib_key in enumerate(LIB_ORDER):
            ax    = axes[row, col]
            trace = contribs[lib_key]
            color = LIB_COLORS[lib_key]

            ax.plot(t_min, trace, color=color, lw=0.4, alpha=0.20, rasterized=True)
            ax.plot(t_min[i_lo:i_hi+1], trace[i_lo:i_hi+1],
                    color=color, lw=1.0, alpha=1.0)
            for xm in (t_min[i_lo], t_min[i_hi]):
                ax.axvline(xm, color="0.45", lw=0.65, ls="--", alpha=0.8)

            ax.grid(True, lw=0.25, alpha=0.40, color="0.7", zorder=0)
            ax.set_ylabel(LIB_LABELS[lib_key], fontsize=9)
            ax.tick_params(labelsize=8)

            rms = np.sqrt(np.nanmean(trace ** 2))
            ax.text(0.98, 0.95, f"RMS = {rms:.1f} Ohm",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=7.5, color="0.3")

            if row == 0:
                ax.set_title(
                    f"{algo_lbl}\nRMSE R={rmse_R:.1f} Ohm  X={rmse_X:.1f} Ohm",
                    fontsize=9)
            if row == 3:
                ax.set_xlabel("Time [min]", fontsize=9)

    for row in range(4):
        lims = [axes[row, c].get_ylim() for c in range(2)]
        y_lo = min(l[0] for l in lims)
        y_hi = max(l[1] for l in lims)
        for c in range(2):
            axes[row, c].set_ylim(y_lo, y_hi)

    fig.suptitle(
        f"{stem} -- Mode trajectory decomposition (d{component} component)",
        fontsize=10, fontweight="bold")

    if save:
        suffix = "" if component == "R" else f"_d{component}"
        out = RESULTS_DIR / f"mode_trajectories_{stem}{suffix}.png"
        fig.savefig(out, dpi=DPI)
        print(f"  Saved {out}")

    return fig, c3, t_min, exp


# ──────────────── FIGURE 2 SHARED HELPERS ────────────────────────────────────

def _vlines(ax, t_lo, t_hi):
    for xm in (t_lo, t_hi):
        ax.axvline(xm, color="0.50", lw=0.60, ls="--", alpha=0.75)


def _full_A(ax, t_min, rec, ls, t_lo, t_hi):
    ax.plot(t_min, rec["dx_um"], color="steelblue", lw=0.6, ls=ls)
    ax.plot(t_min, rec["dy_um"], color="coral",     lw=0.6, ls=ls)
    ax.set_ylabel(r"Displacement [$\mu$m]", fontsize=12)
    ax.tick_params(labelsize=10)
    _vlines(ax, t_lo, t_hi)


def _full_B(ax, t_min, rec, ls, t_lo, t_hi):
    ax.step(t_min, rec["k_rb_seq"], color="red", lw=0.6, where="mid", ls=ls)
    ax.set_ylabel(r"$K_{Rb}$", fontsize=12, color="red")
    ax.tick_params(axis="y", labelsize=10, colors="red")
    ax.spines["left"].set_edgecolor("red")
    ax_tw = ax.twinx()
    ax_tw.step(t_min, rec["lw_seq"].astype(float),
               color="navy", lw=0.6, where="mid", ls=ls, alpha=0.8)
    ax_tw.set_ylabel("lw [px]", fontsize=10, color="navy")
    ax_tw.tick_params(axis="y", labelsize=9, colors="navy")
    ax_tw.spines["right"].set_edgecolor("navy")
    _vlines(ax, t_lo, t_hi)


def _full_D(ax, t_min, rec, ls, t_lo, t_hi):
    ax.axhline(0, color="0.65", lw=0.35)
    for arr, color, alpha in _D_PARAMS:
        ax.step(t_min, rec[arr], color=color, lw=0.6, where="mid",
                ls=ls, alpha=alpha)
    ax.set_ylabel("Param. perturbation [%]", fontsize=12)
    ax.tick_params(labelsize=10)
    _vlines(ax, t_lo, t_hi)


def _full_C(ax, t_min, rec, ls, t_lo, t_hi):
    ax.step(t_min, rec["c_idx_seq"], color="purple", lw=0.7, where="mid", ls=ls)
    ax.set_ylabel("Brick-wall offset [px]", fontsize=12)
    ax.tick_params(labelsize=10)
    _vlines(ax, t_lo, t_hi)


def _zoom_A(ax, t_min, rec, ls, i_lo, i_hi):
    t_z = t_min[i_lo:i_hi+1]
    ax.plot(t_z, rec["dx_um"][i_lo:i_hi+1], color="steelblue", lw=1.0,
            ls=ls, alpha=0.85)
    ax.plot(t_z, rec["dy_um"][i_lo:i_hi+1], color="coral",     lw=1.0,
            ls=ls, alpha=0.85)


def _zoom_B(ax, t_min, rec, ls, i_lo, i_hi):
    t_z = t_min[i_lo:i_hi+1]
    ax.step(t_z, rec["k_rb_seq"][i_lo:i_hi+1], color="red",  lw=1.0,
            where="mid", ls=ls, alpha=0.85)
    ax.set_ylabel(r"$K_{Rb}$", fontsize=12, color="red")
    ax.tick_params(axis="y", labelsize=10, colors="red")
    ax.spines["left"].set_edgecolor("red")
    # Reuse existing twin axis if already created (overlaid zoom calls this twice)
    twins = [c for c in ax.get_shared_x_axes().get_siblings(ax)
             if c is not ax]
    ax_tw = twins[0] if twins else ax.twinx()
    ax_tw.step(t_z, rec["lw_seq"][i_lo:i_hi+1].astype(float),
               color="navy", lw=1.0, where="mid", ls=ls, alpha=0.85)
    ax_tw.set_ylabel("lw [px]", fontsize=10, color="navy")
    ax_tw.tick_params(axis="y", labelsize=9, colors="navy")
    ax_tw.spines["right"].set_edgecolor("navy")


def _zoom_D(ax, t_min, rec, ls, i_lo, i_hi):
    t_z = t_min[i_lo:i_hi+1]
    ax.axhline(0, color="0.65", lw=0.35)
    for arr, color, alpha in _D_PARAMS:
        ax.step(t_z, rec[arr][i_lo:i_hi+1], color=color, lw=1.0,
                where="mid", ls=ls, alpha=alpha)


def _zoom_C(ax, t_min, rec, ls, i_lo, i_hi):
    t_z = t_min[i_lo:i_hi+1]
    ax.step(t_z, rec["c_idx_seq"][i_lo:i_hi+1], color="purple", lw=1.0,
            where="mid", ls=ls, alpha=0.85)


_FULL_FN = {
    "A_shift":     _full_A,
    "B_rb":        _full_B,
    "D_rb":        _full_D,
    "C_brickwall": _full_C,
}
_ZOOM_FN = {
    "A_shift":     _zoom_A,
    "B_rb":        _zoom_B,
    "D_rb":        _zoom_D,
    "C_brickwall": _zoom_C,
}

_STATE_ROWS = [
    ("A_shift",     "A_shift: displacement"),
    ("B_rb",        "B_rb: K_Rb and lw"),
    ("D_rb",        "D_rb: param. perturbations"),
    ("C_brickwall", "C_brickwall: brick-wall offset"),
]


def _add_row_legend(ax, lib_key):
    """Place shared row legend to the right of ax."""
    handles = _ALG_HANDLES + _ROW_PARAM_HANDLES[lib_key]
    # B_rb has a twin axis with right-side labels; push legend further out
    x_anchor = 1.22 if lib_key == "B_rb" else 1.04
    ax.legend(handles=handles,
              loc="center left", bbox_to_anchor=(x_anchor, 0.5),
              bbox_transform=ax.transAxes,
              fontsize=10, handlelength=2.2,
              framealpha=0.85, borderaxespad=0)


def _finalize_states_fig(fig, axes, t_zoom, stems_title, n_cols, save_path,
                         zoom_cols=None):
    if zoom_cols is None:
        zoom_cols = list(range(2, n_cols))
    for row, (lib_key, subtitle) in enumerate(_STATE_ROWS):
        axes[row, 0].set_title(subtitle, fontsize=10, loc="left", pad=2)
        # zoom columns: fix xlim and ticks
        for c in zoom_cols:
            axes[row, c].set_xlim(t_zoom[0], t_zoom[-1])
            axes[row, c].tick_params(labelsize=10)
        # row legend on last column
        _add_row_legend(axes[row, n_cols - 1], lib_key)
        # x-axis labels: bottom row only
        if row < 3:
            for c in range(n_cols):
                axes[row, c].tick_params(labelbottom=False)
        else:
            for c in range(n_cols):
                axes[row, c].set_xlabel("Time [min]", fontsize=12)

    if save_path:
        fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
        print(f"  Saved {save_path}")


# ──────────────── FIGURE 2a: STATE SEQUENCES (3-column, overlaid zoom) ───────

def plot_states_figure(stem: str, save: bool = True):
    """
    4 rows x 3 columns: Greedy | Viterbi | overlaid zoom (both algos).
    """
    v3_path = RESULTS_DIR / f"reconstruct_v3_{stem}.npz"
    v4_path = RESULTS_DIR / f"reconstruct_v4_{stem}.npz"
    missing = [str(p) for p in (v3_path, v4_path) if not p.exists()]
    if missing:
        warnings.warn(f"[{stem}] missing: {missing} -- skipped (states)")
        return None

    v3 = np.load(v3_path, allow_pickle=True)
    v4 = np.load(v4_path, allow_pickle=True)

    t_s   = v3["t"]
    t_min = t_s / 60.0
    i_lo, i_hi = zoom_window(t_s, v3["dR_exp"])
    t_zoom = t_min[i_lo:i_hi+1]

    fig, axes = plt.subplots(
        4, 3, figsize=(16, 10),
        gridspec_kw={"width_ratios": [1, 1, 0.6]},
    )
    fig.subplots_adjust(left=0.08, right=0.85, top=0.96, bottom=0.07,
                        hspace=0.15, wspace=0.32)

    axes[0, 0].set_title("Greedy",  fontsize=12, fontweight="bold")
    axes[0, 1].set_title("Viterbi", fontsize=12, fontweight="bold")
    axes[0, 2].set_title("Zoom",    fontsize=12, fontweight="bold")

    t_lo, t_hi = t_min[i_lo], t_min[i_hi]

    for row, (lib_key, _) in enumerate(_STATE_ROWS):
        _FULL_FN[lib_key](axes[row, 0], t_min, v3, LS_GREEDY,  t_lo, t_hi)
        _FULL_FN[lib_key](axes[row, 1], t_min, v4, LS_VITERBI, t_lo, t_hi)
        # overlaid zoom: both algorithms on ax_z
        _ZOOM_FN[lib_key](axes[row, 2], t_min, v3, LS_GREEDY,  i_lo, i_hi)
        _ZOOM_FN[lib_key](axes[row, 2], t_min, v4, LS_VITERBI, i_lo, i_hi)

    save_path = (RESULTS_DIR / f"mode_states_{stem}.png") if save else None
    _finalize_states_fig(fig, axes, t_zoom, stem, n_cols=3, save_path=save_path)
    return fig


# ──────────────── FIGURE 2b: STATE SEQUENCES v2 (4-column, separated zoom) ──

def plot_states_figure_v2(stem: str, save: bool = True):
    """
    4 rows x 4 columns: Greedy full | Greedy zoom || Viterbi full | Viterbi zoom.
    Two GridSpec objects with a visual gap between the Greedy and Viterbi groups.
    All four data columns have equal width.  Twin-axis (B_rb) panels are color-coded.
    """
    v3_path = RESULTS_DIR / f"reconstruct_v3_{stem}.npz"
    v4_path = RESULTS_DIR / f"reconstruct_v4_{stem}.npz"
    missing = [str(p) for p in (v3_path, v4_path) if not p.exists()]
    if missing:
        warnings.warn(f"[{stem}] missing: {missing} -- skipped (states v2)")
        return None

    v3 = np.load(v3_path, allow_pickle=True)
    v4 = np.load(v4_path, allow_pickle=True)

    t_s   = v3["t"]
    t_min = t_s / 60.0
    i_lo, i_hi = zoom_window(t_s, v3["dR_exp"])
    t_zoom = t_min[i_lo:i_hi+1]

    # Two GridSpec halves of equal width separated by a gap.
    # figure coordinates: left=0.06, gap center at 0.49, right=0.87
    # (legend will overflow right=0.87 and be captured by bbox_inches="tight")
    _GS_KW = dict(top=0.94, bottom=0.07, hspace=0.15, wspace=0.33)
    fig = plt.figure(figsize=(22, 10))
    gs_g = GridSpec(4, 2, figure=fig, left=0.06, right=0.44, **_GS_KW)
    gs_v = GridSpec(4, 2, figure=fig, left=0.50, right=0.87, **_GS_KW)

    # axes layout: col 0 = Greedy full, col 1 = Greedy zoom,
    #              col 2 = Viterbi full, col 3 = Viterbi zoom
    axes = np.empty((4, 4), dtype=object)
    for row in range(4):
        axes[row, 0] = fig.add_subplot(gs_g[row, 0])
        axes[row, 1] = fig.add_subplot(gs_g[row, 1])
        axes[row, 2] = fig.add_subplot(gs_v[row, 0])
        axes[row, 3] = fig.add_subplot(gs_v[row, 1])

    axes[0, 0].set_title("Greedy",       fontsize=12, fontweight="bold")
    axes[0, 1].set_title("Greedy zoom",  fontsize=12, fontweight="bold")
    axes[0, 2].set_title("Viterbi",      fontsize=12, fontweight="bold")
    axes[0, 3].set_title("Viterbi zoom", fontsize=12, fontweight="bold")

    t_lo, t_hi = t_min[i_lo], t_min[i_hi]

    for row, (lib_key, _) in enumerate(_STATE_ROWS):
        _FULL_FN[lib_key](axes[row, 0], t_min, v3, LS_GREEDY,  t_lo, t_hi)
        _ZOOM_FN[lib_key](axes[row, 1], t_min, v3, LS_GREEDY,  i_lo, i_hi)
        _FULL_FN[lib_key](axes[row, 2], t_min, v4, LS_VITERBI, t_lo, t_hi)
        _ZOOM_FN[lib_key](axes[row, 3], t_min, v4, LS_VITERBI, i_lo, i_hi)

    save_path = (RESULTS_DIR / f"mode_states_{stem}_v2.png") if save else None
    _finalize_states_fig(fig, axes, t_zoom, stem, n_cols=4, save_path=save_path,
                         zoom_cols=[1, 3])
    return fig


# ──────────────── FIGURE 2c: STATE SEQUENCES v3 (zoom-only, 2 columns) ──────

def plot_states_figure_v3(stem: str, save: bool = True):
    """
    4 rows x 2 columns: Greedy zoom | Viterbi zoom.
    Zoom window only — no full-trace panels.
    """
    v3_path = RESULTS_DIR / f"reconstruct_v3_{stem}.npz"
    v4_path = RESULTS_DIR / f"reconstruct_v4_{stem}.npz"
    missing = [str(p) for p in (v3_path, v4_path) if not p.exists()]
    if missing:
        warnings.warn(f"[{stem}] missing: {missing} -- skipped (states v3)")
        return None

    v3 = np.load(v3_path, allow_pickle=True)
    v4 = np.load(v4_path, allow_pickle=True)

    t_s   = v3["t"]
    t_min = t_s / 60.0
    i_lo, i_hi = zoom_window(t_s, v3["dR_exp"])
    t_zoom = t_min[i_lo:i_hi+1]

    fig, axes = plt.subplots(4, 2, figsize=(12, 10))
    fig.subplots_adjust(left=0.10, right=0.82, top=0.94, bottom=0.07,
                        hspace=0.15, wspace=0.35)

    axes[0, 0].set_title("Greedy zoom",  fontsize=12, fontweight="bold")
    axes[0, 1].set_title("Viterbi zoom", fontsize=12, fontweight="bold")

    for row, (lib_key, _) in enumerate(_STATE_ROWS):
        _ZOOM_FN[lib_key](axes[row, 0], t_min, v3, LS_GREEDY,  i_lo, i_hi)
        _ZOOM_FN[lib_key](axes[row, 1], t_min, v4, LS_VITERBI, i_lo, i_hi)

    save_path = (RESULTS_DIR / f"mode_states_{stem}_v3.png") if save else None
    _finalize_states_fig(fig, axes, t_zoom, stem, n_cols=2, save_path=save_path,
                         zoom_cols=[0, 1])
    return fig


# ──────────────── COMBINED FIGURE (greedy-only, 2x2) ─────────────────────────

def plot_combined_greedy(stem_data: dict):
    """2x2 grid, greedy-only dR waterfall, Strategy B."""
    stems = list(stem_data.keys())
    if len(stems) != 4:
        warnings.warn(f"Combined requires 4 stems, found {len(stems)} -- skipped.")
        return

    fig = plt.figure(figsize=(16, 14), constrained_layout=True)
    outer = fig.add_gridspec(2, 2)

    for s_idx, stem in enumerate(stems):
        t_min, dR_exp, contribs = stem_data[stem]
        t_s = t_min * 60.0
        i_lo, i_hi = zoom_window(t_s, dR_exp)
        t_lo_min, t_hi_min = t_min[i_lo], t_min[i_hi]

        inner = outer[s_idx // 2, s_idx % 2].subgridspec(4, 1, hspace=0.0)

        for row, lib_key in enumerate(LIB_ORDER):
            ax    = fig.add_subplot(inner[row])
            trace = contribs[lib_key]
            color = LIB_COLORS[lib_key]

            ax.plot(t_min, trace, color=color, lw=0.4, alpha=0.20, rasterized=True)
            ax.plot(t_min[i_lo:i_hi+1], trace[i_lo:i_hi+1],
                    color=color, lw=1.0, alpha=1.0)
            for xm in (t_lo_min, t_hi_min):
                ax.axvline(xm, color="0.50", lw=0.55, ls="--", alpha=0.75)

            ax.set_ylabel(LIB_LABELS[lib_key], fontsize=7)
            rms = np.sqrt(np.nanmean(trace ** 2))
            ax.text(0.98, 0.94, f"RMS={rms:.1f}Ohm",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=6, color="0.3")
            ax.tick_params(labelsize=6)

            if row == 0:
                ax.set_title(f"{stem} -- Greedy", fontsize=8, fontweight="bold")
            if row < 3:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel("Time [min]", fontsize=7)

    fig.suptitle("Mode trajectory decomposition -- dR (all stems, Greedy)",
                 fontsize=10, fontweight="bold")
    out = RESULTS_DIR / "mode_trajectories_all.png"
    fig.savefig(out, dpi=DPI)
    print(f"  Saved {out}")
    plt.close(fig)


# ──────────────────────────────── MAIN ───────────────────────────────────────

def main():
    v3_files = sorted(RESULTS_DIR.glob("reconstruct_v3_*.npz"))
    if not v3_files:
        raise FileNotFoundError(f"No reconstruct_v3_*.npz files in {RESULTS_DIR}/")

    stems = [p.stem.replace("reconstruct_v3_", "") for p in v3_files]
    print(f"Stems found: {stems}")

    # Load library arrays for both components
    lib_dR_v3 = load_all_lib_dZ("R")
    lib_dR_v4 = make_v4_lib_dZ(lib_dR_v3)
    lib_dX_v3 = load_all_lib_dZ("X")
    lib_dX_v4 = make_v4_lib_dZ(lib_dX_v3)

    stem_data = {}   # stem -> (t_min, dR_exp, greedy_dR_contribs)

    for stem in stems:
        print(f"\n--- {stem} ---")

        # Figure 1a: dR waterfall
        fig_w, c3, t_min, dR_exp = plot_waterfall_figure(
            stem, lib_dR_v3, lib_dR_v4, component="R", save=True)
        if fig_w is not None:
            stem_data[stem] = (t_min, dR_exp, c3)
            plt.close(fig_w)

        # Figure 1b: dX waterfall
        fig_wx = plot_waterfall_figure(
            stem, lib_dX_v3, lib_dX_v4, component="X", save=True)[0]
        if fig_wx is not None:
            plt.close(fig_wx)

        # Figure 2a: state sequences, overlaid zoom
        fig_s = plot_states_figure(stem, save=True)
        if fig_s is not None:
            plt.close(fig_s)

        # Figure 2b: state sequences, separated zoom
        fig_s2 = plot_states_figure_v2(stem, save=True)
        if fig_s2 is not None:
            plt.close(fig_s2)

        # Figure 2c: zoom-only state sequences
        fig_s3 = plot_states_figure_v3(stem, save=True)
        if fig_s3 is not None:
            plt.close(fig_s3)

    # Combined greedy dR waterfall
    if len(stem_data) == 4:
        print("\n--- combined ---")
        plot_combined_greedy(stem_data)
    else:
        warnings.warn(f"Combined skipped: need 4 stems, have {len(stem_data)}.")

    print("\nDone.")


if __name__ == "__main__":
    main()
