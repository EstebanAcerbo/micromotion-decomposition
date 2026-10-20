"""
fractal_plots.py — Fractal dimension figures and LaTeX table (v3 report).
==========================================================================
Generates four figures and a LaTeX table using only Higuchi and R/S methods
on both δR and δX signals.

Figures
-------
  fractal_loglog    — 4×4 grid (4 electrodes × [Higuchi-δR, Higuchi-δX, RS-δR, RS-δX])
  fractal_windowed  — 4×2 grid, per-window Higuchi fits, dR and dX columns
  fractal_reduced   — 1×4 horizontal, Higuchi, both δR and δX overlaid in each panel
  table             — LaTeX booktabs table printed to stdout

All computations run once via compute_all_results() and are reused across
all plots and the table.

Usage
-----
    python Report/fractal_plots.py                   # all figures + table
    python Report/fractal_plots.py fractal_loglog    # one figure
    python Report/fractal_plots.py table             # table only

Available keys: fractal_loglog  fractal_windowed  fractal_reduced  table
"""

import argparse
import math
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np

# ── Path setup ─────────────────────────────────────────────────────────────────
SCRIPT_DIR        = os.path.dirname(os.path.abspath(__file__))
SCRIPT_DIR_PARENT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR_PARENT)

RESULTS = os.path.join(SCRIPT_DIR_PARENT, 'results')
os.makedirs(RESULTS, exist_ok=True)

# ── Global matplotlib style ────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size':        10,
    'font.family':      'sans-serif',
    'mathtext.fontset': 'stix',
})

# ── Electrodes and labels ──────────────────────────────────────────────────────
FOCUS_STEMS  = ['eb11', 'eb04', 'eb13', 'eb15']
FOCUS_LABELS = {
    'eb04': 'Seeding',
    'eb11': 'Confluent',
    'eb13': 'Wound healing',
    'eb15': 'Post-healing',
}
# Electrode colours: used only for row-label text decoration
FOCUS_COLORS = {
    'eb04': '#4CAF50',
    'eb11': '#2196F3',
    'eb13': '#FF9800',
    'eb15': '#E91E63',
}

# Signal colours — consistent with timescales figure (tab:red for R, tab:green for X)
COLOR_DR = 'tab:red'
COLOR_DX = 'tab:green'

# Methods used in this script (no multiresolution)
METHODS       = ['higuchi', 'rs']
METHOD_LABELS = {'higuchi': 'Higuchi', 'rs': 'R/S'}


# ── I/O helpers ────────────────────────────────────────────────────────────────

def _save(fig, name):
    out = os.path.join(RESULTS, name)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved -> {out}')


def _load_exp_npz():
    p = os.path.join(RESULTS, 'phase1_experimental.npz')
    if not os.path.exists(p):
        print(f'[SKIP] Required file not found: {p}')
        return None
    return np.load(p, allow_pickle=False)


def _get_stem_arrays(npz, stem, keys):
    out = {}
    for k in keys:
        found = None
        for nk in npz.files:
            if nk == f'{stem}_{k}' or nk == f'{stem}/{k}':
                found = npz[nk]
                break
        out[k] = found
    return out


# ── Formatting helpers ─────────────────────────────────────────────────────────

def _fmt(mean, std):
    """Format mean±std for legend: std rounded to 1 sig digit, mean decimals matched."""
    if not np.isfinite(std) or std == 0:
        return f'{mean:.3f}\\pm{{---}}'
    dec = -int(math.floor(math.log10(abs(std))))
    return f'{round(mean, dec):.{dec}f}\\pm{round(std, dec):.{dec}f}'


def _fmt_cell(mean, std):
    """Compact LaTeX cell: $1.87 (3)$ — std's leading digit in parens at mean's precision."""
    if not np.isfinite(mean) or not np.isfinite(std):
        return '---'
    if std == 0:
        return f'${mean:.3f}$'
    dec = -int(math.floor(math.log10(abs(std))))
    mean_r    = round(mean, dec)
    std_r     = round(std,  dec)
    std_digit = int(round(std_r * 10 ** dec))
    return f'${mean_r:.{dec}f} ({std_digit})$'


# ── Core analysis: compute and cache ──────────────────────────────────────────

def compute_all_results(npz_exp):
    """
    Run windowed_fractal_analysis for all electrodes × methods × signals.

    Returns
    -------
    dict keyed by (stem, method, signal_key) where signal_key is 'dR' or 'dX'.
    Each value is the full dict returned by windowed_fractal_analysis.
    """
    from fractal_dim import windowed_fractal_analysis

    results = {}
    for stem in FOCUS_STEMS:
        arrs = _get_stem_arrays(npz_exp, stem, ['dR', 'dX', 't'])
        dR, dX, t = arrs['dR'], arrs['dX'], arrs['t']
        if t is None:
            print(f'[WARN] {stem}: time array not found — skipping.')
            continue
        dt = float(np.median(np.diff(t)))

        for meth in METHODS:
            for sig_key, sig in [('dR', dR), ('dX', dX)]:
                if sig is None:
                    print(f'[WARN] {stem}/{sig_key}: not found — skipping.')
                    continue
                print(f'  [{stem}/{meth}/{sig_key}] ...', end=' ', flush=True)
                r = windowed_fractal_analysis(sig, dt, meth)
                results[(stem, meth, sig_key)] = r
                print(f"df_single={r['df_single_mean']:.4f}±{r['df_single_std']:.4f}")

    return results


# ── Panel primitive: draw one signal's curves on an axes ──────────────────────

def _draw_signal_curves(ax, r, color, marker='o',
                        ls_fast='-', ls_slow='-', ls_single='--',
                        black_single=True, show_bp=True):
    """
    Draw scatter + fit lines for one signal on *ax*.  No legend, no axes setup.

    Parameters
    ----------
    ax           : Axes
    r            : dict from windowed_fractal_analysis
    color        : primary scatter/line colour
    marker       : scatter marker style ('o' default; use 'x' for dX in reduced)
    ls_fast      : line style for fast-segment fit
    ls_slow      : line style for slow-segment fit
    ls_single    : line style for single-slope fit
    black_single : if True use black for single-slope line; if False use *color*
    show_bp      : if False, suppress all breakpoint analysis (R/S single-slope only)

    Returns
    -------
    (h_single, h_fast, h_slow, bp_valid) — legend handle objects + validity flag
    """
    rgb         = mcolors.to_rgb(color)
    color_light = tuple(c + (1.0 - c) * 0.55 for c in rgb)

    x_arr     = r['x_arr']
    mean_y    = r['mean_y']
    idx_break = r['idx_break']
    x_break   = r['x_break']

    df_single_mean = r['df_single_mean']
    df_single_std  = r['df_single_std']
    df_fast_mean   = r['df_fast_mean']
    df_fast_std    = r['df_fast_std']
    df_slow_mean   = r['df_slow_mean']
    df_slow_std    = r['df_slow_std']

    slope_avg          = r['slope_avg']
    intercept_avg      = r['intercept_avg']
    slope_fast_avg     = r['slope_fast_avg']
    intercept_fast_avg = r['intercept_fast_avg']
    slope_slow_avg     = r['slope_slow_avg']
    intercept_slow_avg = r['intercept_slow_avg']

    bp_valid = show_bp and (
        idx_break is not None
        and np.isfinite(x_break)
        and not np.isnan(df_fast_mean)
        and not np.isnan(df_slow_mean)
    )

    # Averaged log-log scatter
    ax.scatter(x_arr, mean_y, s=3, marker=marker, color=color, alpha=0.4,
               linewidths=0, zorder=3)

    # Single-slope fit line
    x_line      = np.array([x_arr.min(), x_arr.max()])
    y_line      = 10.0 ** (slope_avg * np.log10(x_line) + intercept_avg)
    single_col  = 'k' if black_single else color
    ax.plot(x_line, y_line, color=single_col, ls=ls_single, lw=1.0, alpha=0.5, zorder=4)

    if bp_valid:
        # Fast-segment fit
        x_f = np.array([x_arr[:idx_break].min(), x_arr[:idx_break].max()])
        y_f = 10.0 ** (slope_fast_avg * np.log10(x_f) + intercept_fast_avg)
        ax.plot(x_f, y_f, color=color, ls=ls_fast, lw=1.4, zorder=5)

        # Slow-segment fit
        x_s = np.array([x_arr[idx_break:].min(), x_arr[idx_break:].max()])
        y_s = 10.0 ** (slope_slow_avg * np.log10(x_s) + intercept_slow_avg)
        ax.plot(x_s, y_s, color=color_light, ls=ls_slow, lw=1.4, zorder=5)

        # Vertical breakpoint line
        ax.axvline(x_break, color='grey', lw=0.8, ls='--', alpha=0.6, zorder=2)

    # Build legend handles
    lbl_single = f'Single  $d_f={_fmt(df_single_mean, df_single_std)}$'
    lbl_fast   = (f'Fast  $d_f^f={_fmt(df_fast_mean, df_fast_std)}$'
                  if bp_valid else 'Fast  n/a')
    lbl_slow   = (f'Slow  $d_f^s={_fmt(df_slow_mean, df_slow_std)}$'
                  if bp_valid else 'Slow  n/a')

    h_single = mlines.Line2D([], [], color=single_col, ls=ls_single, lw=1.0,
                              alpha=0.5, label=lbl_single)
    h_fast   = mlines.Line2D([], [], color=color,       ls=ls_fast, lw=1.4,
                              label=lbl_fast)
    h_slow   = mlines.Line2D([], [], color=color_light, ls=ls_slow, lw=1.4,
                              label=lbl_slow)

    return h_single, h_fast, h_slow, bp_valid


# ── Full panel: one signal + legend + axes + residual inset ───────────────────

def _fractal_loglog_panel(ax, r, color, meth, show_inset=True):
    """
    Draw a complete fractal log-log panel on *ax* for a single signal.

    Parameters
    ----------
    ax         : Axes
    r          : dict from windowed_fractal_analysis
    color      : signal colour (COLOR_DR or COLOR_DX)
    meth       : method string ('higuchi' or 'rs') — controls legend/inset placement
    show_inset : whether to draw the residual inset

    Returns *r* for convenience.
    """
    rgb         = mcolors.to_rgb(color)
    color_light = tuple(c + (1.0 - c) * 0.55 for c in rgb)

    show_bp = (meth != 'rs')
    h_single, h_fast, h_slow, bp_valid = _draw_signal_curves(
        ax, r, color, marker='o', black_single=True, show_bp=show_bp)

    n_win  = r['n_win']
    w_size = r['w_size']
    h_info = mpatches.Patch(alpha=0, label=f'{n_win}\u00d7{w_size} pts')

    leg_handles = [h_single]
    if show_bp:
        leg_handles += [h_fast, h_slow]
    leg_handles.append(h_info)

    leg_loc = 'lower right' if meth == 'rs' else 'upper right'
    ax.legend(handles=leg_handles, fontsize=6, loc=leg_loc, framealpha=0.85)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.tick_params(labelsize=7)
    ax.grid(True, which='both', alpha=0.18)

    if show_inset:
        x_arr   = r['x_arr']
        mean_y  = r['mean_y']
        idx_brk = r['idx_break']
        x_brk   = r['x_break']

        slope_avg          = r['slope_avg']
        intercept_avg      = r['intercept_avg']
        slope_fast_avg     = r['slope_fast_avg']
        intercept_fast_avg = r['intercept_fast_avg']
        slope_slow_avg     = r['slope_slow_avg']
        intercept_slow_avg = r['intercept_slow_avg']

        lx_all    = np.log10(x_arr)
        resid_all = np.log10(mean_y) - (slope_avg * lx_all + intercept_avg)

        inset_pos = (0.1, 0.65, 0.32, 0.32) if meth == 'rs' else (0.1, 0.05, 0.32, 0.32)
        axin = ax.inset_axes(inset_pos)
        axin.scatter(lx_all, resid_all, s=3, color='grey', alpha=0.65,
                     linewidths=0, zorder=2)

        if show_bp and bp_valid and idx_brk is not None and np.isfinite(x_brk):
            lx_fast   = np.log10(x_arr[:idx_brk])
            lx_slow   = np.log10(x_arr[idx_brk:])
            resid_fast = (np.log10(mean_y[:idx_brk])
                          - (slope_fast_avg * lx_fast + intercept_fast_avg))
            resid_slow = (np.log10(mean_y[idx_brk:])
                          - (slope_slow_avg * lx_slow + intercept_slow_avg))
            axin.scatter(lx_fast, resid_fast, s=3, color=color,
                         alpha=0.75, linewidths=0, zorder=3)
            axin.scatter(lx_slow, resid_slow, s=3, color=color_light,
                         alpha=0.75, linewidths=0, zorder=3)
            axin.axvline(np.log10(x_brk), color='grey', lw=0.8,
                         ls='--', alpha=0.6, zorder=2)

        axin.axhline(0.0, color='k', lw=0.8)
        axin.tick_params(labelsize=6)
        axin.tick_params(axis='x', which='both', bottom=True, labelbottom=False)

    return r


# ── Figure: fractal_loglog — 4 electrodes × 4 columns ─────────────────────────

def fig_fractal_loglog(results):
    """
    4×4 log-log scatter: rows = electrodes, columns = [Higuchi-δR, Higuchi-δX, RS-δR, RS-δX].
    Each panel: averaged log-log curve + 3 fits + residual inset.
    """
    COLS = [
        ('higuchi', 'dR', COLOR_DR),
        ('higuchi', 'dX', COLOR_DX),
        ('rs',      'dR', COLOR_DR),
        ('rs',      'dX', COLOR_DX),
    ]
    COL_HEADERS = [
        r'Higuchi — $\delta R$',
        r'Higuchi — $\delta X$',
        r'R/S — $\delta R$',
        r'R/S — $\delta X$',
    ]
    N_ROWS = len(FOCUS_STEMS)
    N_COLS = len(COLS)

    fig, axes = plt.subplots(
        N_ROWS, N_COLS,
        figsize=(16, 14),
        constrained_layout=True,
    )

    for row, stem in enumerate(FOCUS_STEMS):
        label      = FOCUS_LABELS[stem]
        elec_color = FOCUS_COLORS[stem]

        for col, (meth, sig_key, sig_color) in enumerate(COLS):
            ax  = axes[row, col]
            key = (stem, meth, sig_key)

            if key not in results:
                ax.set_title(f'{label} / {meth} / {sig_key} — missing', fontsize=7)
                continue

            r = results[key]
            _fractal_loglog_panel(ax, r, sig_color, meth, show_inset=True)

            print(f'  [{stem}/{meth}/{sig_key}] n_win={r["n_win"]}, '
                  f'w={r["w_size"]}, '
                  f'df={r["df_single_mean"]:.4f}±{r["df_single_std"]:.4f}')

            if row == 0:
                ax.set_title(COL_HEADERS[col], fontsize=9, fontweight='bold')
            if col == 0:
                ax.set_ylabel(label, fontsize=8, color=elec_color, fontweight='bold')

    fig.suptitle(
        r'Fractal fit log-log scatter — $\delta R$ (red) and $\delta X$ (green)'
        '\nInset: residuals log\u2081\u2080(y) \u2212 fitted line',
        fontsize=10)
    _save(fig, 'rv2_fractal_loglog.png')


# ── Figure: fractal_windowed — per-window Higuchi, dR and dX ─────────────────

def fig_fractal_windowed(npz_exp):
    """
    Per-window Higuchi log-log fits for each electrode and each signal component.

    Layout: 4 rows (electrodes) × 2 columns (δR left, δX right).
    Each panel overlays N_WINDOWS equal non-overlapping segments coloured by tab10.
    Legend encodes time window bounds and per-window df.
    A residual inset (lower-left) shows deviations from the global single-slope fit.
    """
    from fractal_dim import estimate_fd
    from math import log10

    N_WINDOWS = 8

    if npz_exp is None:
        print('[fractal_windowed] phase1_experimental.npz not found — skipping.')
        return

    n_rows = len(FOCUS_STEMS)
    fig, axes = plt.subplots(
        n_rows, 2,
        figsize=(16, 5 * n_rows),
        constrained_layout=True,
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    components  = [('dR', r'$\delta R$'), ('dX', r'$\delta X$')]
    colors_w    = [plt.cm.tab10(i) for i in range(N_WINDOWS)]

    for row, stem in enumerate(FOCUS_STEMS):
        label   = FOCUS_LABELS[stem]
        arrs    = _get_stem_arrays(npz_exp, stem, ['dR', 'dX', 't'])
        dR_full = arrs['dR']
        dX_full = arrs['dX']
        t_full  = arrs['t']

        if dR_full is None or t_full is None:
            for col in range(2):
                axes[row, col].set_title(f'{label} — data missing', fontsize=8)
            continue

        dt = float(np.median(np.diff(t_full)))

        for col, (comp_key, comp_label) in enumerate(components):
            ax  = axes[row, col]
            sig = dR_full if comp_key == 'dR' else dX_full
            if sig is None:
                ax.set_title(f'{label} / {comp_label} — missing', fontsize=8)
                continue

            N       = len(sig)
            seg_len = N // N_WINDOWS

            # Per-window estimates
            window_results = []
            df_vals        = []

            for i in range(N_WINDOWS):
                i0  = i * seg_len
                i1  = i0 + seg_len if i < N_WINDOWS - 1 else N
                seg = sig[i0:i1]

                t_start_h = float(t_full[i0]) / 3600.0
                t_end_h   = float(t_full[min(i1 - 1, len(t_full) - 1)]) / 3600.0

                if len(seg) < 10:
                    window_results.append(None)
                    continue

                print(f'  [{stem}/{comp_key}] window {i + 1}/{N_WINDOWS} ...')
                res = estimate_fd(seg, dt, 'higuchi')

                x_fit_w    = res['x_fit']
                y_fit_w    = res['y_fit']
                slope_w    = res['slope']
                intercept_w= res['intercept']
                df_w       = res['df']
                df_err_w   = res['df_err']

                if not (np.isfinite(slope_w) and np.isfinite(intercept_w)
                        and len(x_fit_w) >= 2):
                    window_results.append(None)
                    continue

                window_results.append(dict(
                    x_fit=x_fit_w, y_fit=y_fit_w,
                    slope=slope_w, intercept=intercept_w,
                    df=df_w, df_err=df_err_w,
                    t_start_h=t_start_h, t_end_h=t_end_h,
                ))
                if np.isfinite(df_w):
                    df_vals.append(df_w)
                    if df_w > 2.0 or df_w < 1.0:
                        print(f'  [WARN] {stem}/{comp_key} window {i + 1}: '
                              f'df={df_w:.3f} outside [1, 2]')

            # Global fit on full signal
            print(f'  [{stem}/{comp_key}] global fit ...')
            res_full     = estimate_fd(sig, dt, 'higuchi')
            slope_global = res_full['slope']
            int_global   = res_full['intercept']
            global_valid = np.isfinite(slope_global) and np.isfinite(int_global)

            # Plot scatter + fitted lines per window
            for i, wr in enumerate(window_results):
                if wr is None:
                    continue
                c      = colors_w[i]
                ax.scatter(wr['x_fit'], wr['y_fit'], s=2, alpha=0.25, color=c)
                x_line = np.logspace(log10(wr['x_fit'].min()),
                                     log10(wr['x_fit'].max()), 100)
                y_line = 10 ** (wr['intercept'] + wr['slope'] * np.log10(x_line))
                ax.plot(x_line, y_line, lw=1.2, color=c, alpha=0.9)

            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('Lag [samples]', fontsize=8)
            ax.set_ylabel('L(k)', fontsize=8)
            ax.tick_params(labelsize=7)
            ax.set_title(f'{label}  —  {comp_label}', fontsize=8)

            # Legend
            mean_df = float(np.mean(df_vals)) if df_vals else np.nan
            std_df  = float(np.std(df_vals))  if df_vals else np.nan

            legend_handles = []
            for i, wr in enumerate(window_results):
                if wr is None:
                    lbl = f'W{i + 1} [\u2013]  $d_f=?$'
                else:
                    lbl = (f'W{i + 1} [{wr["t_start_h"]:.2f}\u2013{wr["t_end_h"]:.2f} h]'
                           f'  $d_f={wr["df"]:.3f}\\pm{wr["df_err"]:.3f}$')
                legend_handles.append(mpatches.Patch(color=colors_w[i], label=lbl))

            mean_line = mlines.Line2D(
                [], [], color='black', lw=1.2,
                label=f'Mean: $d_f={mean_df:.3f}\\pm{std_df:.3f}$',
            )
            legend_handles.append(mean_line)
            ax.legend(handles=legend_handles, fontsize=5.5, loc='upper right',
                      framealpha=0.85, ncol=1)

            # Residual inset (lower-left)
            if global_valid:
                ax_ins = ax.inset_axes([0.04, 0.04, 0.32, 0.32])
                for i, wr in enumerate(window_results):
                    if wr is None:
                        continue
                    log_x = np.log10(wr['x_fit'])
                    log_y = np.log10(wr['y_fit'])
                    resid = log_y - (slope_global * log_x + int_global)
                    ax_ins.scatter(log_x, resid, s=1.5, alpha=0.4,
                                   color=colors_w[i])
                ax_ins.tick_params(labelsize=5)

            ax.text(0.5, 0.02,
                    f'Mean $d_f = {mean_df:.3f} \\pm {std_df:.3f}$'
                    f' (std over {N_WINDOWS} windows)',
                    transform=ax.transAxes,
                    ha='center', va='bottom', fontsize=7)

    fig.suptitle(
        f'Higuchi log-log fits per temporal window  ({N_WINDOWS} segments)',
        fontsize=10)
    _save(fig, 'rv2_fractal_windowed.png')


# ── Figure: fractal_reduced — 1×4, both signals overlaid ─────────────────────

def fig_fractal_reduced(results, method='higuchi'):
    """
    1×4 horizontal panels (one per electrode), both δR and δX overlaid.

    δR: tab:red, circle scatter, solid fast/slow lines, dashed single-slope.
    δX: tab:green, 'x' scatter, dashed fast/slow lines, dotted single-slope.
    Combined legend shows single/fast/slow for each signal.
    """
    stems = FOCUS_STEMS

    fig, axes = plt.subplots(1, len(stems), figsize=(18, 5), constrained_layout=True)

    for col, stem in enumerate(stems):
        ax    = axes[col]
        label = FOCUS_LABELS[stem]
        color = FOCUS_COLORS[stem]

        key_dr = (stem, method, 'dR')
        key_dx = (stem, method, 'dX')

        has_dr = key_dr in results
        has_dx = key_dx in results

        handles = []

        show_bp = (method != 'rs')

        if has_dr:
            r_dr = results[key_dr]
            h_s, h_f, h_sl, bp = _draw_signal_curves(
                ax, r_dr, COLOR_DR, marker='o',
                ls_fast='-', ls_slow='-', ls_single='--',
                black_single=False, show_bp=show_bp)
            h_s.set_label(r'$\delta R$ ' + h_s.get_label())
            dr_handles = [h_s]
            if show_bp:
                h_f.set_label(r'$\delta R$ ' + h_f.get_label())
                h_sl.set_label(r'$\delta R$ ' + h_sl.get_label())
                dr_handles += [h_f, h_sl]
            handles += dr_handles

        if has_dx:
            r_dx = results[key_dx]
            h_s, h_f, h_sl, bp = _draw_signal_curves(
                ax, r_dx, COLOR_DX, marker='x',
                ls_fast='--', ls_slow='--', ls_single=':',
                black_single=False, show_bp=show_bp)
            h_s.set_label(r'$\delta X$ ' + h_s.get_label())
            dx_handles = [h_s]
            if show_bp:
                h_f.set_label(r'$\delta X$ ' + h_f.get_label())
                h_sl.set_label(r'$\delta X$ ' + h_sl.get_label())
                dx_handles += [h_f, h_sl]
            handles += dx_handles

        if not has_dr and not has_dx:
            ax.set_title(f'{label} — data missing', fontsize=8, color=color)
            continue

        # Residual marker handles — appended to the legend so reader can cross-reference
        # with the inset.  dR uses circles ('o'), dX uses squares ('s').
        if has_dr:
            handles.append(mlines.Line2D([], [], color=COLOR_DR, marker='o',
                                         linestyle='none', markersize=4,
                                         label=r'$\delta R$ (resid.)'))
        if has_dx:
            handles.append(mlines.Line2D([], [], color=COLOR_DX, marker='s',
                                         linestyle='none', markersize=4,
                                         label=r'$\delta X$ (resid.)'))

        ax.legend(handles=handles, fontsize=5.5, loc='upper right', framealpha=0.85)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.tick_params(labelsize=7)
        ax.grid(True, which='both', alpha=0.18)
        ax.set_title(label, fontsize=9, fontweight='bold', color=color)

        # ── Residual inset ─────────────────────────────────────────────────────
        inset_pos = (0.1, 0.65, 0.32, 0.32) if not show_bp else (0.1, 0.05, 0.32, 0.32)
        axin = ax.inset_axes(inset_pos)

        x_break_line = np.nan   # will hold one representative breakpoint for the vline

        # resid_marker matches the legend handle above: 'o' for dR, 's' for dX
        for sig_color, r_sig, resid_marker in [
                (COLOR_DR, results.get(key_dr), 'o'),
                (COLOR_DX, results.get(key_dx), 's')]:
            if r_sig is None:
                continue
            rgb_s         = mcolors.to_rgb(sig_color)
            color_light_s = tuple(c + (1.0 - c) * 0.55 for c in rgb_s)

            x_arr_s  = r_sig['x_arr']
            mean_y_s = r_sig['mean_y']
            lx_s     = np.log10(x_arr_s)
            resid_s  = np.log10(mean_y_s) - (r_sig['slope_avg'] * lx_s
                                              + r_sig['intercept_avg'])
            axin.scatter(lx_s, resid_s, s=8, marker=resid_marker,
                         color=sig_color, alpha=0.35, linewidths=0.5, zorder=2)

            if show_bp:
                idx_brk = r_sig['idx_break']
                x_brk   = r_sig['x_break']
                bp_s = (idx_brk is not None
                        and np.isfinite(x_brk)
                        and not np.isnan(r_sig['df_fast_mean']))
                if bp_s:
                    lx_f  = np.log10(x_arr_s[:idx_brk])
                    lx_sl = np.log10(x_arr_s[idx_brk:])
                    resid_f  = (np.log10(mean_y_s[:idx_brk])
                                - (r_sig['slope_fast_avg'] * lx_f
                                   + r_sig['intercept_fast_avg']))
                    resid_sl = (np.log10(mean_y_s[idx_brk:])
                                - (r_sig['slope_slow_avg'] * lx_sl
                                   + r_sig['intercept_slow_avg']))
                    axin.scatter(lx_f,  resid_f,  s=8, marker=resid_marker,
                                 color=sig_color, alpha=0.75,
                                 linewidths=0.5, zorder=3)
                    axin.scatter(lx_sl, resid_sl, s=8, marker=resid_marker,
                                 color=color_light_s, alpha=0.75,
                                 linewidths=0.5, zorder=3)
                    if np.isnan(x_break_line):
                        x_break_line = x_brk   # use first valid breakpoint for vline

        if show_bp and np.isfinite(x_break_line):
            axin.axvline(np.log10(x_break_line), color='grey', lw=0.8,
                         ls='--', alpha=0.6, zorder=2)
        axin.axhline(0.0, color='k', lw=0.8)
        axin.tick_params(labelsize=6)
        axin.tick_params(axis='x', which='both', bottom=True, labelbottom=False)

        if has_dr:
            print(f'  [{stem}/{method}/dR] n_win={results[key_dr]["n_win"]}, '
                  f'df={results[key_dr]["df_single_mean"]:.4f}±'
                  f'{results[key_dr]["df_single_std"]:.4f}')
        if has_dx:
            print(f'  [{stem}/{method}/dX] n_win={results[key_dx]["n_win"]}, '
                  f'df={results[key_dx]["df_single_mean"]:.4f}±'
                  f'{results[key_dx]["df_single_std"]:.4f}')

    meth_lbl = METHOD_LABELS.get(method, method.upper())
    fig.suptitle(
        f'Fractal dimension — {meth_lbl}'
        r'  ($\delta R$ red circles / solid, $\delta X$ green crosses / dashed)',
        fontsize=11)
    _save(fig, 'rv2_fractal_reduced.png')


# ── LaTeX table ────────────────────────────────────────────────────────────────

def print_fractal_table(results):
    """
    Print a LaTeX booktabs table to stdout.

    Column structure (asymmetric):
        Signal (δR / δX) → Algorithm → Sub-columns
        Higuchi: Single, Fast, Slow  (3 sub-columns)
        R/S:     Single              (1 sub-column — no breakpoint analysis)
    Format: compact $1.87 (3)$ notation.
    """
    SIGNALS    = [('dR', r'$\delta R$'), ('dX', r'$\delta X$')]
    # Each method: (key, label, [sub-column names])
    METHS = [
        ('higuchi', 'Higuchi', ['Single', 'Fast', 'Slow']),
        ('rs',      'R/S',     ['Single']),
    ]
    ROW_ORDER  = ['eb04', 'eb11', 'eb13', 'eb15']
    ROW_LABELS = {
        'eb04': 'Seeding',
        'eb11': 'Confluent',
        'eb13': 'Wound healing',
        'eb15': 'Post-healing',
    }
    NL = '\n'

    # Number of data columns per signal (sum of sub-column counts across methods)
    cols_per_sig = sum(len(subs) for _, _, subs in METHS)   # 3 + 1 = 4
    n_data       = len(SIGNALS) * cols_per_sig               # 2 × 4 = 8

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
    for _, sig_lbl in SIGNALS:
        parts.append(r'\multicolumn{' + str(cols_per_sig) + r'}{c}{' + sig_lbl + '}')
    lines.append(' & '.join(parts) + r' \\')

    # cmidrule for signal groups
    sig_cmidrules = []
    for si in range(len(SIGNALS)):
        start = 2 + si * cols_per_sig
        end   = start + cols_per_sig - 1
        sig_cmidrules.append(r'\cmidrule(lr){' + f'{start}-{end}' + '}')
    lines.append(' '.join(sig_cmidrules))

    # Row 2: algorithm groupings (with correct spans)
    parts2 = ['']
    for _ in SIGNALS:
        for _, meth_lbl, subs in METHS:
            n = len(subs)
            if n == 1:
                parts2.append(meth_lbl)
            else:
                parts2.append(r'\multicolumn{' + str(n) + r'}{c}{' + meth_lbl + '}')
    lines.append(' & '.join(parts2) + r' \\')

    # cmidrule for algorithm groups (only those with >1 sub-column get rules)
    alg_cmidrules = []
    col_cursor = 2
    for _ in SIGNALS:
        for _, _, subs in METHS:
            n = len(subs)
            if n > 1:
                alg_cmidrules.append(
                    r'\cmidrule(lr){' + f'{col_cursor}-{col_cursor + n - 1}' + '}')
            col_cursor += n
    if alg_cmidrules:
        lines.append(' '.join(alg_cmidrules))

    # Row 3: individual sub-column labels
    parts3 = ['']
    for _ in SIGNALS:
        for _, _, subs in METHS:
            parts3 += subs
    lines.append(' & '.join(parts3) + r' \\')
    lines.append(r'\midrule')

    # Data rows
    for stem in ROW_ORDER:
        row_parts = [ROW_LABELS[stem]]
        for sig_key, _ in SIGNALS:
            for meth_key, _, subs in METHS:
                key = (stem, meth_key, sig_key)
                if key not in results:
                    row_parts += ['---'] * len(subs)
                    continue
                v = results[key]
                for sub in subs:
                    if sub == 'Single':
                        row_parts.append(_fmt_cell(v['df_single_mean'], v['df_single_std']))
                    elif sub == 'Fast':
                        row_parts.append(_fmt_cell(v['df_fast_mean'],   v['df_fast_std']))
                    elif sub == 'Slow':
                        row_parts.append(_fmt_cell(v['df_slow_mean'],   v['df_slow_std']))
        lines.append(' & '.join(row_parts) + r' \\')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')

    print()
    print('=' * 70)
    print('LaTeX fractal table:')
    print('=' * 70)
    print(NL.join(lines))


# ── Reconstruction windowed analysis ──────────────────────────────────────────

def compute_reconstruction_results(npz_exp, version='v3'):
    """
    Run windowed_fractal_analysis on reconstruction signals.

    Loads results/reconstruct_{version}_{stem}.npz (keys dR_rec, dX_rec).
    Uses the same dt as the experimental signal.  Because the reconstruction
    has the same number of samples as the experiment, windowed_fractal_analysis
    naturally produces the same n_win / w_size window partition.

    Parameters
    ----------
    npz_exp : loaded experimental npz (for dt extraction)
    version : 'v3' (greedy) or 'v4' (Viterbi)

    Returns
    -------
    dict keyed by (stem, method, signal_key) — same format as compute_all_results().
    """
    from fractal_dim import windowed_fractal_analysis

    results = {}
    for stem in FOCUS_STEMS:
        rec_path = os.path.join(RESULTS, f'reconstruct_{version}_{stem}.npz')
        if not os.path.isfile(rec_path):
            print(f'[WARN] {rec_path} not found — skipping {stem}.')
            continue

        arrs = _get_stem_arrays(npz_exp, stem, ['t'])
        t    = arrs['t']
        if t is None:
            print(f'[WARN] {stem}: time array not found — skipping.')
            continue
        dt = float(np.median(np.diff(t)))

        rec_npz = np.load(rec_path, allow_pickle=False)

        for sig_key, rec_key in [('dR', 'dR_rec'), ('dX', 'dX_rec')]:
            if rec_key not in rec_npz.files:
                print(f'[WARN] {rec_path}: key {rec_key!r} not found — skipping.')
                continue
            sig = rec_npz[rec_key].astype(float)

            for meth in METHODS:
                print(f'  [{version}/{stem}/{meth}/{sig_key}] ...', end=' ', flush=True)
                r = windowed_fractal_analysis(sig, dt, meth)
                results[(stem, meth, sig_key)] = r
                print(f"df_single={r['df_single_mean']:.4f}±{r['df_single_std']:.4f}")

    return results


# ── Figure: fractal_comparison — Exp / Greedy / Viterbi ───────────────────────

def fig_fractal_comparison(results_exp, results_v3=None, results_v4=None):
    """
    4 rows (electrodes) × 6 columns comparison of fractal dimension.

    Columns: [Higuchi/Exp, Higuchi/Greedy, Higuchi/Viterbi,
               RS/Exp,      RS/Greedy,      RS/Viterbi]

    Each panel overlays δR (red circles) and δX (green crosses) with a
    single-slope windowed fit (no breakpoint analysis).  Error in legend
    is windowed std.  Reuses _draw_signal_curves with show_bp=False.
    """
    VERSIONS = [
        ('exp', 'Experimental', results_exp),
        ('v3',  'Greedy',       results_v3),
        ('v4',  'Viterbi',      results_v4),
    ]
    COLS = []
    COL_HEADERS = []
    for meth_key, meth_lbl in [('higuchi', 'Higuchi'), ('rs', 'R/S')]:
        for ver_key, ver_lbl, _ in VERSIONS:
            COLS.append((meth_key, ver_key))
            COL_HEADERS.append(f'{meth_lbl}\n{ver_lbl}')

    N_ROWS = len(FOCUS_STEMS)
    N_COLS = len(COLS)   # 6

    ver_res = {vk: vr for vk, _, vr in VERSIONS}

    fig, axes = plt.subplots(
        N_ROWS, N_COLS,
        figsize=(22, 14),
        constrained_layout=True,
    )

    for row, stem in enumerate(FOCUS_STEMS):
        label      = FOCUS_LABELS[stem]
        elec_color = FOCUS_COLORS[stem]

        for col, (meth, ver_key) in enumerate(COLS):
            ax  = axes[row, col]
            res = ver_res[ver_key]

            if res is None:
                ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes,
                        ha='center', va='center', fontsize=8, color='grey')
                ax.set_axis_off()
                if row == 0:
                    ax.set_title(COL_HEADERS[col], fontsize=7, fontweight='bold')
                continue

            handles  = []
            has_any  = False

            for sig_color, sig_key, resid_marker, ls_s in [
                    (COLOR_DR, 'dR', 'o', '--'),
                    (COLOR_DX, 'dX', 'x', ':')]:
                key = (stem, meth, sig_key)
                if key not in res:
                    continue
                r = res[key]
                h_s, _, _, _ = _draw_signal_curves(
                    ax, r, sig_color,
                    marker=resid_marker,
                    ls_single=ls_s,
                    black_single=False,
                    show_bp=False)
                sig_lbl = (r'$\delta R$' if sig_key == 'dR' else r'$\delta X$')
                h_s.set_label(sig_lbl + ' ' + h_s.get_label())
                handles.append(h_s)
                has_any = True

            if not has_any:
                ax.text(0.5, 0.5, 'missing', transform=ax.transAxes,
                        ha='center', va='center', fontsize=7, color='grey')
                ax.set_axis_off()
            else:
                leg_loc = 'lower right' if meth == 'rs' else 'upper right'
                ax.legend(handles=handles, fontsize=5.5, loc=leg_loc,
                          framealpha=0.85)
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.tick_params(labelsize=6)
                ax.grid(True, which='both', alpha=0.18)

            if row == 0:
                ax.set_title(COL_HEADERS[col], fontsize=7, fontweight='bold')
            if col == 0:
                ax.set_ylabel(label, fontsize=7, color=elec_color,
                              fontweight='bold')

            print(f'  [{stem}/{meth}/{ver_key}] done')

    fig.suptitle(
        r'Fractal dimension — Experimental vs Greedy vs Viterbi'
        '\n'
        r'$\delta R$ red circles / dashed,  $\delta X$ green crosses / dotted'
        '\nWindowed single-slope (mean $\pm$ windowed std)',
        fontsize=10)
    _save(fig, 'rv2_fractal_comparison.png')


# ── LaTeX comparison table ─────────────────────────────────────────────────────

def print_comparison_table(results_exp, results_v3=None, results_v4=None):
    """
    LaTeX table: df_single_mean (std) for all electrodes × signals × methods × versions.

    Column structure:
        Signal (δR / δX) → Method (Higuchi / R/S) → Version (Exp / Greedy / Viterbi)
    Format: compact $1.87 (3)$ notation.
    """
    VER_INFO = [
        ('exp', 'Exp.',    results_exp),
        ('v3',  'Greedy',  results_v3),
        ('v4',  'Viterbi', results_v4),
    ]
    METH_INFO = [('higuchi', 'Higuchi'), ('rs', 'R/S')]
    SIG_INFO  = [('dR', r'$\delta R$'), ('dX', r'$\delta X$')]
    ROW_ORDER = ['eb04', 'eb11', 'eb13', 'eb15']
    ROW_LABELS = {
        'eb04': 'Seeding',
        'eb11': 'Confluent',
        'eb13': 'Wound healing',
        'eb15': 'Post-healing',
    }
    NL = '\n'

    n_ver        = len(VER_INFO)                      # 3
    n_meth       = len(METH_INFO)                     # 2
    cols_per_sig = n_meth * n_ver                     # 6
    n_data       = len(SIG_INFO) * cols_per_sig       # 12

    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'\centering')
    lines.append(
        r'\caption{Fractal dimension single-slope $d_f$ — experimental and '
        r'reconstructed signals ($\delta R$ and $\delta X$). '
        r'Values are mean with windowed std (1 sig.\ digit) in parentheses. '
        r'Greedy: reconstruct\_v3; Viterbi: reconstruct\_v4 '
        r'(eb15 Viterbi not available).}'
    )
    lines.append(r'\label{tab:fractal_comparison}')
    lines.append(r'\scriptsize')
    lines.append(r'\begin{tabular}{l' + 'c' * n_data + '}')
    lines.append(r'\toprule')

    # Row 1: signal groupings
    parts = ['']
    for _, sig_lbl in SIG_INFO:
        parts.append(r'\multicolumn{' + str(cols_per_sig) + r'}{c}{' + sig_lbl + '}')
    lines.append(' & '.join(parts) + r' \\')

    sig_cmidrules = []
    for si in range(len(SIG_INFO)):
        start = 2 + si * cols_per_sig
        end   = start + cols_per_sig - 1
        sig_cmidrules.append(r'\cmidrule(lr){' + f'{start}-{end}' + '}')
    lines.append(' '.join(sig_cmidrules))

    # Row 2: method groupings
    parts2 = ['']
    for _ in SIG_INFO:
        for _, meth_lbl in METH_INFO:
            parts2.append(r'\multicolumn{' + str(n_ver) + r'}{c}{' + meth_lbl + '}')
    lines.append(' & '.join(parts2) + r' \\')

    alg_cmidrules = []
    col_cursor = 2
    for _ in SIG_INFO:
        for _ in METH_INFO:
            alg_cmidrules.append(
                r'\cmidrule(lr){' + f'{col_cursor}-{col_cursor + n_ver - 1}' + '}')
            col_cursor += n_ver
    lines.append(' '.join(alg_cmidrules))

    # Row 3: version labels
    parts3 = ['']
    for _ in range(len(SIG_INFO) * n_meth):
        parts3 += [vl for _, vl, _ in VER_INFO]
    lines.append(' & '.join(parts3) + r' \\')
    lines.append(r'\midrule')

    # Data rows
    ver_res = {vk: vr for vk, _, vr in VER_INFO}
    for stem in ROW_ORDER:
        row_parts = [ROW_LABELS[stem]]
        for sig_key, _ in SIG_INFO:
            for meth_key, _ in METH_INFO:
                for ver_key, _, _ in VER_INFO:
                    res = ver_res[ver_key]
                    key = (stem, meth_key, sig_key)
                    if res is None or key not in res:
                        row_parts.append('---')
                        continue
                    v = res[key]
                    row_parts.append(_fmt_cell(v['df_single_mean'], v['df_single_std']))
        lines.append(' & '.join(row_parts) + r' \\')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')

    print()
    print('=' * 70)
    print('LaTeX comparison table:')
    print('=' * 70)
    print(NL.join(lines))


# ── Main ───────────────────────────────────────────────────────────────────────

FIGURE_KEYS = [
    'fractal_loglog', 'fractal_windowed', 'fractal_reduced', 'table',
    'fractal_comparison', 'comparison_table',
]


def main():
    parser = argparse.ArgumentParser(
        description='Generate fractal dimension figures and LaTeX table.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Available keys: ' + '  '.join(FIGURE_KEYS),
    )
    parser.add_argument(
        'keys', nargs='*',
        help='Figure/table keys to generate (omit for all).',
    )
    parser.add_argument(
        '--method',
        default='higuchi',
        help='Method for fractal_reduced (higuchi|rs). Default: higuchi.',
    )
    args = parser.parse_args()

    keys = args.keys if args.keys else FIGURE_KEYS

    for k in keys:
        if k not in FIGURE_KEYS:
            print(f'Unknown key: {k!r}.  Available: {", ".join(FIGURE_KEYS)}')
            sys.exit(1)

    npz_exp = _load_exp_npz()
    if npz_exp is None:
        print('[ERROR] phase1_experimental.npz not found — cannot proceed.')
        sys.exit(1)

    # Experimental windowed results — needed by most figures
    needs_exp = any(k in keys for k in [
        'fractal_loglog', 'fractal_reduced', 'table',
        'fractal_comparison', 'comparison_table'])
    results_exp = {}
    if needs_exp:
        print('\n=== Computing windowed_fractal_analysis — experimental ===')
        results_exp = compute_all_results(npz_exp)

    # Reconstruction results — only when comparison figures/table are requested
    needs_rec = any(k in keys for k in ['fractal_comparison', 'comparison_table'])
    results_v3 = results_v4 = None
    if needs_rec:
        print('\n=== Computing windowed_fractal_analysis — Greedy (v3) ===')
        results_v3 = compute_reconstruction_results(npz_exp, version='v3')
        print('\n=== Computing windowed_fractal_analysis — Viterbi (v4) ===')
        results_v4 = compute_reconstruction_results(npz_exp, version='v4')

    for key in keys:
        print(f'\n=== {key} ===')
        if key == 'fractal_loglog':
            fig_fractal_loglog(results_exp)
        elif key == 'fractal_windowed':
            fig_fractal_windowed(npz_exp)
        elif key == 'fractal_reduced':
            fig_fractal_reduced(results_exp, method=args.method)
        elif key == 'table':
            print_fractal_table(results_exp)
        elif key == 'fractal_comparison':
            fig_fractal_comparison(results_exp, results_v3, results_v4)
        elif key == 'comparison_table':
            print_comparison_table(results_exp, results_v3, results_v4)


if __name__ == '__main__':
    main()
