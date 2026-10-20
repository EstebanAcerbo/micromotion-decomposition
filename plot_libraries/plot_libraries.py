"""
plot_libraries.py  —  Diagnostic figures for all perturbation libraries
=========================================================================

Layout (2 rows, figsize 18×10):

  Row 0:  [A] Full culture map      [B] Zoom (no legend)      [C] Params text
  Row 1:  [D] |ΔV/V₀|_max @ 1kHz   [E] |dZ|/|Z₀| [%] spectrum

GridSpec: 2 rows × 6 columns, with:
  A = gs[0, 0:2]   B = gs[0, 2:4]   C = gs[0, 4:6]
  D = gs[1, 0:2]   E = gs[1, 2:6]

Zoom rule
---------
  Libraries A / A_ext   : right electrode-edge, ±3.5 cells wide
  Libraries B / C       : grid-centre, ZOOM_CELLS × ZOOM_CELLS

FD solves per library call
--------------------------
  2 × single-frequency FD solves at 1 kHz (baseline + perturbed) for panel D.
  No additional solves required (spectrum comes from pre-computed .npz files).

Run from Micromotion_Simul/:
    PYTHONIOENCODING=utf-8 python plot_libraries.py
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
from matplotlib.patches import Circle

rcParams.update({'font.size': 9, 'font.family': 'sans-serif',
                 'mathtext.fontset': 'stix'})

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'simulation_engine'))

from engine import (
    build_geometry, substrate_offset, build_l_corner_culture,
    build_random_size_corner_culture, build_sparse_corner_culture,
    apply_brick_offset, solve_full, FREQS, SystemConst,
)
from SimulSheet_Funcs import Map_Electrode   # path set by engine import

RESULTS    = os.path.join(_HERE, '..', 'results')
os.makedirs(RESULTS, exist_ok=True)

ZOOM_CELLS = 6      # half-side of default cell-centre zoom (shows 6×6 cells)
F_1K       = 1000.0 # Hz — frequency for ΔV panel


# ── small helpers ─────────────────────────────────────────────────────────────

def _sub_norm(Sub):
    return (Sub != 0).astype(np.int8)


def _combined(Cult, Sub):
    return (2 * Cult.astype(np.int32) + _sub_norm(Sub).astype(np.int32))


def _solve_dV(geo, Cult, Sub):
    """
    Two single-frequency FD solves at F_1K.
    Returns (V_base_flat, V_pert_flat), each shape (dim,) complex.
    """
    f1 = np.array([F_1K])
    _, V_b, _ = solve_full(geo, f1)
    _, V_p, _ = solve_full(geo, f1, Cult=Cult, Sub=Sub)
    return V_b[0], V_p[0]


def _electrode_edge_zoom(geo):
    """
    xlim / ylim centred on the right electrode boundary, ±3.5 cells wide.
    Used for Library A (electrode shift) where the perturbation is at the edge.
    """
    de_um   = geo['p']['de'] * 1e4                              # µm
    cell_um = geo['stencil'].spacing_x * geo['d'] * 1e4         # µm / cell
    pad     = 3.5 * cell_um
    return (de_um / 2 - pad, de_um / 2 + pad), (-pad, pad)


def _cell_centre_zoom(geo):
    """
    xlim / ylim centred on the grid centre, ZOOM_CELLS × ZOOM_CELLS.
    Default for Libraries B and C where perturbation is in cell body.
    """
    cell_um = geo['stencil'].spacing_x * geo['d'] * 1e4
    half    = ZOOM_CELLS / 2 * cell_um
    return (-half, half), (-half, half)


# ── panel drawing functions ───────────────────────────────────────────────────

def _draw_full_map(ax, Cult, Sub, geo):
    p, d, side = geo['p'], geo['d'], geo['side']
    Map_Electrode(ax, _combined(Cult, Sub), p['de'], d, side)
    ax.set_title('Culture map', fontsize=9)


def _draw_zoom(ax, Cult, Sub, geo, xlim, ylim):
    """
    Same as full map but cropped to xlim / ylim, legend removed.
    """
    p, d, side = geo['p'], geo['d'], geo['side']
    Map_Electrode(ax, _combined(Cult, Sub), p['de'], d, side)

    # Crop (overrides Map_Electrode's own xlim/ylim)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    # Remove legend — it is already shown in the full map panel
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()

    # Compact, 5-tick axes
    for lims, setter_t, setter_l in [
        (xlim, ax.set_xticks, ax.set_xticklabels),
        (ylim, ax.set_yticks, ax.set_yticklabels),
    ]:
        ticks = np.linspace(lims[0], lims[1], 5)
        setter_t(ticks)
        setter_l([f'{t:.0f}' for t in ticks], fontsize=7)

    ax.set_xlabel(r'Position [$\mu$m]', fontsize=8)
    ax.set_ylabel(r'Position [$\mu$m]', fontsize=8)
    ax.set_title('Zoom', fontsize=9)


def _draw_dV(ax, V_b_flat, V_p_flat, geo):
    d, side = geo['d'], geo['side']
    de_um   = geo['p']['de'] * 1e4

    V_b2d  = V_b_flat.reshape((side, side), order='F')
    V_p2d  = V_p_flat.reshape((side, side), order='F')
    dV_pct = 100.0 * np.abs(V_p2d - V_b2d) / (np.abs(V_b2d).max() + 1e-30)

    half_um = 1e4 * side * d / 2.0
    extent  = [-half_um, half_um, -half_um, half_um]
    vmax    = float(dV_pct.max()) or 1.0

    im = ax.imshow(dV_pct, origin='lower', extent=extent,
                   cmap='inferno', vmin=0, vmax=vmax, aspect='equal')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label=r'$|\Delta V|/|V_0|_{\max}$ [%]')

    circ = Circle((0, 0), de_um / 2, linestyle='--',
                  color='white', fill=False, linewidth=1.2)
    ax.add_patch(circ)

    ax.set_xlabel(r'Position [$\mu$m]', fontsize=8)
    ax.set_ylabel(r'Position [$\mu$m]', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title(fr'$|\Delta V/V_0|$ @ {F_1K:.0f} Hz', fontsize=9)


def _draw_spectrum(ax, Z_base, Z_lib_rows, freqs, labels, cmap_name,
                   shade=False):
    """
    |dZ|/|Z₀| [%] vs frequency.
      shade=False  →  one coloured line per mode   (deterministic)
      shade=True   →  shaded min–max band + median (random)
    """
    Z0 = np.abs(Z_base)

    # Experimental reference frequencies
    ax.axvline(1e3,  color='#888888', ls=':',  lw=0.9, alpha=0.8, label='1 kHz')
    ax.axvline(64e3, color='#b22222', ls='--', lw=0.9, alpha=0.8, label='64 kHz')

    if shade:
        pct_all = np.array([100.0 * np.abs(Z - Z_base) / Z0 for Z in Z_lib_rows])
        lo, hi, med = pct_all.min(0), pct_all.max(0), np.median(pct_all, 0)
        n    = len(Z_lib_rows)
        cmap = plt.get_cmap(cmap_name, max(n, 2))
        for i, pct in enumerate(pct_all):
            ax.semilogx(freqs, pct, color=cmap(i / max(n - 1, 1)),
                        lw=0.7, alpha=0.30)
        ax.fill_between(freqs, lo, hi, alpha=0.20,
                        color='steelblue', label='Min–max band')
        ax.semilogx(freqs, med, color='steelblue', lw=2.2, label='Median')
    else:
        n    = max(len(Z_lib_rows), 2)
        cmap = plt.get_cmap(cmap_name, n)
        lss  = ['--', ':', '-.', '-']
        mks  = ['o', 'D', 's', '^']
        for i, (Z, lbl) in enumerate(zip(Z_lib_rows, labels)):
            pct = 100.0 * np.abs(Z - Z_base) / Z0
            ax.semilogx(freqs, pct,
                        color=cmap(i / max(n - 1, 1)),
                        linestyle=lss[i % 4], marker=mks[i % 4],
                        markersize=4, lw=1.4, label=lbl, alpha=0.9)

    ax.set_xlabel('Frequency [Hz]', fontsize=8)
    ax.set_ylabel(r'$|\Delta Z|/|Z_0|$ [%]', fontsize=8)
    ax.set_title('Perturbation spectrum', fontsize=9)
    ax.grid(True, which='both', alpha=0.20)
    ax.tick_params(labelsize=7)
    #if len(Z_lib_rows) <= 12 or shade:
    ax.legend(fontsize=10, ncol=4, loc='best')


def _draw_params(ax, geo, extra_lines):
    p    = geo['p']
    d_nm = geo['d'] * 1e7
    sx   = geo['stencil'].spacing_x
    lines = [
        r'$\alpha$' + f' = {p["alpha"].real:.2f} (Ohm*cm)^0.5',
        f'Rb = {abs(p["Rb"]):.2f} Ohm*cm2',
        f'Cm = {abs(p["Cm"])*1e6:.3f} uF/cm2',
        '',
        f'de = {p["de"]*1e4:.0f} um',
        f'ac = {p["ac"]*1e8:.2f} um2',
        f'n={p["n"]}  d={d_nm:.1f} nm/px',
        f'max_lw = {sx-2} px',
        '',
    ] + (extra_lines or [])
    ax.set_axis_off()
    ax.text(0.05, 0.97, '\n'.join(lines),
            transform=ax.transAxes,
            fontfamily='monospace', fontsize=7.5,
            va='top', ha='left',
            bbox=dict(boxstyle='round', facecolor='white',
                      alpha=0.92, edgecolor='#aaaaaa'))


# ── unified 2-row figure builder ──────────────────────────────────────────────

def _draw_phase_space(ax, Z_base_full, Z_lib_rows, freqs, labels,
                      cmap_name, freq_target=1000.0):
    """
    Scatter plot of all library modes in (dR, dX) space at freq_target.
    Each dot = one mode, coloured by mode index.
    """
    i_f = int(np.argmin(np.abs(freqs - freq_target)))
    Z0  = Z_base_full[i_f]
    dR  = np.array([z[i_f].real - Z0.real for z in Z_lib_rows])
    dX  = np.array([z[i_f].imag - Z0.imag for z in Z_lib_rows])

    n    = max(len(Z_lib_rows), 2)
    cmap = plt.get_cmap(cmap_name, n)
    sc   = ax.scatter(dR, dX,
                      c=np.arange(n), cmap=cmap_name,
                      s=14, alpha=0.75, zorder=3, edgecolors='none')

    # Mark extreme modes with larger markers
    if n > 1:
        for ext in [np.argmax(np.hypot(dR, dX)), np.argmin(np.hypot(dR, dX))]:
            ax.scatter(dR[ext], dX[ext], color='k', s=40, zorder=5,
                       edgecolors='k', linewidths=0.8, marker='D')

    ax.axhline(0, color='#aaaaaa', lw=0.6)
    ax.axvline(0, color='#aaaaaa', lw=0.6)
    ax.scatter([0], [0], color='k', s=25, zorder=6, marker='x')

    ax.set_xlabel(r'$\Delta R$ [$\Omega$]', fontsize=8)
    ax.set_ylabel(r'$\Delta X$ [$\Omega$]', fontsize=8)
    ax.set_title(fr'Phase space $(\Delta R,\, \Delta X)$ @ {freq_target:.0f} Hz',
                 fontsize=9)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.20)

    cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('Mode index', fontsize=7)
    cb.ax.tick_params(labelsize=6)


def _make_figure(title, Cult, Sub, geo,
                 Z_base_full, Z_lib_rows, freqs, labels,
                 cmap_name,
                 extra_params=None,
                 shade_spectrum=False,
                 zoom_xlim=None, zoom_ylim=None):
    """
    Row 0: [Full map | Zoom (no legend) | Params text]
    Row 1: [Phase space (dR,dX)         | Spectrum    ]

    GridSpec: 2 rows × 6 cols
      gs[0,0:2] Full map    gs[0,2:4] Zoom    gs[0,4:6] Params
      gs[1,0:2] Phase space gs[1,2:6] Spectrum
    """
    if zoom_xlim is None or zoom_ylim is None:
        zoom_xlim, zoom_ylim = _cell_centre_zoom(geo)

    fig = plt.figure(figsize=(18, 10), constrained_layout=True)
    gs  = gridspec.GridSpec(2, 6, figure=fig,
                            height_ratios=[1, 1],
                            hspace=0.05, wspace=0.10)

    ax_full   = fig.add_subplot(gs[0, 0:2])
    ax_zoom   = fig.add_subplot(gs[0, 2:4])
    ax_params = fig.add_subplot(gs[0, 4:6])
    ax_ps     = fig.add_subplot(gs[1, 0:2])
    ax_spec   = fig.add_subplot(gs[1, 2:6])

    _draw_full_map(ax_full,  Cult, Sub, geo)
    _draw_zoom(ax_zoom,      Cult, Sub, geo, zoom_xlim, zoom_ylim)
    _draw_phase_space(ax_ps, Z_base_full, Z_lib_rows, freqs, labels, cmap_name)
    _draw_spectrum(ax_spec,  Z_base_full, Z_lib_rows, freqs, labels,
                  cmap_name, shade=shade_spectrum)
    _draw_params(ax_params, geo, extra_params)

    fig.suptitle(title, fontsize=10)
    return fig


def _save(fig, name):
    out = os.path.join(RESULTS, name)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    try:
        print(f'  saved -> {out}')
    except UnicodeEncodeError:
        print(f'  saved -> {name}')


# ─────────────────────────────────────────────────────────────────────────────
# BASELINE
# ─────────────────────────────────────────────────────────────────────────────

def plot_baseline(geo):
    print('[baseline] ...')
    Cult   = geo['Culture_vector']
    Sub    = geo['Substrate_vector']
    Z_base = np.load(os.path.join(RESULTS, 'baseline.npz'))['Z'].astype(complex)

    i1k = np.argmin(np.abs(FREQS - 1000))
    extra = [
        'Configuration: confluent',
        'No perturbation applied',
        '',
        f'|Z0| @ 1kHz = {abs(Z_base[i1k]):.1f} Ohm',
    ]
    fig = _make_figure(
        'Baseline — Confluent MDCKII',
        Cult, Sub, geo,
        Z_base, [Z_base], FREQS, ['Baseline'],
        cmap_name='jet', extra_params=extra,
    )
    _save(fig, 'plot_baseline.png')


# ─────────────────────────────────────────────────────────────────────────────
# LIBRARY A: electrode shift (lower-triangle format)
# ─────────────────────────────────────────────────────────────────────────────

def plot_lib_A(geo):
    print('[Library A shift] ...')
    npz     = np.load(os.path.join(RESULTS, 'library_A_shift.npz'))
    dx_px   = npz['dx_px'].astype(int)   # 1-D, length N_modes
    dy_px   = npz['dy_px'].astype(int)   # 1-D, length N_modes
    Z_lib   = npz['Z']                   # (N_modes, n_freq)
    Z_base  = npz['Z_baseline']
    freqs   = npz['freqs']
    max_shift = int(npz['max_shift'])

    N_modes = Z_lib.shape[0]

    dx_rep, dy_rep = 3, 0
    Sub_rep = substrate_offset(geo, x_px=dx_rep, y_px=dy_rep)
    Cult    = geo['Culture_vector']

    # Select representative modes by searching (dx_px, dy_px) pairs
    target_modes = [
        (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0),(7, 0), (8, 0), (9, 0), (10, 0), (11, 0),
        (12, 0), (13, 0), (14, 0), (15, 0), (16, 0),(17, 0), (18, 0), (19, 0), (20, 0), (21, 0),
        (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6),(7, 7), (8, 8), (9, 9), (10, 10), (11, 11),
        (12, 12), (13, 13), (14, 14), (15, 15), (16, 16), (17, 17), (18, 18), (19, 19), (20, 20), (21, 21)
    ]
    sel = []
    for dx, dy in target_modes:
        mask = (dx_px == dx) & (dy_px == dy)
        idx  = np.where(mask)[0]
        if len(idx) > 0:
            sel.append((int(idx[0]), f'({dx},{dy})'))

    i1k = np.argmin(np.abs(freqs - 1000))

    # Representative mode for params box
    rep_mask = (dx_px == dx_rep) & (dy_px == dy_rep)
    rep_idx  = np.where(rep_mask)[0]
    if len(rep_idx) > 0:
        dZ_rep  = Z_lib[rep_idx[0], i1k] - Z_base[i1k]
        pct_rep = 100 * abs(dZ_rep) / abs(Z_base[i1k])
    else:
        dZ_rep  = 0.0
        pct_rep = 0.0

    extra = [
        'Library A — electrode shift',
        f'N_modes = {N_modes}',
        f'max_shift = {max_shift} px',
        f'dx range: [0, {max_shift}] px',
        f'dy range: [0, dx] px',
        '',
        f'Rep: dx={dx_rep}, dy={dy_rep} px',
        f'|Z0| @ 1kHz = {abs(Z_base[i1k]):.1f} Ohm',
        f'|dZ| @ 1kHz = {abs(dZ_rep):.3f} Ohm',
        f'Peak dZ/Z0  = {pct_rep:.4f}%',
    ]
    # Zoom to right electrode boundary: perturbation is at the electrode edge
    zx, zy = _electrode_edge_zoom(geo)
    fig = _make_figure(
        f'Library A — Electrode Shift  (rep: dx={dx_rep}, dy={dy_rep} px)',
        Cult, Sub_rep, geo,
        Z_base, [Z_lib[s[0]] for s in sel], freqs, [s[1] for s in sel],
        cmap_name='jet', extra_params=extra,
        zoom_xlim=zx, zoom_ylim=zy,
    )
    _save(fig, 'plot_lib_A.png')


# ─────────────────────────────────────────────────────────────────────────────
# LIBRARY B_det
# ─────────────────────────────────────────────────────────────────────────────

def plot_lib_B_det(geo):
    print('[Library B_det] ...')
    npz     = np.load(os.path.join(RESULTS, 'library_B_det.npz'))
    lw_vals = npz['lw_vals']
    Z_lib   = npz['Z']
    Z_base  = npz['Z_baseline']
    freqs   = npz['freqs']

    lw_rep    = 4
    geo_lw    = {**geo, 'p': {**geo['p'], 'l_width': lw_rep}}
    Cult_r, _ = build_l_corner_culture(geo_lw, corner=1, all_cells=True)
    Sub       = geo['Substrate_vector']

    i1k     = np.argmin(np.abs(freqs - 1000))
    ilw     = int(np.searchsorted(lw_vals, lw_rep))
    dZ_rep  = Z_lib[ilw, i1k] - Z_base[i1k]
    pct_rep = 100 * abs(dZ_rep) / abs(Z_base[i1k])
    pct_all = 100 * np.abs(Z_lib[ilw] - Z_base) / np.abs(Z_base)
    f_peak  = freqs[np.argmax(pct_all)]

    extra = [
        'Library B_det — UR corner, fixed Rb',
        f'lw: {int(lw_vals[0])}-->{int(lw_vals[-1])} px  ({len(lw_vals)} modes)',
        f'arm px = 2*lw - 1',
        '',
        f'Rep: lw={lw_rep}  arm={2*lw_rep-1}px',
        f'|Z0| @ 1kHz = {abs(Z_base[i1k]):.1f} Ohm',
        f'|dZ| @ 1kHz = {abs(dZ_rep):.3f} Ohm',
        f'Rep dZ/Z0   = {pct_rep:.4f}%',
        f'Rep f_peak  = {f_peak:.0f} Hz',
    ]
    fig = _make_figure(
        f'Library B_det — L-corner UR  (rep: lw={lw_rep})',
        Cult_r, Sub, geo_lw,
        Z_base, list(Z_lib), freqs, [f'lw={lw}' for lw in lw_vals],
        cmap_name='jet', extra_params=extra,
    )
    _save(fig, 'plot_lib_B_det.png')


# ─────────────────────────────────────────────────────────────────────────────
# LIBRARY B_rand
# ─────────────────────────────────────────────────────────────────────────────

def plot_lib_B_rand(geo):
    print('[Library B_rand] ...')
    npz      = np.load(os.path.join(RESULTS, 'library_B_rand.npz'), allow_pickle=True)
    Z_lib    = npz['Z']
    Z_base   = npz['Z_baseline']
    freqs    = npz['freqs']
    seeds    = npz['seeds']
    lw_means = npz['lw_means']
    mu_lw    = float(npz['mu_lw'])
    sigma_lw = float(npz['sigma_lw'])
    max_lw   = float(npz['max_lw'])

    seed_rep = int(seeds[0])
    Cv_rep, _, corners, lws = build_random_size_corner_culture(
        geo, mu_lw=mu_lw, sigma_lw=sigma_lw, seed=seed_rep)
    Sub = geo['Substrate_vector']

    i1k  = np.argmin(np.abs(freqs - 1000))
    pcts = 100 * np.abs(Z_lib[:, i1k] - Z_base[i1k]) / abs(Z_base[i1k])
    live = corners >= 0
    lw_m = float(lws[live].mean())

    extra = [
        'Library B_rand — random corner+N',
        f'mu_lw={mu_lw:.1f}  sigma={sigma_lw:.1f}',
        f'max_lw={int(max_lw)}  N_seeds={len(seeds)}',
        '',
        f'Rep: seed={seed_rep}',
        f'  lw_mean={lw_m:.1f}',
        f'|Z0| @ 1kHz = {abs(Z_base[i1k]):.1f} Ohm',
        f'Median dZ/Z0 @ 1kHz = {np.median(pcts):.4f}%',
        f'Max    dZ/Z0 @ 1kHz = {pcts.max():.4f}%',
    ]
    fig = _make_figure(
        f'Library B_rand — Random Corner + N({mu_lw:.0f},{sigma_lw:.0f}) arm  '
        f'(rep: seed={seed_rep})',
        Cv_rep, Sub, geo,
        Z_base, list(Z_lib), freqs,
        [f'seed={int(s)} lw={m:.1f}' for s, m in zip(seeds, lw_means)],
        cmap_name='jet', extra_params=extra,
        shade_spectrum=True,
    )
    _save(fig, 'plot_lib_B_rand.png')


# ─────────────────────────────────────────────────────────────────────────────
# LIBRARY B_rb
# ─────────────────────────────────────────────────────────────────────────────

def plot_lib_B_rb(geo):
    print('[Library B_rb] ...')
    npz      = np.load(os.path.join(RESULTS, 'library_B_rb.npz'))
    k_rb_arr = npz['k_rb_arr']           # (N_modes,) float
    lw_arr   = npz['lw_arr'].astype(int) # (N_modes,) int
    Rb_vals  = npz['Rb_vals']            # (N_modes,) float
    k_rb_vals= npz['k_rb_vals']          # distinct K_RB values swept
    Z_lib    = npz['Z']                  # (N_modes, n_freq)
    Z_base   = npz['Z_baseline']
    freqs    = npz['freqs']
    Rb_base  = float(npz['Rb_base'])

    # Extract K_RB=1.0 slice for _make_figure (preserves legacy plot structure)
    k_rep  = 1.0
    mask_k = np.isclose(k_rb_arr, k_rep)
    Z_k1   = Z_lib[mask_k]               # (max_lw, n_freq)
    lw_k1  = lw_arr[mask_k]
    Rb_k1  = Rb_vals[mask_k]

    lw_rep    = 4
    ilw       = int(np.searchsorted(lw_k1, lw_rep))
    Rb_rep    = float(Rb_k1[ilw])
    geo_lw    = {**geo, 'p': {**geo['p'], 'l_width': lw_rep, 'Rb': complex(Rb_rep)}}
    Cult_r, _ = build_l_corner_culture(geo_lw, corner=1, all_cells=True)
    Sub       = geo['Substrate_vector']

    i1k     = np.argmin(np.abs(freqs - 1000))
    dZ_rep  = Z_k1[ilw, i1k] - Z_base[i1k]
    pct_rep = 100 * abs(dZ_rep) / abs(Z_base[i1k])

    extra = [
        'Library B_rb — UR corner + Rb co-scaling',
        'Rb(lw,K) = Rb_base*(1 + K*lw/max_lw)',
        f'Rb_base={Rb_base:.2f}  K_RB_VALS={list(k_rb_vals)}',
        f'Total modes: {len(k_rb_arr)}  ({len(k_rb_vals)} K_RB x {len(lw_k1)} lw)',
        f'Shown: K_RB={k_rep} slice only',
        '',
        f'Rep: lw={lw_rep}  Rb={Rb_rep:.2f}',
        f'|Z0| @ 1kHz = {abs(Z_base[i1k]):.1f} Ohm',
        f'|dZ| @ 1kHz = {abs(dZ_rep):.3f} Ohm',
        f'Peak dZ/Z0  = {pct_rep:.4f}%',
    ]
    fig = _make_figure(
        f'Library B_rb — K_RB={k_rep} slice  (rep: lw={lw_rep}, Rb={Rb_rep:.2f} Ohm*cm2)',
        Cult_r, Sub, geo_lw,
        Z_base, list(Z_k1), freqs,
        [f'lw={lw} Rb={rb:.2f}' for lw, rb in zip(lw_k1, Rb_k1)],
        cmap_name='jet', extra_params=extra,
    )
    _save(fig, 'plot_lib_B_rb.png')


# ─────────────────────────────────────────────────────────────────────────────
# LIBRARY B_micro
# ─────────────────────────────────────────────────────────────────────────────

def plot_lib_B_micro(geo):
    print('[Library B_micro] ...')
    npz      = np.load(os.path.join(RESULTS, 'library_B_micro.npz'), allow_pickle=True)
    Z_lib    = npz['Z']            # (N_SEEDS, n_freq)
    Z_base   = npz['Z_baseline']
    freqs    = npz['freqs']
    seeds    = npz['seeds']
    lw_means = npz['lw_means']    # mean lw among PERTURBED cells per seed
    lw_stds  = npz['lw_stds']
    n_pert   = npz['n_perturbed']  # number of cells that received a corner
    lw_max   = int(npz['lw_max'])
    probs    = npz['probs']

    # Representative realisation: seed 0
    seed_rep = int(seeds[0])
    Cv_rep, _, corner_states, lw_states = build_sparse_corner_culture(
        geo, lw_max=lw_max, probs=list(probs), seed=seed_rep)
    Sub = geo['Substrate_vector']

    i1k  = np.argmin(np.abs(freqs - 1000))
    pcts = 100 * np.abs(Z_lib[:, i1k] - Z_base[i1k]) / abs(Z_base[i1k])

    live_perturbed = (corner_states >= 0) & (lw_states > 0)
    lw_m = float(lw_states[live_perturbed].mean()) if live_perturbed.any() else 0.0
    n_p  = int(live_perturbed.sum())
    n_total = int((corner_states >= 0).sum())   # live cells

    extra = [
        'Library B_micro — sparse small-L',
        f'lw in {{0,1,2}}  probs={[f"{p:.2f}" for p in probs]}',
        f'lw_max={lw_max}  N_seeds={len(seeds)}',
        '',
        f'Rep: seed={seed_rep}',
        f'  n_perturbed={n_p}/{n_total}  ({100*n_p/max(n_total,1):.0f}%)',
        f'  lw_mean (perturbed)={lw_m:.2f}',
        f'|Z0| @ 1kHz = {abs(Z_base[i1k]):.1f} Ohm',
        f'Median dZ/Z0 @ 1kHz = {np.median(pcts):.4f}%',
        f'Max    dZ/Z0 @ 1kHz = {pcts.max():.4f}%',
    ]
    fig = _make_figure(
        f'Library B_micro — Sparse L-corner  lw∈{{0,1,2}}  (rep: seed={seed_rep})',
        Cv_rep, Sub, geo,
        Z_base, list(Z_lib), freqs,
        [f'seed={int(s)} n={int(n)} lw={m:.1f}'
         for s, n, m in zip(seeds, n_pert, lw_means)],
        cmap_name='jet', extra_params=extra,
        shade_spectrum=True,
    )
    _save(fig, 'plot_lib_B_micro.png')


# ─────────────────────────────────────────────────────────────────────────────
# LIBRARY C: brick-wall
# ─────────────────────────────────────────────────────────────────────────────

def plot_lib_C(geo):
    print('[Library C brick-wall] ...')
    npz    = np.load(os.path.join(RESULTS, 'library_C_brickwall.npz'))
    deltas = npz['delta']
    off_px = npz['offset_px']
    Z_lib  = npz['Z']
    Z_base = npz['Z_baseline']
    freqs  = npz['freqs']

    delta_rep = 0.50
    id_rep    = int(np.argmin(np.abs(deltas - delta_rep)))
    delta_rep = float(deltas[id_rep])
    off_rep   = int(off_px[id_rep])

    Cult_r = apply_brick_offset(geo, delta_rep)
    Sub    = geo['Substrate_vector']

    i1k     = np.argmin(np.abs(freqs - 1000))
    dZ_rep  = Z_lib[id_rep, i1k] - Z_base[i1k]
    pct_rep = 100 * abs(dZ_rep) / abs(Z_base[i1k])
    d_nm    = geo['d'] * 1e7
    off_max = int(np.abs(off_px).max())
    pct_all = 100 * np.abs(Z_lib - Z_base[np.newaxis, :]) / np.abs(Z_base[np.newaxis, :])
    peak_f  = freqs[np.argmax(pct_all.max(axis=0))]

    extra = [
        'Library C — brick-wall offset',
        f'delta: {deltas[0]:.2f}-->{deltas[-1]:.2f}',
        f'N_modes: {len(deltas)}',
        f'offset range: +/-{off_max}px = +/-{off_max*d_nm:.0f}nm',
        '',
        f'Rep: delta={delta_rep:.2f}  off={off_rep}px',
        f'|Z0| @ 1kHz = {abs(Z_base[i1k]):.1f} Ohm',
        f'Rep dZ/Z0 @ 1kHz = {pct_rep:.4f}%',
        f'Peak dZ/Z0 @ {peak_f:.0f}Hz = {pct_all.max():.4f}%',
    ]
    print([f'delta={d:.2f} ({o}px)' for d, o in zip(deltas, off_px)])
    fig = _make_figure(
        f'Library C — Brick-Wall Offset  (rep: delta={delta_rep:.2f}, off={off_rep}px)',
        Cult_r, Sub, geo,
        Z_base, list(Z_lib), freqs,
        [f'delta={d:.2f} ({o}px)' for d, o in zip(deltas, off_px)],
        cmap_name='jet', extra_params=extra,
    )
    _save(fig, 'plot_lib_C.png')


# ─────────────────────────────────────────────────────────────────────────────
# LIBRARY D_rb: pure Rb sweep (no geometry change)
# ─────────────────────────────────────────────────────────────────────────────

def plot_lib_D(freq_target=1000.0):
    """Library D_rb — 3-parameter (alpha, Rb, Cm) small perturbation sweep.

    Left panel : normalised |dZ(f)|/|Z0(f)| spectra for the 8 sign-corner modes
                 (each parameter at its maximum amplitude ±0.006, all 2^3 sign combos).
    Right panel: phase-space scatter (dR, dX) at freq_target for all 216 modes,
                 coloured by delta_Rb value.
    """
    print('[Library D_rb] ...')
    npz_path = os.path.join(RESULTS, 'library_D_rb.npz')
    if not os.path.exists(npz_path):
        print('  WARNING: results/library_D_rb.npz not found — skipping plot_lib_D')
        return

    npz         = np.load(npz_path)
    d_alpha_arr = npz['delta_alpha']          # (N_modes,)
    d_Rb_arr    = npz['delta_Rb']             # (N_modes,)
    d_Cm_arr    = npz['delta_Cm']             # (N_modes,)
    delta_vals  = npz['delta_vals']           # (6,)
    alpha_base  = float(npz['alpha_base'])
    Rb_base     = float(npz['Rb_base'])
    Cm_base     = float(npz['Cm_base'])
    freqs       = npz['freqs']                # (n_freq,)
    Z_lib       = npz['Z']                    # (N_modes, n_freq) complex
    Z_base      = npz['Z_baseline']           # (n_freq,) complex

    N_modes = Z_lib.shape[0]
    i_tgt   = int(np.argmin(np.abs(freqs - freq_target)))
    f_tgt   = freqs[i_tgt]

    # Normalised dZ spectra: |dZ(f)| / |Z0(f)| in percent
    dZ_all   = Z_lib - Z_base[np.newaxis, :]           # (N_modes, n_freq)
    Z0_mag   = np.abs(Z_base)                           # (n_freq,)
    dZ_norm  = np.abs(dZ_all) / Z0_mag[np.newaxis, :] * 100.0  # percent

    # dZ at target frequency
    dZ_tgt   = dZ_all[:, i_tgt]
    dR_tgt   = dZ_tgt.real
    dX_tgt   = dZ_tgt.imag

    # ── Select 8 sign-corner modes (each param at ±max amplitude) ────────────
    delta_max = float(np.max(np.abs(delta_vals)))       # 0.006
    sign_combos = [
        (+1, +1, +1), (+1, +1, -1), (+1, -1, +1), (+1, -1, -1),
        (-1, +1, +1), (-1, +1, -1), (-1, -1, +1), (-1, -1, -1),
    ]
    corner_colors = [
        '#e41a1c', '#ff7f00', '#4daf4a', '#377eb8',
        '#984ea3', '#a65628', '#f781bf', '#999999',
    ]
    corner_indices = []
    for sa, sr, sc_ in sign_combos:
        target_a = sa * delta_max
        target_r = sr * delta_max
        target_c = sc_ * delta_max
        dist = (d_alpha_arr - target_a)**2 + (d_Rb_arr - target_r)**2 + (d_Cm_arr - target_c)**2
        corner_indices.append(int(np.argmin(dist)))

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, (ax_spec, ax_scat) = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left: normalised spectra for 8 sign-corner modes ─────────────────────
    for idx, (sa, sr, sc_), color in zip(corner_indices, sign_combos, corner_colors):
        sign_str = (f'α{"+" if sa>0 else "−"} '
                    f'Rb{"+" if sr>0 else "−"} '
                    f'Cm{"+" if sc_>0 else "−"}')
        ax_spec.semilogx(freqs, dZ_norm[idx], color=color, lw=1.4,
                         label=sign_str)

    ax_spec.set_xlabel('Frequency [Hz]', fontsize=8)
    ax_spec.set_ylabel(r'$|\Delta Z(f)| / |Z_0(f)|$ [%]', fontsize=8)
    ax_spec.set_title(
        fr'Normalised $|\Delta Z|$ spectra — 8 sign-corner modes'
        fr'  ($\delta={{\pm}}{delta_max*100:.1f}\%$)',
        fontsize=9)
    ax_spec.tick_params(labelsize=7)
    ax_spec.grid(True, which='both', alpha=0.20)
    ax_spec.legend(fontsize=7, loc='best', ncol=2,
                   title=r'$\delta\alpha\ \delta R_b\ \delta C_m$ sign', title_fontsize=7)

    # ── Right: phase-space scatter coloured by delta_Rb ──────────────────────
    d_rb_max = float(np.max(np.abs(d_Rb_arr)))
    norm_rb  = matplotlib.colors.Normalize(vmin=-d_rb_max, vmax=d_rb_max)
    cmap_rb  = plt.get_cmap('RdBu_r')

    sc_plot = ax_scat.scatter(dR_tgt, dX_tgt,
                               c=d_Rb_arr, cmap=cmap_rb, norm=norm_rb,
                               s=18, zorder=3, alpha=0.75,
                               edgecolors='none')

    # Overlay the 8 corner modes with larger markers
    for idx, color in zip(corner_indices, corner_colors):
        ax_scat.scatter(dR_tgt[idx], dX_tgt[idx],
                        color=color, s=70, zorder=5,
                        edgecolors='k', linewidths=0.6)

    ax_scat.axhline(0, color='#aaaaaa', lw=0.6)
    ax_scat.axvline(0, color='#aaaaaa', lw=0.6)
    ax_scat.scatter([0], [0], color='k', s=30, zorder=6, marker='x')

    cb = fig.colorbar(sc_plot, ax=ax_scat, fraction=0.046, pad=0.04)
    cb.set_label(r'$\delta R_b$ (fractional)', fontsize=8)
    cb.ax.tick_params(labelsize=7)

    ax_scat.set_xlabel(r'$\Delta R$ [$\Omega$]', fontsize=8)
    ax_scat.set_ylabel(r'$\Delta X$ [$\Omega$]', fontsize=8)
    ax_scat.tick_params(labelsize=7)
    ax_scat.set_title(
        fr'Phase space ($\Delta R$, $\Delta X$) @ {f_tgt:.0f} Hz — all {N_modes} modes',
        fontsize=9)
    ax_scat.grid(True, alpha=0.20)

    fig.suptitle(
        fr'Library D — 3-param perturbation  '
        fr'($\alpha_0={alpha_base:.1f}$, '
        fr'$R_{{b,0}}={Rb_base:.2f}\ \Omega\cdot\mathrm{{cm}}^2$, '
        fr'$C_{{m,0}}={Cm_base:.2e}\ \mathrm{{F/cm}}^2$,  '
        fr'$\delta \in [{delta_vals[0]*100:.1f}\%,\ {delta_vals[-1]*100:.1f}\%]$,  '
        fr'N={N_modes})',
        fontsize=9,
    )

    print(f'  N_modes={N_modes}  delta_vals={[f"{v*100:.2f}%" for v in delta_vals]}')
    print(f'  alpha_base={alpha_base}  Rb_base={Rb_base}  Cm_base={Cm_base:.3e}')
    print(f'  dR range @ {f_tgt:.0f} Hz: [{dR_tgt.min():.2f}, {dR_tgt.max():.2f}] Ohm')
    print(f'  dX range @ {f_tgt:.0f} Hz: [{dX_tgt.min():.2f}, {dX_tgt.max():.2f}] Ohm')

    plt.tight_layout()
    _save(fig, 'plot_lib_D.png')


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print('Building confluent geometry ...')
    geo = build_geometry()
    sx  = geo['stencil'].spacing_x
    print(
        f'  {geo["n_cols"]}x{geo["n_rows"]} cells  '
        f'side={geo["side"]} px  d={geo["d"]*1e7:.2f} nm/px  max_lw={sx-2}\n'
        f'Output: {RESULTS}\n'
    )

    plot_baseline(geo)
    plot_lib_A(geo)
    plot_lib_B_rb(geo)
    plot_lib_C(geo)
    plot_lib_D()

    print('\nAll plots saved to results/')


if __name__ == '__main__':
    main()
