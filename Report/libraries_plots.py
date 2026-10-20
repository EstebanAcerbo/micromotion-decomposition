"""
libraries_plots.py — Library figure generator (extracted from report_figures_v2.py).
=====================================================================================
Generates the six library-related figures:
    lib_A            — perturbation spectrum + phase space (Library A)
    lib_A_boundary   — dR/dX/|dZ| vs dx for axial and diagonal slices
    lib_B            — culture map zoom + perturbation spectrum (Library B)
    lib_D            — GK sensitivity ±1% + phase space (Library D)
    lib_C            — culture map zoom + perturbation spectrum (Library C)
    coverage         — three-panel phase-space coverage figure

Usage
-----
    python Report/libraries_plots.py              # all figures
    python Report/libraries_plots.py lib_A        # one figure
    python Report/libraries_plots.py lib_B lib_C  # two figures

Available keys: lib_A  lib_A_boundary  lib_B  lib_D  lib_C  coverage
"""

import argparse
import os
import sys

import itertools

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
# This script lives in Micromotion_Simul/Report/; the engines + libraries live
# one directory up.
SCRIPT_DIR        = os.path.dirname(os.path.abspath(__file__))
SCRIPT_DIR_PARENT = os.path.dirname(SCRIPT_DIR)          # Micromotion_Simul/
sys.path.insert(0, SCRIPT_DIR_PARENT)

RESULTS = os.path.join(SCRIPT_DIR_PARENT, 'results')
os.makedirs(RESULTS, exist_ok=True)

# ── Global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size':        12,
    'font.family':      'sans-serif',
    'mathtext.fontset': 'stix',
})

TARGET_FREQ = 1000.0   # Hz — phase-space evaluation frequency

# Biological state labels for the four focus electrodes
FOCUS_STEMS  = ['eb11', 'eb04', 'eb13', 'eb15']
FOCUS_LABELS = {
    'eb04': 'Seeding',
    'eb11': 'Confluent',
    'eb13': 'Wound healing',
    'eb15': 'Post-healing',
}
FOCUS_COLORS = {
    'eb04': '#4CAF50',
    'eb11': '#2196F3',
    'eb13': '#FF9800',
    'eb15': '#E91E63',
}


# Add project root (parent of Report/) to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Add simulation_engine/ explicitly
SIM_ENGINE_DIR = os.path.join(PROJECT_ROOT, 'simulation_engine')
if SIM_ENGINE_DIR not in sys.path:
    sys.path.insert(0, SIM_ENGINE_DIR)

from engine import build_geometry
from SimulSheet_Funcs import Map_Electrode

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_dat_ch(path, ch_select=3):
    """Parse SR530 .dat file; return (freqs, Z_mat) for the requested channel.

    Returns
    -------
    freqs  : (N_frec,) float   – measured frequencies [Hz]
    Z_mat  : (n_sweeps, N_frec) complex – impedance [Ohm] per sweep
    """
    Rs = 99715.0
    with open(path) as fh:
        lines = fh.readlines()

    # Locate FLAG1 → next non-empty line is N_frec
    flag1_idx = next(i for i, ln in enumerate(lines) if ln.strip() == 'FLAG1')
    nf_idx    = next(i for i in range(flag1_idx + 1, len(lines)) if lines[i].strip())
    N_frec    = int(lines[nf_idx].strip())

    # Locate FLAG2 → data starts on next non-empty line (0-based row index)
    flag2_idx  = next(i for i, ln in enumerate(lines) if ln.strip() == 'FLAG2')
    data_start = next(i for i in range(flag2_idx + 1, len(lines)) if lines[i].strip())

    data = np.loadtxt(path, skiprows=data_start)
    # Columns: CH  t[s]  f[Hz]  Amplitude[V]  Phase[deg]

    # CH0 first sweep → reference voltages (one per frequency)
    ch0_rows = data[data[:, 0] == 0][:N_frec]
    freqs    = ch0_rows[:, 2]
    V0       = ch0_rows[:, 3] * np.exp(1j * ch0_rows[:, 4] * np.pi / 180.0)

    # Selected channel
    ch_rows  = data[data[:, 0] == ch_select]
    n_rows   = (len(ch_rows) // N_frec) * N_frec          # trim incomplete sweep
    V_all    = ch_rows[:n_rows, 3] * np.exp(1j * ch_rows[:n_rows, 4] * np.pi / 180.0)
    n_sweeps = n_rows // N_frec
    V_mat    = V_all.reshape(n_sweeps, N_frec)
    Z_mat    = Rs * V_mat / (V0[np.newaxis, :] - V_mat)

    return freqs, Z_mat


def _npz_path(fname):
    return os.path.join(RESULTS, fname)


def _require_npz(fname):
    """Return path or None; print warning if missing."""
    p = _npz_path(fname)
    if not os.path.exists(p):
        print(f'[SKIP] Required file not found: {p}')
        return None
    return p


def _save(fig, name):
    out = os.path.join(RESULTS, name)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved -> {out}')


def _load_exp_npz():
    """Load pre-computed phase1_experimental.npz; return dict or None."""
    p = _require_npz('phase1_experimental.npz')
    if p is None:
        return None
    return np.load(p, allow_pickle=False)


def _get_stem_arrays(npz, stem, keys):
    """Extract arrays for one stem from phase1_experimental.npz."""
    out = {}
    for k in keys:
        arr_key = f'{stem}_{k}'
        if arr_key in npz:
            out[k] = npz[arr_key]
        else:
            out[k] = None
    return out


def _hull_polygon(dR, dX):
    """Return closed hull polygon or None."""
    from scipy.spatial import ConvexHull
    pts = np.column_stack([dR, dX])
    if len(pts) < 3:
        return None
    try:
        h = ConvexHull(pts)
        v = np.vstack([pts[h.vertices], pts[h.vertices[0]]])
        return v
    except Exception:
        return None


def compute_minkowski_hull(point_sets):
    """
    Compute the convex hull of the Minkowski sum of multiple (dR, dX) point sets.

    In the greedy reconstruction, one mode from each library is selected and
    ADDED at every time step.  The set of all achievable combined perturbations
    is the Minkowski sum of the individual library clouds, which is strictly
    larger than any single library hull.

    Each library also admits a null selection (no contribution), so the zero
    vector is included in each library's point set before the hull vertices are
    enumerated.  This ensures the Minkowski sum hull contains every individual
    library cloud as a subset.

    Parameters
    ----------
    point_sets : list of (N_i, 2) array-like
        One array per library; each row is a (dR, dX) mode vector.

    Returns
    -------
    hull_verts : (M, 2) ndarray
        Vertices of the Minkowski sum convex hull in counter-clockwise order
        (open polygon, not closed; use np.vstack([v, v[0]]) for closed plotting).
        Returns None if fewer than 2 non-trivial point sets are provided or if
        the Minkowski sum is degenerate.

    Notes
    -----
    Algorithm:
      1. For each point set, include the zero vector and compute the convex hull.
         Only hull vertices need to be considered because the Minkowski sum of
         two convex sets equals the convex hull of all pairwise vertex sums
         (Minkowski sum theorem for polytopes).
      2. Enumerate all combinations of one vertex from each hull (Cartesian product).
      3. Sum the selected vertices and take the convex hull of all sums.

    Complexity: product of hull sizes across all libraries.  For four libraries
    each with up to ~40 vertices the product is at most 40^4 = 2,560,000 points,
    which is tractable.
    """
    from scipy.spatial import ConvexHull

    if len(point_sets) == 0:
        return None

    # Step 1: compute hull vertices for each library, always including (0, 0).
    hull_vertex_sets = []
    for pts in point_sets:
        pts_arr = np.asarray(pts, dtype=float)
        # Include the zero vector (null selection for this library).
        pts_with_zero = np.vstack([pts_arr, [0.0, 0.0]])
        try:
            hull = ConvexHull(pts_with_zero)
            verts = pts_with_zero[hull.vertices]   # (K_i, 2)
        except Exception:
            # Degenerate (collinear): fall back to using all unique points.
            verts = np.unique(pts_with_zero, axis=0)
        hull_vertex_sets.append(verts)

    # Step 2: enumerate all combinations (Cartesian product) of one vertex per hull
    # and accumulate the vector sums.
    n_combos = 1
    for v in hull_vertex_sets:
        n_combos *= len(v)

    sums = np.zeros((n_combos, 2), dtype=float)
    row = 0
    for combo in itertools.product(*hull_vertex_sets):
        # combo is a tuple of (2,) arrays, one per library.
        sums[row] = sum(c for c in combo)
        row += 1

    # Step 3: convex hull of all sums.
    if len(sums) < 3:
        return None
    try:
        hull_sum = ConvexHull(sums)
        # Return vertices in hull order (CCW), open polygon.
        return sums[hull_sum.vertices]
    except Exception:
        return None


def _in_mink_hull(test_pts, mink_verts):
    """
    Boolean mask: True for each row of test_pts that lies inside (or on the
    boundary of) the polygon defined by mink_verts.

    Uses matplotlib.path.Path for robust point-in-polygon testing.

    Parameters
    ----------
    test_pts   : (N, 2) array
    mink_verts : (M, 2) array — open polygon vertices (not closed)

    Returns
    -------
    mask : (N,) bool array
    """
    if mink_verts is None or len(mink_verts) < 3:
        return np.zeros(len(test_pts), dtype=bool)
    # Close the polygon for Path.
    closed = np.vstack([mink_verts, mink_verts[0]])
    path   = mpath.Path(closed)
    return path.contains_points(test_pts)


def _load_lib_dz(fname, freq=TARGET_FREQ):
    """Load library npz; return (dR, dX, Z_baseline, freqs, npz_dict) or None."""
    p = _require_npz(fname)
    if p is None:
        return None
    npz   = np.load(p, allow_pickle=False)
    freqs = npz['freqs']
    Z     = npz['Z'].astype(complex)
    Zb    = npz['Z_baseline'].astype(complex)
    i_f   = int(np.argmin(np.abs(freqs - freq)))
    n_freq = len(freqs)
    Z_flat = Z.reshape(-1, n_freq)
    dZ     = Z_flat[:, i_f] - Zb[i_f]
    return dZ.real.copy(), dZ.imag.copy(), Zb, freqs, npz


# ─────────────────────────────────────────────────────────────────────────────
# Figure: lib_A
# ─────────────────────────────────────────────────────────────────────────────

def fig_lib_A():
    """
    Library A — perturbation spectrum (left) and boundary slice (right).

    Left panel : |ΔZ|/|Z₀| [%] vs frequency for axial and diagonal shifts,
                 coloured by displacement magnitude (Reds / Blues colormaps).
    Right panel: δR, δX, |δZ| vs dx [µm] for the axial (dy=0) and diagonal
                 (dy=dx) slices at ~924 Hz.
                 Colors: δR = tab:red, δX = tab:green, |δZ| = tab:orange.
    """
    lib_p = _require_npz('library_A_shift.npz')
    bl_p  = _require_npz('baseline.npz')
    if lib_p is None or bl_p is None:
        return

    lib = np.load(lib_p, allow_pickle=True)
    bl  = np.load(bl_p,  allow_pickle=True)

    # ── Shared arrays ─────────────────────────────────────────────────────────
    dx_px    = lib['dx_px'].astype(int)
    dy_px    = lib['dy_px'].astype(int)
    Z_lib    = lib['Z'].astype(complex)
    Z_base   = lib['Z_baseline'].astype(complex)   # embedded baseline
    freqs    = lib['freqs']
    d        = float(lib['d'])                      # cm/px

    Z_baseline = bl['Z'].astype(complex)            # standalone baseline
    side       = int(bl['side'])
    de         = float(bl['de'])

    N_modes  = Z_lib.shape[0]
    n_freq   = len(freqs)
    i_f      = int(np.argmin(np.abs(freqs - TARGET_FREQ)))
    Z0_mag   = np.abs(Z_base)

    # Axial and diagonal masks (shared between both panels)
    mask_ax   = (dy_px == 0) & (dx_px > 0)
    mask_diag = (dy_px == dx_px) & (dx_px > 0)

    idx_ax    = np.where(mask_ax)[0]
    idx_diag  = np.where(mask_diag)[0]
    dx_ax     = dx_px[idx_ax]
    dx_diag   = dx_px[idx_diag]

    order_ax   = np.argsort(dx_ax)
    order_diag = np.argsort(dx_diag)
    idx_ax     = idx_ax[order_ax];   dx_ax   = dx_ax[order_ax]
    idx_diag   = idx_diag[order_diag]; dx_diag = dx_diag[order_diag]

    n_ax   = len(idx_ax)
    n_diag = len(idx_diag)

    # ── Spectrum colormaps and marker-size helpers ────────────────────────────
    cmap_ax   = plt.cm.Reds
    cmap_diag = plt.cm.Blues
    norm_ax   = mcolors.Normalize(vmin=0, vmax=max(dx_ax.max(),   1))
    norm_diag = mcolors.Normalize(vmin=0, vmax=max(dx_diag.max(), 1))

    MS_MAX = 7.0   # marker size at minimum displacement
    MS_MIN = 1.5   # marker size at maximum displacement

    def mode_color_ax(dx):
        return cmap_ax(0.05 + norm_ax(dx))

    def mode_color_diag(dx):
        return cmap_diag(0.05 + norm_diag(dx))

    def marker_size_ax(dx):
        return MS_MAX - (MS_MAX - MS_MIN) * norm_ax(dx)

    def marker_size_diag(dx):
        return MS_MAX - (MS_MAX - MS_MIN) * norm_diag(dx)

    # ── Boundary geometry ─────────────────────────────────────────────────────
    elec_r_px   = (de / 2.0) / d
    dom_cx      = side / 2.0
    d_um_per_px = d * 1e4                           # µm/px

    i_freq = int(np.argmin(np.abs(freqs - 924.0)))

    # Axial slice (boundary panel uses dy=0 without the dx>0 filter)
    mask_ax_bl  = (dy_px == 0)
    idx_ax_bl   = np.where(mask_ax_bl)[0]
    dx_ax_bl    = dx_px[idx_ax_bl]
    order_ax_bl = np.argsort(dx_ax_bl)
    idx_ax_bl   = idx_ax_bl[order_ax_bl]
    dx_ax_bl    = dx_ax_bl[order_ax_bl]
    dx_um_ax    = dx_ax_bl * d_um_per_px

    dZ_ax  = Z_lib[idx_ax_bl, i_freq] - Z_baseline[i_freq]
    dR_ax  = dZ_ax.real
    dX_ax  = dZ_ax.imag
    adZ_ax = np.abs(dZ_ax)

    # Diagonal slice
    idx_diag_bl   = np.where(mask_diag)[0]
    dx_diag_bl    = dx_px[idx_diag_bl]
    order_diag_bl = np.argsort(dx_diag_bl)
    idx_diag_bl   = idx_diag_bl[order_diag_bl]
    dx_diag_bl    = dx_diag_bl[order_diag_bl]
    dx_um_diag    = dx_diag_bl * d_um_per_px

    dZ_diag  = Z_lib[idx_diag_bl, i_freq] - Z_baseline[i_freq]
    dR_diag  = dZ_diag.real
    dX_diag  = dZ_diag.imag
    adZ_diag = np.abs(dZ_diag)

    # ── Build figure ──────────────────────────────────────────────────────────
    fig, (ax_spec, ax_bnd) = plt.subplots(1, 2, figsize=(14, 6),
                                           constrained_layout=True)

    # ── Left: perturbation spectrum ───────────────────────────────────────────
    # Marker spacing: ~10 markers per line across the log-frequency range
    markevery = max(1, n_freq // 10)

    ax_spec.axvline(1e3,  color='#888', ls=':',  lw=0.8, alpha=0.8)
    ax_spec.axvline(64e3, color='#b22', ls=':', lw=0.8, alpha=0.8)

    for i, idx in enumerate(idx_ax):
        pct = 100.0 * np.abs(Z_lib[idx] - Z_base) / Z0_mag
        col = mode_color_ax(dx_ax[i])
        ms  = marker_size_ax(dx_ax[i])
        ax_spec.semilogx(freqs, pct, color=col,
                         lw=0.9, alpha=0.3 + i / 100, ls='-',
                         marker='o', markersize=ms, markevery=markevery,
                         markerfacecolor='none', markeredgecolor=col,
                         markeredgewidth=0.7)

    for i, idx in enumerate(idx_diag):
        pct = 100.0 * np.abs(Z_lib[idx] - Z_base) / Z0_mag
        col = mode_color_diag(dx_diag[i])
        ms  = marker_size_diag(dx_diag[i])
        ax_spec.semilogx(freqs, pct, color=col,
                         lw=0.9, alpha=0.3 + i / 100, ls='-.',
                         marker='^', markersize=ms, markevery=markevery,
                         markerfacecolor='none', markeredgecolor=col,
                         markeredgewidth=0.7)

    h_ax   = ax_spec.plot([], [], color=cmap_ax(0.7),   lw=1.8, ls='-',
                           marker='o', markersize=5, markerfacecolor='none',
                           label=r'Lateral $(dy=0)$')[0]
    h_diag = ax_spec.plot([], [], color=cmap_diag(0.7), lw=1.8, ls='-.',
                           marker='^', markersize=5, markerfacecolor='none',
                           label=r'Diagonal $(dy=dx)$')[0]
    h_1k   = ax_spec.plot([], [], color='#888', ls=':',  lw=0.8,
                           label='1 kHz')[0]
    h_64k  = ax_spec.plot([], [], color='#b22', ls=':',  lw=0.8,
                           label='64 kHz')[0]

    # Encoding annotation entries
    h_col = mlines.Line2D([], [], color=cmap_ax(0.75), lw=2.5, linestyle='-',
                          label=r'colour $\propto$ displacement')
    h_ms  = mlines.Line2D([], [], color='#555', marker='o', linestyle='none',
                          markersize=MS_MAX * 0.7, markerfacecolor='none',
                          markeredgewidth=0.9,
                          label=r'marker size $\propto 1/\mathrm{displacement}$')

    ax_spec.set_xlabel('Frequency [Hz]', fontsize=2+11)
    ax_spec.set_ylabel(r'$|\Delta Z|/|Z_0|$ [%]', fontsize=2+11)
    ax_spec.set_title('Perturbation spectrum — Library A', fontsize=2+11)
    ax_spec.legend(handles=[h_ax, h_diag, h_1k, h_64k, h_col, h_ms],
                   fontsize=2+10, loc='best')
    ax_spec.grid(True, which='both', alpha=0.18)
    ax_spec.tick_params(labelsize=10)

    # ── Right: boundary slice ─────────────────────────────────────────────────
    C_dR  = 'tab:red'
    C_dX  = 'tab:green'
    C_adZ = 'tab:orange'

    ax_bnd.plot(dx_um_ax, dR_ax,  color=C_dR,  lw=1.4, ls='-',  marker='o',
                ms=4, markerfacecolor='white', markeredgecolor=C_dR,
                label=r'$\delta R$  axial')
    ax_bnd.plot(dx_um_ax, dX_ax,  color=C_dX,  lw=1.4, ls='--', marker='o',
                ms=4, markerfacecolor='white', markeredgecolor=C_dX,
                label=r'$\delta X$  axial')
    ax_bnd.plot(dx_um_ax, adZ_ax, color=C_adZ, lw=1.4, ls=':',  marker='o',
                ms=4, markerfacecolor='white', markeredgecolor=C_adZ,
                label=r'$|\delta Z|$  axial')

    ax_bnd.plot(dx_um_diag, dR_diag,  color=C_dR,  lw=1.4, ls='-',  marker='^',
                ms=5, markerfacecolor='white', markeredgecolor=C_dR,
                label=r'$\delta R$  diagonal')
    ax_bnd.plot(dx_um_diag, dX_diag,  color=C_dX,  lw=1.4, ls='--', marker='^',
                ms=5, markerfacecolor='white', markeredgecolor=C_dX,
                label=r'$\delta X$  diagonal')
    ax_bnd.plot(dx_um_diag, adZ_diag, color=C_adZ, lw=1.4, ls=':',  marker='^',
                ms=5, markerfacecolor='white', markeredgecolor=C_adZ,
                label=r'$|\delta Z|$  diagonal')

    ax_bnd.set_xlabel(r'$dx$  [µm]', fontsize=2+11)
    ax_bnd.set_ylabel(r'Impedance component [$\Omega$]', fontsize=2+11)
    ax_bnd.set_title(
        fr'Boundary check — axial $(dy=0)$ and diagonal $(dy=dx)$ slices'
        f'\n(f ≈ {freqs[i_freq]:.1f} Hz; '
        fr'$\delta R$ solid, $\delta X$ dashed, $|\delta Z|$ dotted; '
        r'$\circ$ axial, $\triangle$ diagonal)',
        fontsize=2+11,
    )
    ax_bnd.legend(fontsize=2+10, ncol=1, loc='upper right')
    ax_bnd.tick_params(labelsize=10)
    ax_bnd.grid(True, alpha=0.20)

    fig.suptitle('Library A — Electrode shift perturbations', fontsize=2+13)
    _save(fig, 'rv2_lib_A.png')


# ─────────────────────────────────────────────────────────────────────────────
# Figure: lib_B
# ─────────────────────────────────────────────────────────────────────────────
def fig_lib_B():
    """Library B_rb — culture map zoom + perturbation spectrum."""
    p = _require_npz('library_B_rb.npz')
    if p is None:
        return

    # We need the engine to build the culture geometry for the zoom
    try:
        from engine import (
            build_geometry, build_l_corner_culture,
        )
        from SimulSheet_Funcs import Map_Electrode
        geo_available = True
    except ImportError as exc:
        print(f'[lib_B] engine import failed ({exc}); skipping culture map panel.')
        geo_available = False

    npz      = np.load(p, allow_pickle=False)
    k_rb_arr = npz['k_rb_arr']
    lw_arr   = npz['lw_arr'].astype(int)
    Rb_vals  = npz['Rb_vals']
    Z_lib    = npz['Z'].astype(complex)
    Z_base   = npz['Z_baseline'].astype(complex)
    freqs    = npz['freqs']
    Rb_base  = float(npz['Rb_base'])

    Z0_mag   = np.abs(Z_base)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    ax_map  = axes[0]
    ax_spec = axes[1]

    # ── Left: culture map zoom ────────────────────────────────────────────────
    if geo_available:
        try:
            geo    = build_geometry()
            # Use lw=4, K_RB=1.0 as representative mode
            k_rep  = 1.0
            mask_k = np.isclose(k_rb_arr, k_rep)
            lw_k1  = lw_arr[mask_k]
            Rb_k1  = Rb_vals[mask_k]
            lw_rep  = 4
            ilw     = int(np.searchsorted(lw_k1, lw_rep))
            Rb_rep  = float(Rb_k1[ilw])
            geo_lw  = {**geo, 'p': {**geo['p'], 'l_width': lw_rep,
                                    'Rb': complex(Rb_rep)}}
            Cult_r, _ = build_l_corner_culture(geo_lw, corner=1, all_cells=True)
            Sub       = geo['Substrate_vector']

            # Combined map value
            def _combined(Cult, Sub_):
                sub_n = (Sub_ != 0).astype(np.int8)
                return 2 * Cult.astype(np.int32) + sub_n.astype(np.int32)

            p_geo  = geo['p']
            d_geo  = geo['d']
            side   = geo['side']
            Map_Electrode(ax_map, _combined(Cult_r, Sub), p_geo['de'], d_geo, side)
            # Add simulation_engine/ explicitly
            PLOT_LIB_DIR = os.path.join(PROJECT_ROOT, 'plot_libraries')
            if PLOT_LIB_DIR not in sys.path:
                sys.path.insert(0, PLOT_LIB_DIR)
            # Zoom: cell-centre zoom
            from plot_libraries import ZOOM_CELLS
            cell_um = geo['stencil'].spacing_x * d_geo * 1e4
            half    = ZOOM_CELLS / 2 * cell_um
            ax_map.set_xlim(-half, half)
            ax_map.set_ylim(-half, half)
            leg = ax_map.get_legend()
            if leg is not None:
                leg.remove()
            ax_map.set_title('Culture map zoom (L-corner TJ protrusions)', fontsize=2+11)
            ax_map.set_xlabel(r'Position [µm]', fontsize=2+10)
            ax_map.set_ylabel(r'Position [µm]', fontsize=2+10)
            ax_map.tick_params(labelsize=9)
        except Exception as exc:
            ax_map.set_title(f'Culture map unavailable\n({exc})')
            ax_map.axis('off')
    else:
        ax_map.set_title('Culture map unavailable\n(engine not imported)')
        ax_map.axis('off')

    # ── Right: perturbation spectrum ──────────────────────────────────────────
    ax_spec.axvline(1e3,  color='#888', ls=':',  lw=0.8, alpha=0.8, label='1 kHz')
    ax_spec.axvline(64e3, color='#b22', ls='--', lw=0.8, alpha=0.8, label='64 kHz')

    import colorsys
    k_unique  = np.unique(k_rb_arr)
    lw_unique = np.unique(lw_arr)

    # Discrete tab10 colour per k_rb value
    tab10 = plt.cm.tab10
    k_to_base = {kv: tab10(i % 10) for i, kv in enumerate(k_unique)}

    # Saturation encodes arm length: short arm → desaturated, long arm → full colour
    lw_norm = mcolors.Normalize(vmin=lw_unique.min(), vmax=lw_unique.max())

    def _lib_b_color(kv, lw):
        r, g, b, _ = k_to_base[kv]
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        s_new = 0.20 + 0.80 * float(lw_norm(lw))
        r2, g2, b2 = colorsys.hsv_to_rgb(h, s_new, v)
        return (r2, g2, b2)

    # Show all modes; colour by k_rb (hue) and lw (saturation)
    n_modes = Z_lib.shape[0]
    for i in range(n_modes):
        pct = 100.0 * np.abs(Z_lib[i] - Z_base) / Z0_mag
        col = _lib_b_color(k_rb_arr[i], lw_arr[i])
        ax_spec.semilogx(freqs, pct, color=col, lw=0.6, alpha=0.55)

    # Discrete legend: one entry per k_rb value (shown at max saturation)
    for kv in k_unique:
        r, g, b, _ = k_to_base[kv]
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        r2, g2, b2 = colorsys.hsv_to_rgb(h, 1.0, v)
        ax_spec.plot([], [], color=(r2, g2, b2), lw=2.0,
                     label=fr'$k_{{Rb}}={kv:+.2g}$')
    # Saturation legend for lw (show min and max)
    for lw_v in [lw_unique.min(), lw_unique.max()]:
        r, g, b, _ = k_to_base[k_unique[0]]
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        s_v = 0.20 + 0.80 * float(lw_norm(lw_v))
        r2, g2, b2 = colorsys.hsv_to_rgb(h, s_v, v)
        ax_spec.plot([], [], color=(r2, g2, b2), lw=2.0, ls='-',
                     label=fr'$L_w={lw_v}$ px')

    ax_spec.set_xlabel('Frequency [Hz]', fontsize=2+11)
    ax_spec.set_ylabel(r'$|\Delta Z|/|Z_0|$ [%]', fontsize=2+11)
    ax_spec.set_title('Perturbation spectrum — Library B', fontsize=2+11)
    ax_spec.legend(fontsize=2+10, ncol=2, loc='best')
    ax_spec.grid(True, which='both', alpha=0.18)
    ax_spec.tick_params(labelsize=10)

    fig.suptitle(
        fr'Library B — L-corner junction protrusion + $R_b$ co-scaling'
        f'  (N={n_modes} modes)',
        fontsize=2+13,
    )
    _save(fig, 'rv2_lib_B.png')
# ─────────────────────────────────────────────────────────────────────────────
# Figure: lib_D
# ─────────────────────────────────────────────────────────────────────────────

def fig_lib_D():
    """Library D — GK sensitivity at ±1% + phase space scatter."""
    p = _require_npz('library_D_rb.npz')
    if p is None:
        return

    npz         = np.load(p, allow_pickle=False)
    d_alpha_arr = npz['delta_alpha']
    d_Rb_arr    = npz['delta_Rb']
    d_Cm_arr    = npz['delta_Cm']
    delta_vals  = npz['delta_vals']
    freqs       = npz['freqs']
    Z_lib       = npz['Z'].astype(complex)
    Z_base      = npz['Z_baseline'].astype(complex)

    N_modes = Z_lib.shape[0]
    i_tgt   = int(np.argmin(np.abs(freqs - TARGET_FREQ)))
    Z0_mag  = np.abs(Z_base)

    dZ_all  = Z_lib - Z_base[np.newaxis, :]
    dZ_norm = np.abs(dZ_all) / Z0_mag[np.newaxis, :] * 100.0  # %

    dZ_tgt = dZ_all[:, i_tgt]
    dR_tgt = dZ_tgt.real
    dX_tgt = dZ_tgt.imag

    # Determine ±1% modes from library_D_rb.npz
    # Select modes with |fractional change| closest to 1% for each parameter.
    # delta_alpha/Rb/Cm are fractional deviations.
    PCT_TARGET = 0.01  # 1%
    params_info = [
        ('alpha', d_alpha_arr, r'$\alpha$ +1%', r'$\alpha$ −1%', plt.cm.Reds),
        ('Rb',    d_Rb_arr,    r'$R_b$ +1%',   r'$R_b$ −1%',   plt.cm.Blues),
        ('Cm',    d_Cm_arr,    r'$C_m$ +1%',   r'$C_m$ −1%',   plt.cm.Greens),
    ]

    # Try to load or compute ±1% sensitivity data
    out_1pct = _npz_path('rv2_gk_sensitivity_1pct.npz')
    if os.path.exists(out_1pct):
        d1p = np.load(out_1pct, allow_pickle=False)
        mode_indices_1pct = {k: (int(d1p[f'{k}_idx_pos'].flat[0]),
                                  int(d1p[f'{k}_idx_neg'].flat[0]))
                             for k in ('alpha', 'Rb', 'Cm')}
    else:
        mode_indices_1pct = {}
        save_dict_1pct = {}
        for pname, d_arr, _, _, _ in params_info:
            # Find modes where this parameter is ±1% and others are near 0
            other_names = [x[0] for x in params_info if x[0] != pname]
            d_others = [p_[1] for p_ in params_info if p_[0] != pname]

            # Score: want |d_pname - PCT| small AND |d_others| small
            diff_pos = np.abs(d_arr - PCT_TARGET)
            diff_neg = np.abs(d_arr + PCT_TARGET)
            other_penalty_pos = sum(np.abs(o) for o in d_others)
            other_penalty_neg = sum(np.abs(o) for o in d_others)

            score_pos = diff_pos + 0.5 * other_penalty_pos
            score_neg = diff_neg + 0.5 * other_penalty_neg

            idx_pos = int(np.argmin(score_pos))
            idx_neg = int(np.argmin(score_neg))
            mode_indices_1pct[pname] = (idx_pos, idx_neg)
            save_dict_1pct[f'{pname}_idx_pos'] = np.array([idx_pos])
            save_dict_1pct[f'{pname}_idx_neg'] = np.array([idx_neg])
            save_dict_1pct[f'{pname}_d_pos']   = np.array([d_arr[idx_pos]])
            save_dict_1pct[f'{pname}_d_neg']   = np.array([d_arr[idx_neg]])

        np.savez(out_1pct, **save_dict_1pct)
        print(f'  saved -> {out_1pct}')

    fig, (ax_spec, ax_scat) = plt.subplots(1, 2, figsize=(14, 5),
                                            constrained_layout=True)

    # ── Left: ±1% spectra ─────────────────────────────────────────────────────
    param_colors = {'alpha': '#c62828', 'Rb': '#1565c0', 'Cm': '#2e7d32'}

    for pname, d_arr, lbl_pos, lbl_neg, _ in params_info:
        col = param_colors[pname]
        idx_pos, idx_neg = mode_indices_1pct[pname]
        ax_spec.semilogx(freqs, dZ_norm[idx_pos], color=col, lw=1.4, ls='-',
                         label=lbl_pos)
        ax_spec.semilogx(freqs, dZ_norm[idx_neg], color=col, lw=1.4, ls='--',
                         label=lbl_neg)

    # All-parameters simultaneous ±1% modes
    # Score: sum of |d_param ∓ PCT| across all three parameters
    score_all_pos = (np.abs(d_alpha_arr - PCT_TARGET)
                     + np.abs(d_Rb_arr   - PCT_TARGET)
                     + np.abs(d_Cm_arr   - PCT_TARGET))
    score_all_neg = (np.abs(d_alpha_arr + PCT_TARGET)
                     + np.abs(d_Rb_arr   + PCT_TARGET)
                     + np.abs(d_Cm_arr   + PCT_TARGET))
    idx_all_pos = int(np.argmin(score_all_pos))
    idx_all_neg = int(np.argmin(score_all_neg))
    col_all = '#6a0dad'   # purple — distinct from individual-param colors
    ax_spec.semilogx(freqs, dZ_norm[idx_all_pos], color=col_all, lw=1.6, ls='-',
                     label=r'All +1%')
    ax_spec.semilogx(freqs, dZ_norm[idx_all_neg], color=col_all, lw=1.6, ls='--',
                     label=r'All −1%')

    ax_spec.axvline(1e3,  color='#888', ls=':',  lw=0.8, alpha=0.8)
    ax_spec.axvline(64e3, color='#b22', ls='--', lw=0.8, alpha=0.8)
    ax_spec.set_xlabel('Frequency [Hz]', fontsize=2+11)
    ax_spec.set_ylabel(r'$|\Delta Z(f)| / |Z_0(f)|$ [%]', fontsize=2+11)
    ax_spec.set_title(r'GK sensitivity — $\pm1\%$ parameter perturbations', fontsize=2+11)
    ax_spec.legend(fontsize=2+10, ncol=2, loc='best')
    ax_spec.grid(True, which='both', alpha=0.18)
    ax_spec.tick_params(labelsize=10)

    # ── Right: phase space scatter ────────────────────────────────────────────
    d_rb_max = float(np.max(np.abs(d_Rb_arr)))
    norm_rb  = mcolors.Normalize(vmin=-d_rb_max, vmax=d_rb_max)
    cmap_rb  = plt.cm.RdBu_r

    sc = ax_scat.scatter(dR_tgt, dX_tgt,
                          c=d_Rb_arr, cmap=cmap_rb, norm=norm_rb,
                          s=16, alpha=0.70, edgecolors='none', zorder=2)

    # Max/min |dZ| markers — distinct marker shapes to distinguish them
    adz_tgt = np.hypot(dR_tgt, dX_tgt)
    idx_max = int(np.argmax(adz_tgt))
    idx_min = int(np.argmin(adz_tgt))
    extremes = [
        (idx_max, 'max |dZ|', '*', 25),
        #(idx_min, 'min |dZ|', 'D', 15),
    ]
    for ext_idx, mk_label, mk_shape, mk_size in extremes:
        ax_scat.scatter(dR_tgt[ext_idx], dX_tgt[ext_idx],
                        marker=mk_shape, s=mk_size, color='k', edgecolors='k',
                        linewidths=0.8, zorder=5,
                        label=f'{mk_label} mode')

    ax_scat.axhline(0, color='#aaaaaa', lw=0.5)
    ax_scat.axvline(0, color='#aaaaaa', lw=0.5)
    ax_scat.scatter([0], [0], color='k', s=25, zorder=6, marker='x')

    cb = fig.colorbar(sc, ax=ax_scat, fraction=0.046, pad=0.04)
    cb.set_label(r'$\delta R_b$ (fractional)', fontsize=2+10)
    cb.ax.tick_params(labelsize=9)

    ax_scat.set_xlabel(r'$\delta R$ [$\Omega$]', fontsize=2+11)
    ax_scat.set_ylabel(r'$\delta X$ [$\Omega$]', fontsize=2+11)
    ax_scat.set_title(
        fr'Phase space at {TARGET_FREQ:.0f} Hz — all {N_modes} Library D modes'
        r' (★: max $|\delta Z|$)',
        fontsize=2+11,
    )
    #ax_scat.legend(fontsize=2+10, loc='best')
    ax_scat.tick_params(labelsize=10)
    ax_scat.grid(True, alpha=0.18)

    fig.suptitle('Library D — GK parameter perturbations', fontsize=2+13)
    _save(fig, 'rv2_lib_D.png')


# ─────────────────────────────────────────────────────────────────────────────
# Figure: lib_C
# ─────────────────────────────────────────────────────────────────────────────

def fig_lib_C():
    """Library C brickwall — culture map zoom + perturbation spectrum."""
    p = _require_npz('library_C_brickwall.npz')
    if p is None:
        return

    try:
        from engine import build_geometry, apply_brick_offset
        from SimulSheet_Funcs import Map_Electrode
        geo_available = True
    except ImportError as exc:
        print(f'[lib_C] engine import failed ({exc}); skipping culture map panel.')
        geo_available = False

    npz    = np.load(p, allow_pickle=False)
    deltas = npz['delta']
    off_px = npz['offset_px']
    Z_lib  = npz['Z'].astype(complex)
    Z_base = npz['Z_baseline'].astype(complex)
    freqs  = npz['freqs']
    Z0_mag = np.abs(Z_base)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    ax_map  = axes[0]
    ax_spec = axes[1]

    # ── Left: culture map zoom ────────────────────────────────────────────────
    if geo_available:
        try:
            geo        = build_geometry()
            delta_rep  = 0.50
            id_rep     = int(np.argmin(np.abs(deltas - delta_rep)))
            delta_rep  = float(deltas[id_rep])
            Cult_r     = apply_brick_offset(geo, delta_rep)
            Sub        = geo['Substrate_vector']

            def _combined(Cult, Sub_):
                sub_n = (Sub_ != 0).astype(np.int8)
                return 2 * Cult.astype(np.int32) + sub_n.astype(np.int32)

            Map_Electrode(ax_map, _combined(Cult_r, Sub),
                          geo['p']['de'], geo['d'], geo['side'])
            # Add simulation_engine/ explicitly
            PLOT_LIB_DIR = os.path.join(PROJECT_ROOT, 'plot_libraries')
            if PLOT_LIB_DIR not in sys.path:
                sys.path.insert(0, PLOT_LIB_DIR)
            from plot_libraries import ZOOM_CELLS
            cell_um = geo['stencil'].spacing_x * geo['d'] * 1e4
            half    = ZOOM_CELLS / 2 * cell_um
            ax_map.set_xlim(-half, half)
            ax_map.set_ylim(-half, half)
            leg = ax_map.get_legend()
            if leg is not None:
                leg.remove()
            ax_map.set_title(
                f'Culture map zoom (brick-wall offset δ={delta_rep:.2f})',
                fontsize=2+11,
            )
            ax_map.set_xlabel(r'Position [µm]', fontsize=2+10)
            ax_map.set_ylabel(r'Position [µm]', fontsize=2+10)
            ax_map.tick_params(labelsize=9)
        except Exception as exc:
            ax_map.set_title(f'Culture map unavailable\n({exc})')
            ax_map.axis('off')
    else:
        ax_map.set_title('Culture map unavailable')
        ax_map.axis('off')

    # ── Right: perturbation spectrum ──────────────────────────────────────────
    ax_spec.axvline(1e3,  color='#888', ls=':',  lw=0.8, alpha=0.8, label='1 kHz')
    ax_spec.axvline(64e3, color='#b22', ls='--', lw=0.8, alpha=0.8, label='64 kHz')

    n_modes = Z_lib.shape[0]
    cmap_c  = plt.cm.plasma
    for i, (delta_v, off_v) in enumerate(zip(deltas, off_px)):
        pct = 100.0 * np.abs(Z_lib[i] - Z_base) / Z0_mag
        col = cmap_c(i / max(n_modes - 1, 1))
        lbl = f'δ= {off_v}px'
        ax_spec.semilogx(freqs, pct, color=col, lw=1.0, alpha=0.75, label=lbl)

    ax_spec.set_xlabel('Frequency [Hz]', fontsize=2+11)
    ax_spec.set_ylabel(r'$|\Delta Z|/|Z_0|$ [%]', fontsize=2+11)
    ax_spec.set_title('Perturbation spectrum — Library C', fontsize=2+11)
    ax_spec.legend(fontsize=2+10, ncol=2, loc='best')
    ax_spec.grid(True, which='both', alpha=0.18)
    ax_spec.tick_params(labelsize=10)

    fig.suptitle(
        fr'Library C — Brick-wall cell offset (N={n_modes} modes)',
        fontsize=2+13,
    )
    _save(fig, 'rv2_lib_C.png')


# ─────────────────────────────────────────────────────────────────────────────
# Figure: lib_C_v2  (boundary-style right panel)
# ─────────────────────────────────────────────────────────────────────────────

def fig_lib_C_v2():
    """
    Library C brickwall — culture map zoom (left) + impedance components vs
    offset (right).

    Right panel mirrors the lib_A boundary style: δR (tab:red), δX (tab:green),
    and |δZ| (tab:orange) plotted against brick-wall offset in µm at TARGET_FREQ.
    X-axis is derived from the stored offset_nm array when available, otherwise
    falls back to offset_px.
    """
    p = _require_npz('library_C_brickwall.npz')
    if p is None:
        return

    try:
        from engine import build_geometry, apply_brick_offset
        from SimulSheet_Funcs import Map_Electrode
        geo_available = True
    except ImportError as exc:
        print(f'[lib_C_v2] engine import failed ({exc}); skipping culture map panel.')
        geo_available = False

    npz    = np.load(p, allow_pickle=False)
    deltas = npz['delta']
    off_px = npz['offset_px']
    Z_lib  = npz['Z'].astype(complex)
    Z_base = npz['Z_baseline'].astype(complex)
    freqs  = npz['freqs']
    Z0_mag = np.abs(Z_base)

    # Physical x-axis: prefer offset_nm → µm; fall back to pixels
    if 'offset_nm' in npz.files:
        x_vals  = npz['offset_nm'] * 1e-3        # nm → µm
        x_label = r'Brick-wall offset [µm]'
    else:
        x_vals  = off_px.astype(float)
        x_label = r'Brick-wall offset [px]'

    # Target frequency index
    i_f = int(np.argmin(np.abs(freqs - TARGET_FREQ)))

    # Impedance components at target frequency
    dZ   = Z_lib[:, i_f] - Z_base[i_f]
    dR   = dZ.real
    dX   = dZ.imag
    adZ  = np.abs(dZ)

    n_modes = Z_lib.shape[0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    ax_map  = axes[0]
    ax_bnd  = axes[1]

    # ── Left: culture map zoom (identical to fig_lib_C) ───────────────────────
    if geo_available:
        try:
            geo        = build_geometry()
            delta_rep  = 0.50
            id_rep     = int(np.argmin(np.abs(deltas - delta_rep)))
            delta_rep  = float(deltas[id_rep])
            Cult_r     = apply_brick_offset(geo, delta_rep)
            Sub        = geo['Substrate_vector']

            def _combined(Cult, Sub_):
                sub_n = (Sub_ != 0).astype(np.int8)
                return 2 * Cult.astype(np.int32) + sub_n.astype(np.int32)

            Map_Electrode(ax_map, _combined(Cult_r, Sub),
                          geo['p']['de'], geo['d'], geo['side'])
            # Add simulation_engine/ explicitly
            PLOT_LIB_DIR = os.path.join(PROJECT_ROOT, 'plot_libraries')
            if PLOT_LIB_DIR not in sys.path:
                sys.path.insert(0, PLOT_LIB_DIR)
            from plot_libraries import ZOOM_CELLS
            cell_um = geo['stencil'].spacing_x * geo['d'] * 1e4
            half    = ZOOM_CELLS / 2 * cell_um
            ax_map.set_xlim(-half, half)
            ax_map.set_ylim(-half, half)
            leg = ax_map.get_legend()
            if leg is not None:
                leg.remove()
            ax_map.set_title(
                f'Culture map zoom (brick-wall offset δ={delta_rep:.2f})',
                fontsize=2+11,
            )
            ax_map.set_xlabel(r'Position [µm]', fontsize=2+10)
            ax_map.set_ylabel(r'Position [µm]', fontsize=2+10)
            ax_map.tick_params(labelsize=9)
        except Exception as exc:
            ax_map.set_title(f'Culture map unavailable\n({exc})')
            ax_map.axis('off')
    else:
        ax_map.set_title('Culture map unavailable')
        ax_map.axis('off')

    # ── Right: δR, δX, |δZ| vs offset ────────────────────────────────────────
    C_dR  = 'tab:red'
    C_dX  = 'tab:green'
    C_adZ = 'tab:orange'

    ax_bnd.plot(x_vals, dR,  color=C_dR,  lw=1.4, ls='-',  marker='o',
                ms=5, markerfacecolor='white', markeredgecolor=C_dR,
                label=r'$\delta R$')
    ax_bnd.plot(x_vals, dX,  color=C_dX,  lw=1.4, ls='--', marker='o',
                ms=5, markerfacecolor='white', markeredgecolor=C_dX,
                label=r'$\delta X$')
    ax_bnd.plot(x_vals, adZ, color=C_adZ, lw=1.4, ls=':',  marker='o',
                ms=5, markerfacecolor='white', markeredgecolor=C_adZ,
                label=r'$|\delta Z|$')

    ax_bnd.axhline(0, color='#aaaaaa', lw=0.5, zorder=0)
    ax_bnd.set_xlabel(x_label, fontsize=2+11)
    ax_bnd.set_ylabel(r'Impedance component [$\Omega$]', fontsize=2+11)
    ax_bnd.set_title(
        fr'Impedance components at $f \approx {freqs[i_f]:.0f}$ Hz'
        '\n'
        r'$\delta R$ solid, $\delta X$ dashed, $|\delta Z|$ dotted',
        fontsize=2+11,
    )
    ax_bnd.legend(fontsize=2+10, loc='best')
    ax_bnd.tick_params(labelsize=10)
    ax_bnd.grid(True, alpha=0.20)

    fig.suptitle(
        fr'Library C — Brick-wall cell offset (N={n_modes} modes)',
        fontsize=2+13,
    )
    _save(fig, 'rv2_lib_C_v2.png')


def _pca_arrows(ax, dR, dX, scale=1.0, color='k', lw=1.6, label_pc1=True,
                label_pc2=True, prefix='PC'):
    """
    Compute PCA of (dR, dX) and plot PC1, PC2 arrows from the centroid.
    scale  : arrow length in data units (fraction of std used if <=0 → auto).
    prefix : label prefix, e.g. 'PCf', 'PCs', 'PC'.
    """
    pts  = np.column_stack([dR, dX])
    mu   = pts.mean(axis=0)
    cov  = np.cov(pts.T)
    vals, vecs = np.linalg.eigh(cov)
    # eigh gives ascending eigenvalues; PC1 = last
    pc1  = vecs[:, -1]
    pc2  = vecs[:, -2]
    l1   = np.sqrt(vals[-1])
    l2   = np.sqrt(vals[-2])

    for show, pc, l, lbl in [(label_pc1, pc1, l1, f'{prefix}1'),
                              (label_pc2, pc2, l2, f'{prefix}2')]:
        if not show:
            continue
        arrow_len = scale * l if scale > 0 else l
        ax.annotate(
            '', xytext=mu, xy=mu + arrow_len * pc,
            arrowprops=dict(arrowstyle='->', color=color, lw=lw),
        )
        ax.text(*(mu + arrow_len * 1.10 * pc), lbl,
                color=color, fontsize=10, ha='center', va='center')

def fig_timescales():
    """Two-timescale decomposition for four focus electrodes."""
    npz = _load_exp_npz()
    if npz is None:
        return

    from scipy.ndimage import uniform_filter1d

    WINDOW_SEC = 30.0   # fast/slow split window [s]

    fig = plt.figure(figsize=(16, 11), constrained_layout=True)
    outer = gridspec.GridSpec(4, 3, figure=fig,
                              hspace=0.00, wspace=0.05)

    col_titles = [
        r'$R(t)$ & $X(t)$ with slow component ($T=30\ \mathrm{s}$ MA)',
        r'Fast component phase space ($\delta\!R_\mathrm{fast},\,\delta\!X_\mathrm{fast}$)',
        r'Slow component phase space ($\delta\!R_\mathrm{slow},\,\delta\!X_\mathrm{slow}$)',
    ]

    for row, stem in enumerate(FOCUS_STEMS):
        label = FOCUS_LABELS[stem]
        color = FOCUS_COLORS[stem]

        arrs = _get_stem_arrays(npz, stem, ['dR', 'dX', 't'])
        dR, dX, t_arr = arrs['dR'], arrs['dX'], arrs['t']
        if dR is None or t_arr is None:
            for col in range(3):
                ax = fig.add_subplot(outer[row, col])
                ax.set_title(f'{label} — data missing')
            continue

        # Compute theta_all: PCA angle of full (dR, dX) cloud [degrees]
        _pts_all = np.column_stack([dR.astype(float), dX.astype(float)])
        _cov_all = np.cov(_pts_all.T)
        _vals_all, _vecs_all = np.linalg.eigh(_cov_all)
        _pc1_all = _vecs_all[:, -1]   # eigenvector of largest eigenvalue
        theta_all = float(np.degrees(np.arctan2(_pc1_all[1], _pc1_all[0])))

        dt = float(np.median(np.diff(t_arr)))

        # Reconstruct original R from dR + moving-average trend
        # Phase1_experimental stores the detrended signal; we need the original R.
        # Load from load_experimental if available.
        R_orig = None
        try:
            _exp_dir = os.path.join(
                os.path.abspath(os.path.join(os.path.dirname(__file__), '..')),
                'experimental_signal_analysis'
            )
            if _exp_dir not in sys.path:
                sys.path.insert(0, _exp_dir)
            from load_experimental import load_dat, micromotion_stats
            t_raw, R_raw, X_raw, _, _ = load_dat(stem)
            if len(t_raw) == len(t_arr):
                R_orig = R_raw
                X_orig = X_raw
        except Exception as exc:
            print(f'[timescales] load_experimental failed for {stem}: {exc}')

        # Slow component of dR (already detrended at 900 s; now apply 30 s MA)
        n_win = max(3, int(round(WINDOW_SEC / dt)))
        if n_win % 2 == 0:
            n_win += 1
        slow_dR = uniform_filter1d(dR.astype(float), size=n_win, mode='nearest')
        fast_dR = dR - slow_dR

        slow_dX = uniform_filter1d(dX.astype(float), size=n_win, mode='nearest')
        fast_dX = dX - slow_dX

        t_h = t_arr / 3600.0

        # ── Left: R(t) or dR(t) with slow overlay ────────────────────────────
        ax0 = fig.add_subplot(outer[row, 0])
        if R_orig is not None:
            t_h_raw = t_raw / 3600.0
            ax0.plot(t_h_raw, R_orig, color='tab:red', lw=0.35, alpha=0.7,
                     label=r'$R(t)$')
            # Slow overlay: reconstruct from R_raw
            n_win2 = max(3, int(round(WINDOW_SEC / float(np.median(np.diff(t_raw))))))
            if n_win2 % 2 == 0:
                n_win2 += 1
            slow_R = uniform_filter1d(R_raw.astype(float), size=n_win2,
                                      mode='nearest')
            ax0.plot(t_h_raw, slow_R, color='k', linestyle='--', lw=1.0, alpha=0.6,
                     label=f'Slow R')
            ax0.set_ylabel(r'$R$ [$\Omega$]', fontsize=12)
            # Secondary y-axis for X_orig
            ax0_right = ax0.twinx()
            ax0_right.plot(t_h_raw, X_orig, color='tab:green', lw=0.35,
                           alpha=0.7, label=r'$X(t)$')
            slow_X = uniform_filter1d(X_raw.astype(float), size=n_win2,
                                      mode='nearest')
            ax0_right.plot(t_h_raw, slow_X, color='k', linestyle=':',
                           lw=1.0, alpha=0.6,
                           label=f'Slow X')
            ax0_right.set_ylabel(r'$X$ [$\Omega$]', fontsize=12)
            ax0_right.tick_params(labelsize=9, colors='black')
            ax0_right.spines['right'].set_color('black')
            # Combined legend
            handles0, labels0 = ax0.get_legend_handles_labels()
            handles0r, labels0r = ax0_right.get_legend_handles_labels()
            if row == 1: ax0.legend(handles0 + handles0r, labels0 + labels0r,
                       fontsize=10, ncol=2, loc='lower right')
        else:
            # Fallback: use stored dR
            ax0.plot(t_h, dR, color='tab:red', lw=0.35, alpha=0.7,
                     label=r'$\delta R(t)$')
            ax0.plot(t_h, slow_dR, color='tab:red', linestyle='--', lw=1.0, alpha=0.8,
                     label=f'Slow R')
            ax0.set_ylabel(r'$\delta R$ [$\Omega$]', fontsize=12)
            # Secondary y-axis for dX
            ax0_right = ax0.twinx()
            ax0_right.plot(t_h, dX, color='tab:green', lw=0.35, alpha=0.7,
                           label=r'$X(t)$')
            ax0_right.plot(t_h, slow_dX, color='tab:green', linestyle='--',
                           lw=1.0, alpha=0.8,
                           label=f'Slow X')
            ax0_right.set_ylabel(r'$X$ [$\Omega$]', fontsize=12,
                                 color='tab:green')
            ax0_right.tick_params(labelsize=9, colors='black')
            ax0_right.spines['right'].set_color('black')
            # Combined legend
            handles0, labels0 = ax0.get_legend_handles_labels()
            handles0r, labels0r = ax0_right.get_legend_handles_labels()
            ax0.legend(handles0 + handles0r, labels0 + labels0r,
                       fontsize=8, ncol=2, loc='upper right')

        ax0.set_xlabel('Time [h]', fontsize=12)
        ax0.tick_params(labelsize=9)
        ax0.grid(True, alpha=0.18)
        if row == 0:
            ax0.set_title(col_titles[0], fontsize=13)
        ax0.text(0.5, 0.96, label, transform=ax0.transAxes,
                 fontsize=12, va='top', color=color)

        # ── Centre: fast phase space + PC1 ───────────────────────────────────
        ax1 = fig.add_subplot(outer[row, 1])
        ax1.scatter(fast_dR, fast_dX, s=2, color=color, alpha=0.25,
                    edgecolors='none')
        _pca_arrows(ax1, fast_dR, fast_dX, scale=2.0, color='k', lw=1.4,
                    label_pc1=True, label_pc2=False, prefix='PCf')
        ax1.text(0.97, 0.04, r'$\theta_\mathrm{all}=' + f'{theta_all:.1f}' + r'^\circ$',
                 transform=ax1.transAxes, fontsize=12, ha='right', va='bottom',
                 bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))
        # Robust axis limits: 1st–99th percentile + 10 % padding
        _r1, _r99 = np.percentile(fast_dR, [1, 99])
        _x1, _x99 = np.percentile(fast_dX, [1, 99])
        _rpad = 0.10 * (_r99 - _r1)
        _xpad = 0.10 * (_x99 - _x1)
        ax1.set_xlim(_r1 - _rpad, _r99 + _rpad)
        ax1.set_ylim(_x1 - _xpad, _x99 + _xpad)
        ax1.axhline(0, color='#aaaaaa', lw=0.5)
        ax1.axvline(0, color='#aaaaaa', lw=0.5)
        ax1.set_xlabel(r'$\delta R_\mathrm{fast}$ [$\Omega$]', fontsize=12)
        ax1.set_ylabel(r'$\delta X_\mathrm{fast}$ [$\Omega$]', fontsize=12)
        ax1.tick_params(labelsize=9)
        ax1.grid(True, alpha=0.18)
        if row == 0:
            ax1.set_title(col_titles[1], fontsize=13)

        # ── Right: slow phase space ───────────────────────────────────────────
        ax2 = fig.add_subplot(outer[row, 2])
        ax2.scatter(slow_dR, slow_dX, s=2, color=color, alpha=0.35,
                    edgecolors='none')
        _pca_arrows(ax2, slow_dR, slow_dX, scale=2.0, color='k', lw=1.4,
                    label_pc1=True, label_pc2=False, prefix='PCs')
        ax2.text(0.97, 0.04, r'$\theta_\mathrm{all}=' + f'{theta_all:.1f}' + r'^\circ$',
                 transform=ax2.transAxes, fontsize=12, ha='right', va='bottom',
                 bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))
        ax2.axhline(0, color='#aaaaaa', lw=0.5)
        ax2.axvline(0, color='#aaaaaa', lw=0.5)
        ax2.set_xlabel(r'$\delta R_\mathrm{slow}$ [$\Omega$]', fontsize=12)
        ax2.set_ylabel(r'$\delta X_\mathrm{slow}$ [$\Omega$]', fontsize=12)
        ax2.tick_params(labelsize=11)
        ax2.grid(True, alpha=0.18)
        if row == 0:
            ax2.set_title(col_titles[2], fontsize=13)

    fig.suptitle(
        r'Two-timescale decomposition  '
        r'($\delta = \mathrm{signal} - \mathrm{moving\ average},\ T=30\ \mathrm{s}$)',
        fontsize=15,
    )
    _save(fig, 'rv2_timescales.png')



# ─────────────────────────────────────────────────────────────────────────────
# Figure: coverage
# ─────────────────────────────────────────────────────────────────────────────

def fig_coverage():
    """
    Three-panel coverage figure.
    Left  : full phase space (all library modes + experimental trajectories).
    Centre: zoom ±3000 Ω on both axes.
    Right : |dZ| amplitude distribution.
    """
    # Library catalogue in desired legend order: B, A, C, D
    lib_specs = [
        ('B_rb',        'library_B_rb.npz',        'B', '#8B4513', 30, 0.65, 3, 'D'),
        ('A_shift',     'library_A_shift.npz',      'A', '#1565C0', 15, 0.40, 2, 'o'),
        ('C_brickwall', 'library_C_brickwall.npz',  'C', '#7B1FA2', 20, 0.50, 2, '^'),
        ('D_rb',        'library_D_rb.npz',         'D', '#00695C', 30, 0.80, 4, 's'),
    ]

    lib_data = []
    for key, fname, display_key, color, ms, alpha, zo, marker in lib_specs:
        res = _load_lib_dz(fname)
        if res is None:
            continue
        dR, dX, _, _, npz_d = res
        meta = {}
        if 'k_rb_arr' in npz_d:
            meta['k_rb_arr'] = npz_d['k_rb_arr'].copy()
        lib_data.append((display_key, dR, dX, meta, color, ms, alpha, zo, marker))

    if not lib_data:
        print('[SKIP] coverage: no library files found.')
        return

    # Experimental data
    try:
        sys.path.insert(0, SCRIPT_DIR_PARENT)
        from load_experimental import load_dat, micromotion_stats
        exp_data = {}
        for stem in FOCUS_STEMS:
            try:
                t_raw, R_raw, X_raw, _, _ = load_dat(stem)
                stats = micromotion_stats(R_raw, X_raw, t_raw,
                                          detrend='moving_avg',
                                          window_sec=900)
                exp_data[stem] = {
                    'dR': stats['dR'],
                    'dX': stats['dX'],
                    't':  stats['t'],
                }
            except Exception as exc:
                print(f'[coverage] Could not load {stem}: {exc}')
    except ImportError:
        exp_data = {}
        # Fall back to phase1_experimental.npz
        npz_exp = _load_exp_npz()
        if npz_exp is not None:
            for stem in FOCUS_STEMS:
                arrs = _get_stem_arrays(npz_exp, stem, ['dR', 'dX', 't'])
                if arrs['dR'] is not None:
                    exp_data[stem] = arrs

    # Combined library hull (union of all individual mode clouds).
    if lib_data:
        all_lib_pts = np.vstack([
            np.column_stack([dR, dX]) for _, dR, dX, _, _, _, _, _, _ in lib_data
        ])
    else:
        all_lib_pts = np.empty((0, 2))

    # Minkowski sum hull: the true reachable region when one mode from each
    # library is ADDED together at every reconstruction time step.
    print('  [coverage] Computing Minkowski sum hull...')
    if lib_data:
        _point_sets = [np.column_stack([dR, dX])
                       for _, dR, dX, _, _, _, _, _, _ in lib_data]
        mink_verts = compute_minkowski_hull(_point_sets)
        if mink_verts is not None:
            print(f'  [coverage] Minkowski hull: {len(mink_verts)} vertices')
        else:
            print('  [coverage] WARNING: Minkowski hull computation returned None.')
    else:
        mink_verts = None

    cmap_time = plt.cm.plasma

    fig = plt.figure(figsize=(22, 7), constrained_layout=True)
    gs_outer = fig.add_gridspec(1, 3, width_ratios=[6, 4, 5])
    ax_full = fig.add_subplot(gs_outer[0])
    ax_zoom = fig.add_subplot(gs_outer[1])
    gs_right = gs_outer[2].subgridspec(4, 1, hspace=0.05)
    ax_hist_panels = [fig.add_subplot(gs_right[i]) for i in range(4)]


    def _draw_scatter(ax, xlim=None, ylim=None, show_exp=True, lib_data_use=None,
                      show_krb_legend=False):
        if lib_data_use is None:
            lib_data_use = lib_data
        # Minkowski sum hull outline (true reachable region)
        if mink_verts is not None and len(mink_verts) >= 3:
            mink_closed = np.vstack([mink_verts, mink_verts[0]])
            ax.plot(mink_closed[:, 0], mink_closed[:, 1],
                    color='black', ls='--', lw=1.5, alpha=0.55, zorder=6,
                    label='Minkowski sum hull')

        legend_handles = []

        for display_key, dR, dX, meta, color, ms, alpha, zo, marker in lib_data_use:
            pts  = np.column_stack([dR, dX])
            poly = _hull_polygon(dR, dX)
            if poly is not None:
                ax.fill(poly[:, 0], poly[:, 1],
                        color=color, alpha=0.07, linewidth=0, zorder=2)

            if display_key == 'B' and 'k_rb_arr' in meta:
                norm_rb  = mcolors.Normalize(vmin=meta['k_rb_arr'].min(),
                                              vmax=meta['k_rb_arr'].max())
                colors_rb = plt.cm.RdYlGn(norm_rb(meta['k_rb_arr']))
                ax.scatter(dR, dX, s=ms, c=colors_rb, alpha=alpha, zorder=zo,
                           marker=marker)
            else:
                ax.scatter(dR, dX, s=ms, color=color, alpha=alpha, zorder=zo,
                           marker=marker)

            proxy = mlines.Line2D(
                [], [], color=color, marker=marker, linestyle='None',
                markersize=5, alpha=min(alpha + 0.2, 1.0),
                label=f'Lib {display_key} (N={len(dR)})',
            )
            legend_handles.append(proxy)

        # Experimental
        exp_proxies = []
        for stem in (FOCUS_STEMS if show_exp else []):
            if stem not in exp_data:
                continue
            dR_e = exp_data[stem]['dR']
            dX_e = exp_data[stem]['dX']
            t_e  = exp_data[stem].get('t', np.linspace(0, 1, len(dR_e)))
            if t_e is None:
                t_e = np.linspace(0, 1, len(dR_e))
            span   = t_e.max() - t_e.min()
            t_norm = (t_e - t_e.min()) / max(span, 1.0)
            ax.scatter(dR_e, dX_e,
                       c=t_norm, cmap=cmap_time,
                       marker='x', s=8, linewidths=0.5,
                       alpha=0.30, zorder=1, vmin=0.0, vmax=1.0)
            ep = mlines.Line2D([], [], color=cmap_time(0.5),
                               marker='x', linestyle='None',
                               markersize=5, alpha=0.6,
                               label=f'{FOCUS_LABELS[stem]}')
            exp_proxies.append(ep)

        #if mink_verts is not None and len(mink_verts) >= 3:
        #    hull_proxy = mlines.Line2D([], [], color='black', ls='--',
        #                               lw=1.5, alpha=0.55, label='Minkowski sum hull')
        #    legend_handles.insert(0, hull_proxy)

        ax.axvline(0, color='grey', ls='--', lw=0.5, zorder=0)
        ax.axhline(0, color='grey', ls='--', lw=0.5, zorder=0)
        ax.set_xlabel(r'$\delta R$ [$\Omega$]', fontsize=2+12)
        ax.set_ylabel(r'$\delta X$ [$\Omega$]', fontsize=2+12)
        ax.tick_params(labelsize=10)
        ax.grid(True, alpha=0.15)
        

        all_handles = []
        if mink_verts is not None:
            hull_proxy = mlines.Line2D(
                [], [], color='black', linestyle='--',
                linewidth=1.5, alpha=0.3, label='Minkowski sum hull',
            )
            all_handles.append(hull_proxy)
        all_handles.extend(legend_handles)
        all_handles.extend(exp_proxies)
        ax_full.legend(
            handles=all_handles,
            loc='upper right', fontsize=2+9, framealpha=0.7, ncol=2,
        )
        
        exp_cmap_names = ['#E91E63', '#2196F3', '#FF9800', '#4CAF50']
        # Experimental scatter per electrode.
        right_handles = []
        annotation_lines = []
        for i, (stem, ed) in enumerate(exp_data.items()):
            dR_e  = ed['dR']
            dX_e  = ed['dX']
            color = exp_cmap_names[i % len(exp_cmap_names)]

            # Fraction inside Minkowski hull.
            test_pts = np.column_stack([dR_e, dX_e])
            if mink_verts is not None and len(mink_verts) >= 3:
                mask_mink = _in_mink_hull(test_pts, mink_verts)
                frac      = mask_mink.mean() * 100.0
                label_str = f'{stem} ({frac:.1f}% inside)'
                annotation_lines.append(
                    f'{stem}: {frac:.1f}% of {len(dR_e)} pts inside Mink. hull'
                )
            else:
                label_str = stem
            proxy = mlines.Line2D(
                [], [], color=color, marker='x', linestyle='None',
                markersize=5, alpha=0.7, label=label_str,
            )
            right_handles.append(proxy)
        # Annotation box: per-electrode coverage fractions.
        if annotation_lines:
            box_text = '\n'.join(annotation_lines)
            ax_full.text(
                0.50, 0.70,
                box_text,
                transform=ax_full.transAxes,
                fontsize=2+9, verticalalignment='bottom',
                bbox=dict(
                    boxstyle='round,pad=0.3',
                    facecolor='lightyellow',
                    alpha=0.85,
                    edgecolor='grey',
                ),
            )

        all_h = legend_handles + exp_proxies
        ax.legend(handles=all_handles, fontsize=2+9, ncol=2,
                  loc='upper right', framealpha=0.75)
        
        # Discrete k_rb legend for Library B (left panel only)
        if show_krb_legend:
            for dk, _, _, meta_b, _, _, _, _, mk_b in lib_data_use:
                if dk == 'B' and 'k_rb_arr' in meta_b:
                    k_unique = np.unique(meta_b['k_rb_arr'])
                    norm_k   = mcolors.Normalize(vmin=k_unique.min(),
                                                 vmax=k_unique.max())
                    cmap_k   = plt.cm.RdYlGn
                    krb_handles = []
                    for kv in k_unique:
                        note = '(+: tighter)' if kv > 0 else ('(−: looser)' if kv < 0 else '')
                        krb_handles.append(
                            mlines.Line2D([], [], color=cmap_k(norm_k(kv)),
                                          marker=mk_b, linestyle='None',
                                          markersize=5,
                                          label=f'$k_{{Rb}}$ = {kv:+.2g}  {note}')
                        )
                    krb_leg = ax.legend(
                        handles=krb_handles,
                        title=r'Lib B: $R_b$ coupling ($k_{Rb}$)',
                        loc='lower left', fontsize=2+9, title_fontsize=2+10.5,
                        framealpha=0.85, edgecolor='saddlebrown',
                    )
                    ax.add_artist(krb_leg)
                    break

        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)

    # ── Left: full ────────────────────────────────────────────────────────────
    # Draw order D→C→B→A: D sits just above experimental (zo=2), A on top (zo=5).
    # Original alpha and marker sizes are preserved; only z-orders are adjusted.
    _lib_by_key_full = {entry[0]: list(entry) for entry in lib_data}
    _full_order = ['D', 'C', 'B', 'A']
    _full_zo    = {'D': 2, 'C': 3, 'B': 4, 'A': 5}
    full_lib_data = []
    for _k in _full_order:
        if _k in _lib_by_key_full:
            _entry = list(_lib_by_key_full[_k])
            _entry[7] = _full_zo[_k]   # index 7: zo
            full_lib_data.append(tuple(_entry))
    _draw_scatter(ax_full, show_krb_legend=True, lib_data_use=full_lib_data)
    ax_full.set_title(
        'Phase space coverage — all library modes + experimental\n(at 1 kHz)',
        fontsize=2+11,
    )

    # Colourbar for time
    if exp_data:
        sm = plt.cm.ScalarMappable(cmap=cmap_time,
                                   norm=mcolors.Normalize(0, 1))
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax_full, fraction=0.03, pad=0.02)
        cb.set_label('Norm. time', fontsize=2+10)
        cb.set_ticks([0, 0.5, 1]); cb.set_ticklabels(['start', 'mid', 'end'])

    # ── Centre: zoom ─────────────────────────────────────────────────────────
    ZOOM_X = 500.0
    ZOOM_Y = 1000.0
    # Zoom panel: no experimental data; order D→C→B→A.
    # D is drawn first (lowest z-order, very low alpha) so A/C/B are visible.
    _lib_by_key = {entry[0]: list(entry) for entry in lib_data}
    _zoom_order = ['D', 'C', 'B', 'A']
    _zoom_alpha = {'D': 0.20, 'C': 0.90, 'B': 0.90, 'A': 0.90}
    _zoom_zo    = {'D': 1,    'C': 2,    'B': 3,    'A': 4}
    zoom_lib_data = []
    for _k in _zoom_order:
        if _k in _lib_by_key:
            _entry = list(_lib_by_key[_k])
            _entry[5] = 4 if _k == 'D' else _entry[5]  # index 5: ms; shrink D
            _entry[6] = _zoom_alpha[_k]                  # index 6: alpha
            _entry[7] = _zoom_zo[_k]                     # index 7: zo
            zoom_lib_data.append(tuple(_entry))
    _draw_scatter(ax_zoom, xlim=(-ZOOM_X, ZOOM_X), ylim=(-ZOOM_Y, ZOOM_Y),
                  show_exp=False, lib_data_use=zoom_lib_data)
    ax_zoom.set_title(
        fr'Zoom: $\pm{ZOOM_X:.0f}\ \Omega$ (x) / $\pm{ZOOM_Y:.0f}\ \Omega$ (y)',
        fontsize=2+11,
    )

    # ── Right: |dZ| amplitude distribution — 4 vertical subplots ─────────────
    n_panels = len(ax_hist_panels)
    # Pad lib_data with None entries if fewer than 4 libraries exist
    lib_data_padded = list(lib_data) + [None] * (n_panels - len(lib_data))

    # Pre-pass: global x limits across all libraries + experimental std lines
    _all_adz = [np.hypot(e[1], e[2]) for e in lib_data_padded if e is not None]
    _exp_stds = [
        np.hypot(exp_data[s]['dR'], exp_data[s]['dX']).std()
        for s in FOCUS_STEMS if s in exp_data
    ]
    _all_vals = np.concatenate(_all_adz) if _all_adz else np.array([1.0])
    _pos_all  = _all_vals[_all_vals > 0]
    _x_lo = max(_pos_all.min() if len(_pos_all) else 1.0, 1.0) * 0.8
    _x_hi = _all_vals.max() * 1.2
    if _exp_stds:
        _x_hi = max(_x_hi, max(_exp_stds) * 1.2)
    bins_global = np.logspace(np.log10(_x_lo), np.log10(_x_hi), 40)

    for panel_idx, (panel_ax, entry) in enumerate(
        zip(ax_hist_panels, lib_data_padded)
    ):
        is_bottom = (panel_idx == n_panels - 1)
        is_top    = (panel_idx == 0)

        if entry is None:
            panel_ax.set_visible(False)
            continue

        display_key, dR, dX, meta, color, ms, alpha_v, zo, marker = entry
        adz = np.hypot(dR, dX)

        # Histogram — shared bins across all panels
        counts, edges = np.histogram(adz, bins=bins_global)
        widths  = np.diff(edges)
        density = counts / max((counts * widths).sum(), 1e-30)
        centres = 0.5 * (edges[:-1] + edges[1:])

        panel_ax.bar(
            centres, density, width=widths,
            color=color, alpha=0.6,
            align='center',
           # label=f'Lib {display_key} (N={len(adz)})',
        )
        panel_ax.set_xscale('log')
        panel_ax.set_yscale('log')

        # Experimental std as vertical lines (x = |dZ| std in Ohms)
        for stem in FOCUS_STEMS:
            if stem not in exp_data:
                continue
            adz_e = np.hypot(exp_data[stem]['dR'], exp_data[stem]['dX'])
            panel_ax.axvline(
                adz_e.std(),
                color=FOCUS_COLORS[stem],
                ls='--', lw=1.2, alpha=0.85,
                label=f'{FOCUS_LABELS[stem]} std',
            )

        panel_ax.set_title(f'Library {display_key}', fontsize=2+10, pad=2)
        panel_ax.tick_params(labelsize=9)
        panel_ax.grid(True, alpha=0.15)

        # x-axis label only on the bottom panel
        if is_bottom:
            panel_ax.set_xlabel(r'$|\delta Z|$ [$\Omega$]', fontsize=2+10)
        else:
            panel_ax.tick_params(labelbottom=False)

        # y-axis label only on the top panel
        if is_top:
            panel_ax.set_ylabel('Norm. density', fontsize=2+10)

        # Legend on Library C panel (most empty space); upper left
        if display_key == 'C':
            panel_ax.legend(fontsize=2+8, loc='upper left', framealpha=0.75,
                            ncol=1)

    # Enforce shared x limits on all visible histogram panels
    for panel_ax in ax_hist_panels:
        if panel_ax.get_visible():
            panel_ax.set_xlim(_x_lo, _x_hi)

    fig.suptitle('Phase-space coverage: simulation libraries vs experiment at 1 kHz',
                 fontsize=2+13)
    _save(fig, 'rv2_coverage.png')

def fig_baseline():
    """Three-panel baseline figure: geometry map | |Z| spectrum | phase spectrum.

    Curves shown: FD engine (baseline.npz), GK analytical (GKSpectre),
    bare CPE, and experimental CH3 confluent spectra.
    """
    import colorsys  # already imported for lib_B; safe to re-import

    # ── Load FD baseline ──────────────────────────────────────────────────────
    p = _require_npz('baseline.npz')
    if p is None:
        return
    npz    = np.load(p, allow_pickle=False)
    freqs  = npz['freqs']
    Z_fd   = npz['Z'].astype(complex)
    de_cm  = float(npz['de'])          # electrode diameter [cm]

    # ── GK analytical (GKSpectre) ─────────────────────────────────────────────
    try:
        shift2 = os.path.join(SCRIPT_DIR_PARENT, 'Shift2')
        if shift2 not in sys.path:
            sys.path.insert(0, shift2)
        from CPE_GK_Funs import GKSpectre
        gk_available = True
    except ImportError as exc:
        print(f'[baseline] GKSpectre unavailable ({exc}); skipping GK curve.')
        gk_available = False

    alpha_fit = 19.4
    Rb_fit    = 14.4          # Ω·cm²
    Cm_fit    = 4.95e-6       # F/cm²
    A_cpe     = 6.98e-9       # Ω⁻¹·s^N (CPE A, eb17 mean)
    N_cpe     = 0.928         # CPE N
    ae        = np.pi * de_cm ** 2 / 4 /2   # electrode area [cm²]
    omega     = 2 * np.pi * freqs
    ZN_spec   = ae / (A_cpe * (1j * omega) ** N_cpe)   # specific Ω·cm²

    if gk_available:
        Z_gk, _, _ = GKSpectre(alpha_fit, Rb_fit, Cm_fit,
                                     ZN_spec, freqs, 0.0, ae) 

    # ── Bare CPE ──────────────────────────────────────────────────────────────
    Z_cpe = ZN_spec / ae    # Ω

    # ── Experimental CH3 ──────────────────────────────────────────────────────
    dat_path = os.path.join(SCRIPT_DIR_PARENT, 'experimental_signal_analysis', 'ConfluentCH3Data.dat')
    exp_available = os.path.isfile(dat_path)
    if exp_available:
        try:
            f_exp, Z_exp_mat = _load_dat_ch(dat_path, ch_select=3)
            Z_exp_mean = Z_exp_mat.mean(axis=0)
            exp_available = True
        except Exception as exc:
            print(f'[baseline] Could not parse ConfluentCH3Data.dat: {exc}')
            exp_available = False

    # ── Geometry panel ────────────────────────────────────────────────────────
    try:
        from engine import build_geometry
        from SimulSheet_Funcs import Map_Electrode
        geo_available = True
    except ImportError as exc:
        print(f'[baseline] engine import failed ({exc}); skipping geometry panel.')
        geo_available = False

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    ax_geo, ax_mag, ax_ph = axes

    # Panel 1 — geometry map
    if geo_available:
        try:
            geo  = build_geometry()
            Cult = geo['Culture_vector']
            Sub  = geo['Substrate_vector']
            p_geo, d_geo, side = geo['p'], geo['d'], geo['side']

            def _combined(C, S):
                return 2 * C.astype(np.int32) + (S != 0).astype(np.int32)

            Map_Electrode(ax_geo, _combined(Cult, Sub), p_geo['de'], d_geo, side)
            leg = ax_geo.get_legend()
            if leg is not None:
                leg.remove()
            ax_geo.set_title('Simulation domain', fontsize=2+13)
            ax_geo.set_xlabel(r'Position [$\mu$m]', fontsize=2+12)
            ax_geo.set_ylabel(r'Position [$\mu$m]', fontsize=2+12)
            ax_geo.tick_params(labelsize=9)
        except Exception as exc:
            ax_geo.set_title(f'Geometry unavailable\n({exc})', fontsize=2+12)
            ax_geo.axis('off')
    else:
        ax_geo.set_title('Geometry unavailable\n(engine not imported)', fontsize=2+12)
        ax_geo.axis('off')

    # Two-electrode corrected curves: Z_meas = 2*(Z_single + R_constr)
    R_constr  = 1000.0   # Ω per electrode (eb17 mean, analytical fit)
    Z_fd_2el  =  2*(Z_fd + R_constr)
    if gk_available:
        Z_gk_orig = Z_gk.copy()   # keep single-electrode value for annotation
        Z_gk_2el  = (Z_gk  + 2.0 * R_constr)
    Z_cpe_2el =  (Z_cpe + 2.0 * R_constr)

    # Shared axis settings
    for ax in (ax_mag, ax_ph):
        ax.axvline(1e3,  color='#888', ls=':',  lw=0.8, alpha=0.8)
        ax.axvline(64e3, color='#b22', ls='--', lw=0.8, alpha=0.8)
        ax.set_xscale('log')
        ax.set_xlabel('Frequency [Hz]', fontsize=2+13)
        ax.grid(True, which='both', alpha=0.18)
        ax.tick_params(labelsize=10)

    # Experimental mean ± std per frequency
    if exp_available:
        mag_sweeps = np.abs(Z_exp_mat)                      # (n_sweeps, N_frec)
        ph_sweeps  = np.angle(Z_exp_mat, deg=True)
        mag_mean   = mag_sweeps.mean(axis=0)
        mag_std    = mag_sweeps.std(axis=0)
        ph_mean    = ph_sweeps.mean(axis=0)
        ph_std     = ph_sweeps.std(axis=0)

    # |Z| — log–log
    ax_mag.set_yscale('log')
    ax_mag.plot(freqs, np.abs(Z_fd_2el), color='tab:blue', lw=1.6,
                label=fr'FD $\times2$ + $R_\mathrm{{constr}}$')
    if gk_available:
        ax_mag.plot(freqs, np.abs(Z_gk_2el), color='tab:orange', lw=1.4,
                    ls='--', label=r'GK analytical $\times2$ + $R_\mathrm{constr}$')
    ax_mag.plot(freqs, np.abs(Z_cpe_2el), color='tab:grey', lw=1.2,
                ls=':', label=r'Bare CPE $\times2$ + $R_\mathrm{constr}$')
    if exp_available:
        ax_mag.errorbar(f_exp, mag_mean, yerr=mag_std,
                        fmt='o', color='tab:red', ms=3, lw=1.2, capsize=3,
                        label='Experimental confluent culture')

    # Annotate 3%/0.7° discrepancy at 1 kHz (single-electrode values, no R_constr)
    if gk_available:
        i1k_fd   = int(np.argmin(np.abs(freqs - 1e3)))
        z_fd_1k  = np.abs(Z_fd_2el[i1k_fd])
        pct_diff = 100.0 * abs(np.abs(Z_fd_2el[i1k_fd]) - np.abs(Z_gk_orig[i1k_fd])) \
                   / np.abs(Z_gk_orig[i1k_fd])
        ax_mag.annotate(
            fr'Finite Difference vs GK: $\Delta|Z|={pct_diff:.1f}\%$''\n@ 1 kHz',
            xy=(1e3, z_fd_1k), xytext=(3e3, z_fd_1k * 1.15),
            fontsize=2+9, color='#444',
            arrowprops=dict(arrowstyle='->', color='#444', lw=0.8),
        )
    ax_mag.set_xlim(freqs.min(), freqs.max())
    mag_fd = np.abs(Z_fd_2el)
    ax_mag.set_ylim(mag_fd.min() * 0.5, mag_fd.max() * 2.0)
    ax_mag.set_ylabel(r'$|Z|$ [$\Omega$]', fontsize=2+13)
    ax_mag.set_title('Impedance magnitude', fontsize=2+13)
    #ax_mag.legend(fontsize=2+9, loc='best')

    # Phase — log x, linear y
    ax_ph.plot(freqs, np.angle(Z_fd_2el, deg=True), color='tab:blue', lw=1.6,
               label=r'Finite dif. + $R_\mathrm{constr}$')
    if gk_available:
        ax_ph.plot(freqs, np.angle(Z_gk_2el, deg=True), color='tab:orange', lw=1.4,
                   ls='--', label=r'GK analytical + $R_\mathrm{constr}$')
    ax_ph.plot(freqs, np.angle(Z_cpe_2el, deg=True), color='tab:grey', lw=1.2,
               ls=':', label=r'Bare CPE Impedance + $R_\mathrm{constr}$')
    if exp_available:
        ax_ph.errorbar(f_exp, ph_mean, yerr=ph_std,
                       fmt='o', color='tab:red', ms=3, lw=1.2, capsize=3,
                       label='Experimental confluent culture')

    # Annotate 0.7° phase discrepancy at 1 kHz
    if gk_available:
        ph_fd_1k = np.angle(Z_fd_2el[i1k_fd], deg=True)
        ph_diff  = abs(np.angle(Z_fd_2el[i1k_fd], deg=True) - np.angle(Z_gk_orig[i1k_fd], deg=True))
        ax_ph.annotate(
            fr'$\Delta\varphi={ph_diff:.1f}^\circ$ @ 1 kHz',
            xy=(1e3, ph_fd_1k), xytext=(3e3, ph_fd_1k + 2),
            fontsize=2+9, color='#444',
            arrowprops=dict(arrowstyle='->', color='#444', lw=0.8),
        )

    ax_ph.set_xlim(freqs.min(), freqs.max())
    ph_fd = np.angle(Z_fd_2el, deg=True)
    ax_ph.set_ylim(ph_fd.min() - 5, ph_fd.max() + 5)
    ax_ph.set_ylabel(r'Phase [$^\circ$]', fontsize=2+13)
    ax_ph.set_title('Impedance phase', fontsize=2+13)
    ax_ph.legend(fontsize=2+11, loc='best')

    fig.suptitle('Baseline — FD engine validation against GK analytical and experiment',
                 fontsize=2+13)
    _save(fig, 'rv2_baseline.png')

# ─────────────────────────────────────────────────────────────────────────────
# Registry and main
# ─────────────────────────────────────────────────────────────────────────────

FIGURE_KEYS = [
    'lib_A', 'lib_B', 'lib_D', 'lib_C', 'lib_C_v2', 'coverage', 'baseline', 'timescales'
]

FIGURES = {
    'lib_A':    fig_lib_A,
    'lib_B':    fig_lib_B,
    'lib_D':    fig_lib_D,
    'lib_C':    fig_lib_C,
    'lib_C_v2': fig_lib_C_v2,
    'coverage': fig_coverage,
    'baseline': fig_baseline,
    'timescales': fig_timescales
}


def main():
    parser = argparse.ArgumentParser(
        description='Generate library figures for the v3 report.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Available keys: ' + '  '.join(FIGURE_KEYS),
    )
    parser.add_argument(
        'keys', nargs='*',
        help='Figure keys to generate (omit for all).',
    )
    args = parser.parse_args()

    keys = args.keys if args.keys else FIGURE_KEYS

    for k in keys:
        if k not in FIGURES:
            print(f'Unknown key: {k!r}.  Available: {", ".join(FIGURE_KEYS)}')
            sys.exit(1)

    for key in keys:
        print(f'\n=== {key} ===')
        FIGURES[key]()

    print('\nDone. Outputs in:', RESULTS)


if __name__ == '__main__':
    main()
