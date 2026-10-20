"""
phase_space_coverage.py
=======================
Diagnostic plot showing the coverage of the (dR, dX) phase space at 1 kHz by
all simulation library modes (each mode at coefficient c=1, i.e. raw dZ vectors),
overlaid with the experimental dZ(t) trajectories from multiple electrodes.

The goal is to visually determine whether experimental fluctuations fall WITHIN
the convex hull of the library modes or far outside it (indicating a systematic
amplitude gap that would require coefficient rescaling before NNLS reconstruction).

Usage
-----
    python phase_space_coverage.py [--nosave] [--noshow]

Outputs
-------
    results/phase_space_coverage.png  (unless --nosave)
    Text report to stdout.

Data sources
------------
    results/library_A_shift.npz
    results/library_B_det.npz
    results/library_B_rand.npz
    results/library_B_micro.npz
    results/library_B_rb.npz
    results/library_C_brickwall.npz
    Experimental .dat files via load_experimental.load_dat / micromotion_stats
"""

import argparse
import os
import sys

import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull, Delaunay

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR  = os.path.join(SCRIPT_DIR, '..', 'results')
OUTPUT_PATH  = os.path.join(RESULTS_DIR, 'phase_space_coverage.png')

TARGET_FREQ  = 1000.0   # Hz — evaluation frequency

# ---------------------------------------------------------------------------
# Library catalogue: (key, filename_stem, colour, marker_size, alpha, zorder, marker)
# ---------------------------------------------------------------------------
LIBRARY_SPECS = [
    ('A_shift',     'library_A_shift',     'steelblue',   15, 0.4, 2, 'o'),
    #('B_det',       'library_B_det',       'firebrick',   30, 0.7, 3, 'o'),
    #('B_rand',      'library_B_rand',      'olive',       20, 0.5, 2, 'o'),
    #('B_micro',     'library_B_micro',     'darkorange',  20, 0.5, 2, 'o'),
    ('B_rb',        'library_B_rb',        'saddlebrown', 30, 0.7, 3, 'D'),
    ('D_rb',        'library_D_rb',        'teal',        30, 0.8, 4, 's'),
    ('C_brickwall', 'library_C_brickwall', 'purple',      20, 0.5, 2, '^'),
]

# Fast lookup: key -> (color, ms, alpha, zorder, marker)
_STYLE = {k: (c, ms, a, zo, mk) for k, _, c, ms, a, zo, mk in LIBRARY_SPECS}

EXPERIMENTAL_STEMS = ['eb11', 'eb04', 'eb13', 'eb15']

# ---------------------------------------------------------------------------
# Library loading
# ---------------------------------------------------------------------------

def _load_library(stem):
    """
    Load one library npz from results/, compute (dR, dX) at TARGET_FREQ.

    Returns
    -------
    dR   : ndarray (N_modes,)
    dX   : ndarray (N_modes,)
    meta : dict — extra per-mode metadata (e.g. 'k_rb_arr' for B_rb)
    """
    path = os.path.join(RESULTS_DIR, stem + '.npz')
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    data   = np.load(path, allow_pickle=False)
    freqs  = data['freqs']
    Z_raw  = data['Z']
    Z_base = data['Z_baseline']

    n_freq = len(freqs)
    i_f    = int(np.argmin(np.abs(freqs - TARGET_FREQ)))

    # Flatten to (N_modes, n_freq) regardless of original shape
    Z_flat = Z_raw.reshape(-1, n_freq)

    dZ = Z_flat[:, i_f] - Z_base[i_f]

    meta = {}
    if 'k_rb_arr' in data:
        meta['k_rb_arr'] = data['k_rb_arr'].copy()

    return dZ.real.copy(), dZ.imag.copy(), meta


# ---------------------------------------------------------------------------
# Convex hull helpers
# ---------------------------------------------------------------------------

def _hull_polygon(points):
    """
    Return hull vertex coordinates as a closed (N+1, 2) array, or None if
    fewer than 3 non-collinear points.
    """
    if len(points) < 3:
        return None
    try:
        hull = ConvexHull(points)
        verts = np.vstack([points[hull.vertices], points[hull.vertices[0]]])
        return verts
    except Exception:
        return None


def _in_hull(test_pts, hull_pts):
    """
    Boolean mask: True for each row of test_pts inside the convex hull of
    hull_pts.  Uses Delaunay triangulation.
    """
    if len(hull_pts) < 3:
        return np.zeros(len(test_pts), dtype=bool)
    try:
        tri = Delaunay(hull_pts)
        return tri.find_simplex(test_pts) >= 0
    except Exception:
        return np.zeros(len(test_pts), dtype=bool)


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
         two convex sets is the convex hull of all pairwise vertex sums
         (Minkowski sum theorem for polytopes).
      2. Enumerate all combinations of one vertex from each hull (Cartesian product).
      3. Sum the selected vertices and take the convex hull of all sums.

    Complexity: product of hull sizes across all libraries.  For four libraries
    each with up to ~40 vertices the product is at most 40^4 = 2,560,000 points,
    which is tractable.
    """
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
    # Use itertools.product for clarity; it handles arbitrary numbers of sets.
    import itertools
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


# ---------------------------------------------------------------------------
# Experimental data loading
# ---------------------------------------------------------------------------

def load_experimental():
    """
    Load detrended dR(t), dX(t) for each stem in EXPERIMENTAL_STEMS.

    Returns
    -------
    dict: stem -> {'dR': ndarray, 'dX': ndarray, 't': ndarray}
    Missing files are skipped with a printed warning.
    """
    # Ensure SCRIPT_DIR is on the path so load_experimental can be imported
    # when the script is invoked from a different working directory.
    _exp_dir = os.path.join(SCRIPT_DIR, '..', 'experimental_signal_analysis')
    if _exp_dir not in sys.path:
        sys.path.insert(0, _exp_dir)

    try:
        from load_experimental import load_dat, micromotion_stats
    except ImportError as exc:
        print(f'[WARNING] Cannot import load_experimental: {exc}')
        return {}

    out = {}
    for stem in EXPERIMENTAL_STEMS:
        try:
            t, R, X, C, f_meas = load_dat(stem)
            stats = micromotion_stats(
                R, X, t,
                detrend='moving_avg',
                window_sec=900,
                group='confluent',
            )
            out[stem] = {
                'dR': stats['dR'],
                'dX': stats['dX'],
                't':  stats['t'],
            }
        except FileNotFoundError as exc:
            print(f'[WARNING] Experimental file not found for {stem}: {exc}')
        except Exception as exc:
            print(f'[WARNING] Could not load {stem}: {exc}')

    return out


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------

def print_report(lib_data, exp_data, all_lib_pts, mink_verts=None):
    """
    Print a summary table to stdout.

    Parameters
    ----------
    lib_data     : list of (key, dR, dX, meta) tuples
    exp_data     : dict stem -> {'dR', 'dX', 't'}
    all_lib_pts  : (N_total, 2) array of all library (dR, dX) points
    mink_verts   : (M, 2) array or None — Minkowski sum hull vertices
    """
    header_lib = (
        f'{"Library":<14}  {"N_modes":>8}  {"dZ_min [Ohm]":>13}'
        f'  {"dZ_max [Ohm]":>13}  {"mean_dR":>10}  {"mean_dX":>10}'
    )
    print()
    print(header_lib)
    print('-' * len(header_lib))

    for key, dR, dX, _meta in lib_data:
        adz = np.hypot(dR, dX)
        print(
            f'{key:<14}  {len(dR):>8d}  {adz.min():>13.4f}'
            f'  {adz.max():>13.4f}  {dR.mean():>10.4f}  {dX.mean():>10.4f}'
        )

    print()
    header_exp = (
        f'{"Experiment":<12}  {"N_pts":>8}  {"std_dR [Ohm]":>14}'
        f'  {"std_dX [Ohm]":>14}  {"max|dZ| [Ohm]":>15}'
    )
    print(header_exp)
    print('-' * len(header_exp))

    for stem, ed in exp_data.items():
        dR_e, dX_e = ed['dR'], ed['dX']
        adz_e = np.hypot(dR_e, dX_e)
        print(
            f'{stem:<12}  {len(dR_e):>8d}  {dR_e.std():>14.4f}'
            f'  {dX_e.std():>14.4f}  {adz_e.max():>15.4f}'
        )

    # Fraction of experimental points inside combined library hull
    if len(all_lib_pts) >= 3:
        print()
        print('Fraction of experimental dZ(t) inside combined library hull:')
        for stem, ed in exp_data.items():
            test_pts = np.column_stack([ed['dR'], ed['dX']])
            mask     = _in_hull(test_pts, all_lib_pts)
            frac     = mask.mean() * 100.0
            print(f'  {stem:<10}  {frac:.1f}%  ({mask.sum()}/{len(mask)} pts)')

    # Fraction of experimental points inside Minkowski sum hull
    if mink_verts is not None and len(mink_verts) >= 3:
        print()
        print('Fraction of experimental dZ(t) inside Minkowski sum hull '
              '(true reachable region):')
        for stem, ed in exp_data.items():
            test_pts  = np.column_stack([ed['dR'], ed['dX']])
            mask_mink = _in_mink_hull(test_pts, mink_verts)
            frac_mink = mask_mink.mean() * 100.0
            print(f'  {stem:<10}  {frac_mink:.1f}%  '
                  f'({mask_mink.sum()}/{len(mask_mink)} pts)')

    print()


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(lib_data, exp_data, all_lib_pts, nosave, noshow,
                mink_verts=None):
    """
    Create and optionally save/show the phase-space coverage figure.

    Left panel : (dR, dX) scatter at 1 kHz — library modes + experimental
                 trajectories, individual-library convex hulls overlaid.
    Right panel: Minkowski sum hull (true reachable region) as a filled polygon,
                 overlaid with the experimental (dR, dX) scatter per electrode.
                 Fraction of experimental points inside the Minkowski hull is
                 annotated on the panel.

    Parameters
    ----------
    mink_verts : (M, 2) ndarray or None
        Open-polygon vertices of the Minkowski sum hull returned by
        compute_minkowski_hull().  If None the right panel falls back to a
        simple scatter of all experimental points for reference.
    """
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 7))
    fig.subplots_adjust(wspace=0.38)

    cmap_time = plt.cm.plasma

    # ── Left panel ──────────────────────────────────────────────────────────

    # Minkowski sum hull outline (replaces the former combined-hull outline)
    mink_closed_left = None
    if mink_verts is not None and len(mink_verts) >= 3:
        mink_closed_left = np.vstack([mink_verts, mink_verts[0]])
        ax_left.plot(
            mink_closed_left[:, 0], mink_closed_left[:, 1],
            color='black', linestyle='--', linewidth=1.5,
            alpha=0.3, zorder=5,
        )

    # Per-library hull fill + scatter
    legend_handles = []
    for key, dR, dX, meta in lib_data:
        color, ms, alpha, zo, marker = _STYLE.get(key, ('grey', 15, 0.5, 2, 'o'))
        pts  = np.column_stack([dR, dX])
        poly = _hull_polygon(pts)
        if poly is not None:
            ax_left.fill(
                poly[:, 0], poly[:, 1],
                color=color, alpha=0.08, linewidth=0, zorder=2,
            )

        # B_rb: colour each dot by its K_RB value
        if key == 'B_rb' and 'k_rb_arr' in meta:
            norm_rb   = mcolors.Normalize(
                vmin=meta['k_rb_arr'].min(), vmax=meta['k_rb_arr'].max()
            )
            cmap_rb   = plt.cm.RdYlGn
            colors_rb = cmap_rb(norm_rb(meta['k_rb_arr']))
            ax_left.scatter(dR, dX, s=ms, c=colors_rb, alpha=alpha, zorder=zo,
                            marker=marker)
            proxy = mlines.Line2D(
                [], [], color=color,
                marker=marker, linestyle='None',
                markersize=4, alpha=min(alpha + 0.2, 1.0),
                label=f'{key} (N={len(dR)}, K_RB sweep)',
            )
        else:
            ax_left.scatter(dR, dX, s=ms, color=color, alpha=alpha, zorder=zo,
                            marker=marker)
            # Proxy handle for legend (PathCollection does not carry a label when
            # added manually, so build a Line2D proxy for cleaner legend control)
            proxy = mlines.Line2D(
                [], [], color=color,
                marker=marker, linestyle='None',
                markersize=4, alpha=min(alpha + 0.2, 1.0),
                label=f'{key} (N={len(dR)})',
            )
        legend_handles.append(proxy)

    # Build K_RB sub-legend for B_rb (lower-left corner)
    for key, dR, dX, meta in lib_data:
        if key == 'B_rb' and 'k_rb_arr' in meta:
            k_rb_unique = np.unique(meta['k_rb_arr'])
            norm_rb = mcolors.Normalize(vmin=k_rb_unique.min(), vmax=k_rb_unique.max())
            cmap_rb = plt.cm.RdYlGn
            krb_handles = []
            for kv in k_rb_unique:
                col = cmap_rb(norm_rb(kv))
                if kv > 0:
                    sign_note = '(+: tighter junction)'
                elif kv < 0:
                    sign_note = '(−: looser junction)'
                else:
                    sign_note = ''
                krb_handles.append(
                    mlines.Line2D([], [], color=col, marker='D', linestyle='None',
                                  markersize=5, label=f'K_RB = {kv:+.2g}  {sign_note}')
                )
            krb_legend = ax_left.legend(
                handles=krb_handles,
                title='B_rb: Rb coupling strength\n(+ = Rb ↑ as junction opens\n− = Rb ↓ as junction opens)',
                loc='lower left',
                fontsize=7,
                title_fontsize=6.5,
                framealpha=0.85,
                edgecolor='saddlebrown',
            )
            ax_left.add_artist(krb_legend)
            break

    # Experimental trajectories — time-coloured scatter (behind library dots)
    exp_proxy_handles = []
    for stem, ed in exp_data.items():
        dR_e, dX_e, t_e = ed['dR'], ed['dX'], ed['t']
        span = t_e.max() - t_e.min()
        t_norm = (t_e - t_e.min()) / max(span, 1.0)
        ax_left.scatter(
            dR_e, dX_e,
            c=t_norm, cmap=cmap_time,
            marker='x', s=8, linewidths=0.5, alpha=0.3, zorder=1,
            vmin=0.0, vmax=1.0,
        )
        proxy_exp = mlines.Line2D(
            [], [], color=cmap_time(0.5),
            marker='x', linestyle='None',
            markersize=5, alpha=0.6,
            label=f'{stem} (exp)',
        )
        exp_proxy_handles.append(proxy_exp)

    # Colorbar for time axis
    if exp_data:
        sm = plt.cm.ScalarMappable(
            cmap=cmap_time, norm=mcolors.Normalize(vmin=0, vmax=1)
        )
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax_left, fraction=0.03, pad=0.02)
        cb.set_label('Normalised time', fontsize=8)
        cb.set_ticks([0.0, 0.5, 1.0])
        cb.set_ticklabels(['t_min', 't_mid', 't_max'])

    # Origin crosshairs
    ax_left.axvline(0, color='grey', linestyle='--', linewidth=0.5, zorder=0)
    ax_left.axhline(0, color='grey', linestyle='--', linewidth=0.5, zorder=0)

    ax_left.set_xlabel('dR [Ohm]', fontsize=10)
    ax_left.set_ylabel('dX [Ohm]', fontsize=10)
    ax_left.set_title(
        'Phase space coverage — library modes vs experimental dZ(t) at 1 kHz',
        fontsize=9,
    )

    # Build legend
    all_handles = []
    if mink_closed_left is not None:
        hull_proxy = mlines.Line2D(
            [], [], color='black', linestyle='--',
            linewidth=1.5, alpha=0.3, label='Minkowski sum hull',
        )
        all_handles.append(hull_proxy)
    all_handles.extend(legend_handles)
    all_handles.extend(exp_proxy_handles)
    ax_left.legend(
        handles=all_handles,
        loc='upper right', fontsize=7, framealpha=0.7, ncol=2,
    )

    # ── Right panel ──────────────────────────────────────────────────────────
    # Shows the Minkowski sum hull (true reachable region when one mode from
    # each library is summed at every time step) overlaid with experimental
    # (dR, dX) scatter for each electrode.

    # Experimental colour map — one distinct colour per electrode.
    exp_cmap_names = ['#E91E63', '#2196F3', '#FF9800', '#4CAF50']

    # Draw Minkowski sum hull as a filled grey polygon.
    if mink_verts is not None and len(mink_verts) >= 3:
        mink_closed = np.vstack([mink_verts, mink_verts[0]])
        ax_right.fill(
            mink_closed[:, 0], mink_closed[:, 1],
            color='lightgrey', alpha=0.60, linewidth=0, zorder=1,
        )
        ax_right.plot(
            mink_closed[:, 0], mink_closed[:, 1],
            color='dimgrey', linestyle='-', linewidth=1.5,
            alpha=0.85, zorder=2,
            label='Minkowski sum (true reachable region)',
        )

    # Individual-library combined hull for comparison reference.
    ind_poly = _hull_polygon(all_lib_pts) if len(all_lib_pts) >= 3 else None
    if ind_poly is not None:
        ax_right.plot(
            ind_poly[:, 0], ind_poly[:, 1],
            color='black', linestyle='--', linewidth=1.2,
            alpha=0.45, zorder=3,
            label='Individual modes hull',
        )

    # Experimental scatter per electrode.
    right_handles = []
    annotation_lines = []
    for i, (stem, ed) in enumerate(exp_data.items()):
        dR_e  = ed['dR']
        dX_e  = ed['dX']
        color = exp_cmap_names[i % len(exp_cmap_names)]
        ax_right.scatter(
            dR_e, dX_e,
            color=color, marker='x', s=8, linewidths=0.6,
            alpha=0.35, zorder=4,
        )
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

    ax_right.axvline(0, color='grey', linestyle='--', linewidth=0.5, zorder=0)
    ax_right.axhline(0, color='grey', linestyle='--', linewidth=0.5, zorder=0)

    ax_right.set_xlabel('dR [Ohm]', fontsize=10)
    ax_right.set_ylabel('dX [Ohm]', fontsize=10)
    ax_right.set_title(
        'Minkowski sum hull (true reachable region)\nvs experimental dZ(t)',
        fontsize=9,
    )

    # Legend: hull patches first, then per-electrode proxies.
    legend_right = []
    if mink_verts is not None and len(mink_verts) >= 3:
        legend_right.append(
            mpatches.Patch(facecolor='lightgrey', edgecolor='dimgrey',
                           linewidth=1.5, label='Minkowski sum (true reachable region)')
        )
    if ind_poly is not None:
        legend_right.append(
            mlines.Line2D([], [], color='black', linestyle='--',
                          linewidth=1.2, alpha=0.45, label='Individual modes hull')
        )
    legend_right.extend(right_handles)
    ax_right.legend(
        handles=legend_right,
        loc='upper right', fontsize=7, framealpha=0.75, ncol=1,
    )

    # Annotation box: per-electrode coverage fractions.
    if annotation_lines:
        box_text = '\n'.join(annotation_lines)
        ax_right.text(
            0.02, 0.03,
            box_text,
            transform=ax_right.transAxes,
            fontsize=7, verticalalignment='bottom',
            bbox=dict(
                boxstyle='round,pad=0.3',
                facecolor='lightyellow',
                alpha=0.85,
                edgecolor='grey',
            ),
        )

    fig.tight_layout()

    if not nosave:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight')
        print(f'Figure saved to {OUTPUT_PATH}')

    if not noshow:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Phase-space coverage diagnostic at 1 kHz.'
    )
    parser.add_argument(
        '--nosave', action='store_true',
        help='Do not save the figure to disk.',
    )
    parser.add_argument(
        '--noshow', action='store_true',
        help='Do not call plt.show().',
    )
    args = parser.parse_args()

    # ── Load libraries ──────────────────────────────────────────────────────
    lib_data = []   # list of (key, dR, dX, meta)

    for key, fstem, _color, _ms, _alpha, _zo, _mk in LIBRARY_SPECS:
        try:
            dR, dX, meta = _load_library(fstem)
            lib_data.append((key, dR, dX, meta))
            print(f'Loaded {key:<14}: {len(dR):>6} modes')
        except FileNotFoundError as exc:
            print(f'[WARNING] Library file not found: {exc}')
        except Exception as exc:
            print(f'[WARNING] Failed to load {key}: {exc}')

    # Combined array of all library (dR, dX) points
    if lib_data:
        all_lib_pts = np.vstack([
            np.column_stack([dR, dX]) for _, dR, dX, _m in lib_data
        ])
    else:
        all_lib_pts = np.empty((0, 2))

    # ── Minkowski sum hull ───────────────────────────────────────────────────
    # The true reachable set in reconstruction is the Minkowski sum of all
    # library clouds (one mode per library is added at each time step).
    print('Computing Minkowski sum hull...')
    if lib_data:
        point_sets = [np.column_stack([dR, dX]) for _, dR, dX, _m in lib_data]
        mink_verts = compute_minkowski_hull(point_sets)
        if mink_verts is not None:
            print(f'  Minkowski hull: {len(mink_verts)} vertices')
        else:
            print('  [WARNING] Minkowski hull computation returned None.')
    else:
        mink_verts = None

    # ── Load experimental data ──────────────────────────────────────────────
    exp_data = load_experimental()

    # ── Text report ─────────────────────────────────────────────────────────
    print_report(lib_data, exp_data, all_lib_pts, mink_verts=mink_verts)

    # ── Figure ──────────────────────────────────────────────────────────────
    make_figure(lib_data, exp_data, all_lib_pts, args.nosave, args.noshow,
                mink_verts=mink_verts)


if __name__ == '__main__':
    main()
