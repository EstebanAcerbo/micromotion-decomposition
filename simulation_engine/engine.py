"""
engine.py
=========
Shared simulation engine for Micromotion_Simul scripts.

Uses SQUARE (RecCell) cell geometry from ModelCellShape.  Perturbation
mechanisms:

  A — Electrode offset sweep  (substrate_offset)
  B — L-shaped corner Rb extension  (apply_l_corner_to_cell)
      Each live cell gets one of 4 corners (UL=0, UR=1, LL=2, LR=3) assigned
      an L-shaped cluster of l_width interior MEMBRANE pixels converted to TJ.
      This is the square-cell equivalent of the hex pivot: a subtle,
      asymmetric increase in local paracellular admittance at one corner.

Two-electrode note
------------------
The measurement uses two identical 350 um electrodes in series.
  Z_measured = 2 * Z_sim(one electrode)
  dZ_measured = 2 * dZ_sim  (both electrodes contribute)
Relative sensitivity dZ/Z is the same as single-electrode model.
"""

import sys
import os
import numpy as np

# ── path: add the simulation engine ──────────────────────────────────────────
_HERE       = os.path.dirname(os.path.abspath(__file__))
_ENGINE_DIR = _HERE  # GK/FD kernel files co-located in simulation_engine/
if _ENGINE_DIR not in sys.path:
    sys.path.insert(0, _ENGINE_DIR)

from culture_construction_vectorized import (
    construct_culture,
    construct_substrate_vectorized,
    RecCell,
    Monolayer,
)
from SolveMatrix      import V_vector
from Calc_Curr        import In_value
from CPE_GK_Funs      import Z_CPE_B
from ConstantsFile    import SystemConst


# ── Default physical parameters (MDCKII / two 350 um ECIS electrodes) ────────
#
# GK parameters: time-series GK fit to eb12 CH3 (confluent MDCKII, same
#   electrode as eb11).  Source: gk_fit_results (6).csv, CH3 column.
#   alpha=19.4, Rb=14.4, Cm=4.95e-6  ->  f_c = 2226 Hz
#
# CPE parameters: analytical CPE fit to eb17 bare electrode spectra.
#   Source: analytical_fit_results.csv, grand mean of CH4/CH6/CH7.
#   A_fit = 6.98e-9 was fitted to the total two-electrode Z_measured.
#   Each single-electrode simulation must use A = 2 * A_fit so that
#   each electrode contributes half the total CPE admittance.
#   Zn at 1 kHz ~ 28 Ohm (comparable to Zm; enters gamma_rc nonlinearly).
#
# R_constr convention — B_abs = 0.0 always:
#   The constriction resistance is a bare-electrode hardware property.
#   It does NOT enter the GK model or the perturbation simulations.
#   Added only post-hoc for experimental comparison:
#       Z_compared = 2 * (Z_FD + R_constr_per_electrode)
#   eb17 reference value:  R_constr_per_electrode ~ 1013 Ohm.
#   R_constr cancels exactly in dZ = Z_pert - Z_base.
# ─────────────────────────────────────────────────────────────────────────────
P = dict(
    CellType = 'rec',
    de       = 0.035,          # electrode diameter [cm]  (350 um, one of two)
    ac       = 4e-6,        # cell area [cm²]
    hw_ratio = 1.0,            # height/width ratio (square cells)
    n        = 20,             # pixel resolution (even; spacing_x = n)
    l_width  = 2,              # L-corner arm width in interior pixels
    alpha    = 19.4  + 0j,     # alpha [(Ohm*cm)^0.5]  eb12 CH3 GK fit
    Rb       = 14.4  + 0j,     # TJ resistance [Ohm*cm2]  eb12 CH3 GK fit
    Cm       = 4.95e-6 + 0j,   # membrane capacitance [F/cm2]  eb12 CH3 GK fit
    A        = 2*6.98e-9,      # CPE coefficient  [S*cm^-2*(rad/s)^-N]  2 x eb17 two-electrode fit (per-electrode value)
    B_abs    = 0.0,            # R_constr NOT in FD solve — added externally
    N        = 0.928,          # CPE exponent  eb17 fit
    solver   = 'pardiso',
)

CORNER_NAMES = ['UL', 'UR', 'LL', 'LR']   # corner index 0-3

# Frequency sweep: 100 Hz -> 1 MHz, 30 log-spaced points
FREQS = np.geomspace(100, 1e6, 25)


# ── Geometry builder ──────────────────────────────────────────────────────────

def build_geometry(params=None, extra_pad=0):
    """
    Compute all geometry-derived quantities for a confluent square-cell culture.

    Parameters
    ----------
    params : dict, optional
        Override any key in the default P dict.
    extra_pad : int, optional
        Additional cell-width padding beyond the default 5. Use extra_pad=2
        for shift_library to avoid domain-boundary artifacts at large
        electrode offsets.

    Returns
    -------
    dict with keys:
        p, stencil, d, side, dim, l, n_cols, n_rows,
        stats, Culture_vector, Substrate_vector
    """
    p = {**P, **(params or {})}

    l      = np.sqrt(p['ac'])            # square cell side length [cm]
    lado   = int(p['de'] / l) + 5 + extra_pad
    if lado % 2 == 0:
        lado += 1                        # ensure odd for symmetric electrode coverage
    n_cols = n_rows = lado

    Culture_vector, stats = construct_culture(
        p['CellType'], n_cols, n_rows,
        p['n'],
        simulation_type='confluent',
        height_width_ratio=p['hw_ratio'],
    )

    stencil  = RecCell(p['n'], p['hw_ratio'])
    cell_px  = stencil.membrane_count() + stencil.tj_count() / 2.0
    d        = np.sqrt(p['ac'] / cell_px)
    side     = stats['side']

    Substrate = construct_substrate_vectorized(side=side, de=p['de'], d=d)

    return dict(
        p              = p,
        stencil        = stencil,
        d              = d,
        l              = l,
        side           = side,
        dim            = stats['dim'],
        n_cols         = n_cols,
        n_rows         = n_rows,
        stats          = stats,
        Culture_vector = Culture_vector,
        Substrate_vector = Substrate,
    )


# ── Electrode substrate with offset ──────────────────────────────────────────

def substrate_offset(geo, x_px, y_px):
    """
    Return a Substrate_vector with the electrode centre shifted by
    (x_px, y_px) pixels from the grid centre.
    """
    side, de, d = geo['side'], geo['p']['de'], geo['d']
    xs   = np.arange(side, dtype=float) * d
    X, Y = np.meshgrid(xs, xs, indexing='xy')
    cx   = side * d / 2.0 + x_px * d
    cy   = side * d / 2.0 + y_px * d
    mask = (X - cx)**2 + (Y - cy)**2 <= (de / 2.0)**2
    return (SystemConst.ELECTRODE * mask.astype(np.int8)).ravel(order='F')


# ── L-corner cell modification ────────────────────────────────────────────────

def apply_l_corner_to_cell(culture_2d, cx_local, cy_local,
                            spacing_x, spacing_y, corner, l_width, side):
    """
    Convert l_width interior MEMBRANE pixels at one corner to TJ in-place.

    The L-shape covers a vertical arm (l_width rows, 1 col) and a horizontal
    arm (1 row, l_width cols) meeting at the corner.  Total pixels changed:
    2*l_width - 1 (corner overlap removed).

    Parameters
    ----------
    culture_2d  : 2-D int8 array  (side x side)
    cx_local    : int — column of cell centre in culture_2d
    cy_local    : int — row    of cell centre in culture_2d
    spacing_x/y : int — cell spacing in pixels
    corner      : 0=UL, 1=UR, 2=LL, 3=LR
    l_width     : int — arm length in pixels (default 2 -> 3 pixels total)
    side        : int — grid side length (for bounds check)
    """
    hx = spacing_x // 2
    hy = spacing_y // 2

    # Interior pixel bounds (inside the TJ border)
    r_min = cy_local - hy + 1
    r_max = cy_local + hy - 1
    c_min = cx_local - hx + 1
    c_max = cx_local + hx - 1

    if corner == 0:    # Upper-Left
        v_rows = range(r_min, min(r_min + l_width, r_max + 1))
        v_col  = c_min
        h_row  = r_min
        h_cols = range(c_min, min(c_min + l_width, c_max + 1))
    elif corner == 1:  # Upper-Right
        v_rows = range(r_min, min(r_min + l_width, r_max + 1))
        v_col  = c_max
        h_row  = r_min
        h_cols = range(max(c_max - l_width + 1, c_min), c_max + 1)
    elif corner == 2:  # Lower-Left
        v_rows = range(max(r_max - l_width + 1, r_min), r_max + 1)
        v_col  = c_min
        h_row  = r_max
        h_cols = range(c_min, min(c_min + l_width, c_max + 1))
    else:              # Lower-Right  (corner == 3)
        v_rows = range(max(r_max - l_width + 1, r_min), r_max + 1)
        v_col  = c_max
        h_row  = r_max
        h_cols = range(max(c_max - l_width + 1, c_min), c_max + 1)

    for r in v_rows:
        if 0 <= r < side and 0 <= v_col < side:
            if culture_2d[r, v_col] == SystemConst.MEMBRANE:
                culture_2d[r, v_col] = SystemConst.TJ

    for c in h_cols:
        if 0 <= h_row < side and 0 <= c < side:
            if culture_2d[h_row, c] == SystemConst.MEMBRANE:
                culture_2d[h_row, c] = SystemConst.TJ

    return culture_2d


def _apply_corners(geo, corner_per_cell, lw_per_cell=None):
    """
    Internal: build a modified culture_2d by applying per-cell corner assignments.

    corner_per_cell : array-like (n_cells,) — corner index 0-3 or -1 for none
    lw_per_cell     : array-like (n_cells,) or None — per-cell arm length in pixels.
                      If None, uses scalar geo['p']['l_width'] for all cells.
    Returns (Culture_vector flat, square dict)
    """
    p     = geo['p']
    mono  = Monolayer(p['CellType'], geo['n_cols'], geo['n_rows'],
                      p['n'], p['hw_ratio'])
    sq    = mono.rasterize_square()
    cult2d = sq['type_map'].copy()
    side   = sq['shape']
    ox, oy = sq['offset']
    sx     = mono.stencil.spacing_x
    sy     = mono.stencil.spacing_y
    lw_scalar = p.get('l_width', 2)

    for idx in range(mono.n_cells):
        corner = int(corner_per_cell[idx])
        if corner < 0 or mono.Culture_indices[idx] != SystemConst.ALIVE:
            continue
        lw = int(lw_per_cell[idx]) if lw_per_cell is not None else lw_scalar
        cx, cy = mono.centers[idx]
        apply_l_corner_to_cell(
            cult2d,
            int(cx - ox), int(cy - oy),
            sx, sy, corner, lw, side,
        )

    return cult2d.ravel(order='F'), sq


def build_l_corner_culture(geo, corner, all_cells=True):
    """
    Apply one specific L-corner configuration to all (or central) live cells.

    corner : 0=UL, 1=UR, 2=LL, 3=LR
    Returns (Culture_vector, square_dict)
    """
    p     = geo['p']
    mono  = Monolayer(p['CellType'], geo['n_cols'], geo['n_rows'],
                      p['n'], p['hw_ratio'])

    assignments = np.full(mono.n_cells, -1, dtype=np.int8)

    if all_cells:
        for idx in range(mono.n_cells):
            if mono.Culture_indices[idx] == SystemConst.ALIVE:
                assignments[idx] = corner
    else:
        cc = geo['n_cols'] // 2
        cr = geo['n_rows'] // 2
        idx = cc * geo['n_rows'] + cr
        assignments[idx] = corner

    return _apply_corners(geo, assignments)


def build_fixed_corners_culture(geo, corners):
    """
    Apply a fixed list of L-corners to every live cell.

    corners : list of int — corner indices (0=UL, 1=UR, 2=LL, 3=LR) to apply
    lw is read from geo['p']['l_width'].

    Returns (Culture_vector flat Fortran order, square_dict)
    """
    p      = geo['p']
    mono   = Monolayer(p['CellType'], geo['n_cols'], geo['n_rows'],
                       p['n'], p['hw_ratio'])
    side   = geo['side']
    sq     = mono.rasterize_square()
    cult2d = geo['Culture_vector'].reshape((side, side), order='F').copy()
    ox, oy = sq['offset']
    sx     = mono.stencil.spacing_x
    sy     = mono.stencil.spacing_y
    lw     = p.get('l_width', 2)

    for idx in range(mono.n_cells):
        if mono.Culture_indices[idx] != SystemConst.ALIVE:
            continue
        cx, cy = mono.centers[idx]
        for c in corners:
            apply_l_corner_to_cell(
                cult2d,
                int(cx - ox), int(cy - oy),
                sx, sy, c, lw, side,
            )

    return cult2d.ravel(order='F'), sq


def build_random_corner_culture(geo, seed=None):
    """
    Each live cell independently draws a uniform random corner (0-3).

    Returns (Culture_vector, square_dict, corner_states)
    where corner_states is (n_cells,) int8, -1 for dead cells.
    """
    p    = geo['p']
    mono = Monolayer(p['CellType'], geo['n_cols'], geo['n_rows'],
                     p['n'], p['hw_ratio'])
    rng  = np.random.default_rng(seed)

    assignments = np.full(mono.n_cells, -1, dtype=np.int8)
    for idx in range(mono.n_cells):
        if mono.Culture_indices[idx] == SystemConst.ALIVE:
            assignments[idx] = int(rng.integers(0, 4))

    Cv, sq = _apply_corners(geo, assignments)
    return Cv, sq, assignments


def build_random_size_corner_culture(geo, mu_lw=4, sigma_lw=2, seed=None):
    """
    Each live cell independently draws:
      corner  ~ Uniform({0, 1, 2, 3})
      l_width ~ clamp(round(Normal(mu_lw, sigma_lw)), 1, max_lw)

    where max_lw = spacing_x - 2  (full interior width, e.g. 8 for n=10).

    Parameters
    ----------
    mu_lw    : float — mean arm length [px]. Default 4 = half interior.
    sigma_lw : float — std of arm length. Default 2 (so 2σ spans [0, 8]).
    seed     : int or None

    Returns
    -------
    Cv              : Culture_vector (flat, Fortran order)
    sq              : rasterize_square() dict
    corner_states   : (n_cells,) int8, -1 for dead
    lw_states       : (n_cells,) int8, 0 for dead
    """
    p    = geo['p']
    mono = Monolayer(p['CellType'], geo['n_cols'], geo['n_rows'],
                     p['n'], p['hw_ratio'])
    rng  = np.random.default_rng(seed)

    # Maximum arm: full interior width (spacing_x - 2 border pixels = 8 for n=10)
    max_lw = geo['stencil'].spacing_x - 2   # e.g. 8

    corner_states = np.full(mono.n_cells, -1, dtype=np.int8)
    lw_states     = np.zeros(mono.n_cells,    dtype=np.int8)

    for idx in range(mono.n_cells):
        if mono.Culture_indices[idx] == SystemConst.ALIVE:
            corner_states[idx] = int(rng.integers(0, 4))
            lw_raw = int(round(float(rng.normal(mu_lw, sigma_lw))))
            lw_states[idx] = max(1, min(max_lw, lw_raw))

    Cv, sq = _apply_corners(geo, corner_states, lw_per_cell=lw_states)
    return Cv, sq, corner_states, lw_states


def build_sparse_corner_culture(geo, lw_max=2, probs=None, seed=None):
    """
    Micro-perturbation library: each live cell independently draws
    l_width ~ Discrete({0, 1, ..., lw_max}) with given probabilities.

    lw=0 → cell receives no L-corner modification (corner index = -1).
    lw>0 → random corner in {0,1,2,3} with the sampled arm width.

    Parameters
    ----------
    lw_max : int — maximum arm width (default 2).
    probs  : array-like (lw_max+1,) or None.
             Probability for each lw in {0, 1, ..., lw_max}.
             If None, uniform over {0, 1, ..., lw_max}.
    seed   : int or None

    Returns
    -------
    Cv              : Culture_vector (flat, Fortran order)
    sq              : rasterize_square() dict
    corner_states   : (n_cells,) int8, -1 for skipped / dead
    lw_states       : (n_cells,) int8, 0 for skipped / dead
    """
    p    = geo['p']
    mono = Monolayer(p['CellType'], geo['n_cols'], geo['n_rows'],
                     p['n'], p['hw_ratio'])
    rng  = np.random.default_rng(seed)

    lw_choices = np.arange(0, lw_max + 1, dtype=int)
    if probs is None:
        probs = np.ones(lw_max + 1) / (lw_max + 1)
    else:
        probs = np.asarray(probs, dtype=float)
        probs = probs / probs.sum()   # normalise

    corner_states = np.full(mono.n_cells, -1, dtype=np.int8)
    lw_states     = np.zeros(mono.n_cells,    dtype=np.int8)

    for idx in range(mono.n_cells):
        if mono.Culture_indices[idx] == SystemConst.ALIVE:
            lw = int(rng.choice(lw_choices, p=probs))
            if lw > 0:
                corner_states[idx] = int(rng.integers(0, 4))
                lw_states[idx]     = lw
            # else: corner_states[idx] = -1  → no modification

    Cv, sq = _apply_corners(geo, corner_states, lw_per_cell=lw_states)
    return Cv, sq, corner_states, lw_states


def apply_brick_offset(geo, delta):
    """
    Return a modified Culture_vector where alternating cell-columns are shifted
    vertically (in y) by delta * spacing_y pixels — the brick-wall (Library C)
    perturbation.

    Cell-columns 1, 3, 5, ... (0-indexed from left) are shifted;
    even columns remain unshifted.

    Implementation
    --------------
    The baseline culture uses center-to-center geometry (side = (N-1)*sx + 1),
    so edge cells are half-clipped.  np.roll on raw pixel bands would wrap
    those clipped halves to the wrong boundary, splitting cells mid-stencil.

    Instead we extend the culture_2d by one full cell period (sy rows) at the
    bottom using periodic tiling — since all cells are identical the pattern
    repeats exactly every sy pixels.  For each odd cell-column we then copy
    the pixel band from the extended array starting at row offset_px, giving
    complete cells throughout with correct boundary continuity.

    Parameters
    ----------
    geo   : dict from build_geometry()
    delta : float in [0, 0.5] — fractional offset.
            0   = square packing (no change).
            0.5 = full brick-wall (half-cell y-shift, offset_px = sy//2).

    Returns
    -------
    Culture_vector : 1-D int8 array (flat, Fortran order)
    """
    side = geo['side']
    sx   = geo['stencil'].spacing_x
    sy   = geo['stencil'].spacing_y

    offset_px = int(round(delta * sy))
    if offset_px == 0:
        return geo['Culture_vector'].copy()

    # Reshape to 2-D (row = y, col = x), Fortran-order flattening
    cult2d = geo['Culture_vector'].reshape((side, side), order='F')

    # Extend by one full cell period at BOTH ends (periodic tiling).
    # Cell pattern repeats every sy rows, so:
    #   cult2d[side-sy:]  is the periodic predecessor (top extension)
    #   cult2d[:sy]       is the periodic successor  (bottom extension)
    # This handles both positive (downward) and negative (upward) offsets.
    big_cult = np.vstack([
        cult2d[side - sy:, :],   # sy rows prepended (top)
        cult2d,                  # original (side rows)
        cult2d[:sy, :],          # sy rows appended  (bottom)
    ])
    # shape: (side + 2*sy, side)
    # cult2d rows map to big_cult rows [sy : sy+side]
    # odd-column slice: big_cult[sy + offset_px : sy + offset_px + side]

    result = cult2d.copy()
    hx = sx // 2

    for col_idx in range(geo['n_cols']):
        if col_idx % 2 == 0:
            continue  # even columns unchanged

        # x-centre of this cell column in culture_2d coords (center-to-center,
        # so first cell center is at px=0, second at px=sx, etc.)
        cx = col_idx * sx
        c0 = max(0, cx - hx)
        c1 = min(side, cx + hx + 1)

        # Slice the shifted band; sy + offset_px is always in [0, 2*sy]
        r0 = sy + offset_px
        result[:, c0:c1] = big_cult[r0 : r0 + side, c0:c1]

    return result.ravel(order='F')


# ── Solver (returns Z only) ───────────────────────────────────────────────────

def solve_Z(geo, f, Sub=None, Cult=None):
    """
    Assemble and solve the FD system; return complex Z(f) for the electrode.

    Parameters
    ----------
    geo  : dict from build_geometry()
    f    : ndarray — frequency array [Hz]
    Sub  : ndarray, optional — Substrate_vector override
    Cult : ndarray, optional — Culture_vector override

    Returns
    -------
    Z : ndarray, shape (len(f),), complex
    """
    p  = geo['p']
    ae = np.pi * p['de']**2 / 4.0
    Zn = ae * Z_CPE_B(f, p['A'], 0, p['N']) + p.get('B_abs', 0.0)

    S = geo['Substrate_vector'] if Sub  is None else Sub
    C = geo['Culture_vector']   if Cult is None else Cult

    V, _ = V_vector(
        geo['d'], geo['dim'],
        geo['stats']['membrane_per_cell'],
        geo['stats']['tj_per_cell'],
        f, Zn, np.sqrt(p['ac'] / np.pi),
        p['alpha'], p['Rb'], p['Cm'],
        S, C,
        solver_mode=p['solver'],
    )

    I = In_value(V, geo['d'], Zn, 1.0 + 0j, S)
    return (1.0 + 0j) / I


def solve_full(geo, f, Sub=None, Cult=None):
    """
    Like solve_Z but also returns V_solutions and Zn for plotting.

    Returns
    -------
    Z           : ndarray (n_freq,) complex
    V_solutions : ndarray (n_freq, dim) complex
    Zn          : ndarray (n_freq,) complex
    """
    p  = geo['p']
    ae = np.pi * p['de']**2 / 4.0
    Zn = ae * Z_CPE_B(f, p['A'], 0, p['N']) + p.get('B_abs', 0.0)

    S = geo['Substrate_vector'] if Sub  is None else Sub
    C = geo['Culture_vector']   if Cult is None else Cult

    V, _ = V_vector(
        geo['d'], geo['dim'],
        geo['stats']['membrane_per_cell'],
        geo['stats']['tj_per_cell'],
        f, Zn, np.sqrt(p['ac'] / np.pi),
        p['alpha'], p['Rb'], p['Cm'],
        S, C,
        solver_mode=p['solver'],
    )

    I = In_value(V, geo['d'], Zn, 1.0 + 0j, S)
    Z = (1.0 + 0j) / I
    return Z, V, Zn


# ── Diagnostic plot ───────────────────────────────────────────────────────────

def quick_plot(geo, Cult=None, Sub=None, f_plot=1000.0, title='', savepath=None):
    """
    Diagnostic figure: culture map (left) + |ΔV / V₀|_max [%] field (right).

    Performs two FD solves at f_plot Hz:
      1. Baseline  — geo's default Culture_vector and Substrate_vector.
      2. Perturbed — using the Cult / Sub overrides (if provided).

    The right panel shows where in the electrode plane the perturbation
    changes the electric potential field, normalised to the baseline max.

    Parameters
    ----------
    geo      : dict from build_geometry()
    Cult     : ndarray, optional — perturbed Culture_vector (flat, Fortran order)
    Sub      : ndarray, optional — perturbed Substrate_vector
    f_plot   : float — frequency for ΔV map [Hz].  Default 1000.
    title    : str   — figure suptitle; omitted if empty.
    savepath : str or None — if given, save PNG at 150 dpi and close figure.

    Returns
    -------
    (fig, axes) if savepath is None, else None.
    """
    # Lazy matplotlib import — keeps module lightweight when plotting is unused.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from SimulSheet_Funcs import Map_Electrode

    p    = geo['p']
    de   = p['de']
    d    = geo['d']
    side = geo['side']

    f_arr = np.array([f_plot])

    # ── Two FD solves ─────────────────────────────────────────────────────────
    _, V_base, _ = solve_full(geo, f_arr)                        # baseline
    _, V_pert, _ = solve_full(geo, f_arr, Sub=Sub, Cult=Cult)    # perturbed

    V_b = V_base[0]   # shape (dim,) complex
    V_p = V_pert[0]

    # ── ΔV normalised field ───────────────────────────────────────────────────
    V_b2d  = V_b.reshape((side, side), order='F')
    V_p2d  = V_p.reshape((side, side), order='F')
    dV_pct = 100.0 * np.abs(V_p2d - V_b2d) / (np.abs(V_b2d).max() + 1e-30)

    # Physical extent [µm] — matches Map_Electrode convention (d in cm, 1e4 µm/cm)
    half_um = 1e4 * side * d / 2.0
    extent  = [-half_um, half_um, -half_um, half_um]

    # ── Combined culture+substrate vector for Map_Electrode ───────────────────
    S        = geo['Substrate_vector'] if Sub  is None else Sub
    C        = geo['Culture_vector']   if Cult is None else Cult
    combined = (2 * C.astype(np.int32) + S.astype(np.int32))

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Left — culture / electrode map
    Map_Electrode(axes[0], combined, de, d, side)
    axes[0].set_title('Culture map', fontsize=11)

    # Right — |ΔV / V₀|_max field
    vmax = float(dV_pct.max()) or 1.0
    im   = axes[1].imshow(
        dV_pct, origin='lower', extent=extent,
        cmap='inferno', vmin=0, vmax=vmax, aspect='equal',
    )
    plt.colorbar(im, ax=axes[1], label=r'$|\Delta V| / |V_0|_{\max}$  [%]')
    axes[1].set_xlabel(r'Position [$\mu$m]', fontsize=10)
    axes[1].set_ylabel(r'Position [$\mu$m]', fontsize=10)
    axes[1].set_title(
        fr'$|\Delta V / V_0|$  @  {f_plot:.0f} Hz', fontsize=11,
    )

    if title:
        fig.suptitle(title, fontsize=11, y=1.01)
    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return None
    return fig, axes
