"""
reconstruct_v4.py  --  Sparse Viterbi Temporal Regularization
==============================================================
Extends the greedy sequential state matching of v3 with a Viterbi
temporal regularization step per library.  The greedy baseline is always
computed first (fast, < 3 s per electrode) and is used as a comparison
reference.  The Viterbi pass enforces temporal smoothness by penalizing
large jumps in state space with a hard constraint (inf cost beyond
max_jump) rather than a soft quadratic penalty.

Algorithm
---------
For each library L in order:
    1. Compute residual r(t) = y(t) - sum of contributions from
       previously decoded libraries.
    2. Build a neighbourhood graph: mode j is reachable from mode i if
       dist(i, j) <= max_jump_L (otherwise transition cost = inf).
    3. Run Viterbi over (T, M_L) to find the globally optimal sequence
       s*(0..T-1) = argmin  sum_t obs_cost(s_t, r_t)
                            subject to  trans_cost(s_{t-1}, s_t) < inf.
    4. Subtract s* contribution from residual for next library.

Null mode (index 0 after add_null_modes):
    dZ = [0, 0].  When the residual is small the null mode is optimal,
    meaning the library contributes nothing at that step — physically
    correct.

Memory management:
    In-memory Viterbi when T * M * 4 bytes < 500 MB.
    Chunked checkpointing otherwise (re-runs forward per chunk).

Usage
-----
    python reconstruct_v4.py --stem eb11 [--savefig] [--show] [--cal]
    python reconstruct_v4.py --stem eb11 --no_reg          # same as v3
    python reconstruct_v4.py --stem eb11 --max_jump_A 2.0 --max_jump_B 0.2 --savefig --show
"""

import os
import sys
import re
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'experimental_signal_analysis'))

from load_experimental import load_dat, EXPERIMENT_MAP

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS = os.path.join(_HERE, '..', 'results')
os.makedirs(RESULTS, exist_ok=True)

# ── Library definitions ───────────────────────────────────────────────────────
LIBRARY_FILES = {
    'A_shift':     'library_A_shift.npz',
    'B_rb':        'library_B_rb.npz',
    'D_rb':        'library_D_rb.npz',
    'C_brickwall': 'library_C_brickwall.npz',
}

LIBRARY_COLORS = {
    'A_shift':     'steelblue',
    'B_rb':        'red',
    'D_rb':        'teal',
    'C_brickwall': 'purple',
}

DEFAULT_ORDER = ['B_rb',  'D_rb','A_shift',  'C_brickwall']


# ── Library loading ───────────────────────────────────────────────────────────

def load_libraries(freq_target=1000.0):
    """
    Load perturbation libraries for A_shift, B_rb, D_rb, C_brickwall.

    Returns
    -------
    libs   : dict  name -> dict with keys:
                 dZ      : (N_modes, 2)  [dR, dX] at freq_target
                 labels  : list[str]     mode label per mode
                 dx_px   : (N_modes,) int or None   (A_shift only)
                 dy_px   : (N_modes,) int or None   (A_shift only)
                 k_rb    : (N_modes,) float or None  (B_rb only)
                 lw      : (N_modes,) int or None    (B_rb only)
                 d_alpha : (N_modes,) float or None  (D_rb only)
                 d_Rb    : (N_modes,) float or None  (D_rb only)
                 d_Cm    : (N_modes,) float or None  (D_rb only)
    d_um   : float  pixel size for A_shift in µm (0.0 if not loaded)
    Z0_sim : complex  baseline impedance at freq_target (from first loaded lib)
    """
    libs   = {}
    Z0_sim = None
    d_um   = 0.0

    for lname, fname in LIBRARY_FILES.items():
        fpath = os.path.join(RESULTS, fname)
        if not os.path.exists(fpath):
            print(f'  {lname:20s}: SKIPPED (file not found: {fpath})')
            continue

        d     = np.load(fpath, allow_pickle=True)
        freqs = d['freqs']
        i_f   = np.argmin(np.abs(freqs - freq_target))
        Z_bl  = d['Z_baseline'].astype(complex)

        if Z0_sim is None:
            Z0_sim = Z_bl[i_f]

        # ── Library A: flat format ────────────────────────────────────────────
        if lname == 'A_shift' and 'dx_px' in d and d['Z'].ndim == 2:
            Z_lib  = d['Z'].astype(complex)          # (N_modes, n_freq)
            dx_arr = d['dx_px'].astype(int)
            dy_arr = d['dy_px'].astype(int)
            if 'd' in d:
                d_um = float(d['d']) * 1e4           # cm -> µm
            dZ_cx  = Z_lib[:, i_f] - Z_bl[i_f]     # (N_modes,) complex
            dZ_2d  = np.column_stack([dZ_cx.real, dZ_cx.imag])  # (N_modes, 2)
            labels = [f'A_shift[{k}]({dx_arr[k]},{dy_arr[k]})' for k in range(len(dZ_cx))]
            libs[lname] = dict(
                dZ    = dZ_2d,
                labels= labels,
                dx_px = dx_arr,
                dy_px = dy_arr,
                k_rb  = None,
                lw    = None,
            )
            dz_mag = np.sqrt(dZ_2d[:, 0]**2 + dZ_2d[:, 1]**2)
            print(
                f'  {lname:20s}: {len(dZ_cx):4d} modes  '
                f'd={d_um:.3f} µm/px  '
                f'|dZ| in [{dz_mag.min():.1f}, {dz_mag.max():.1f}] Ohm'
            )
            continue

        # ── Library B_rb: flat format with k_rb_arr / lw_arr ─────────────────
        if lname == 'B_rb' and 'k_rb_arr' in d:
            Z_lib     = d['Z'].astype(complex)       # (N_modes, n_freq)
            k_rb_arr_ = d['k_rb_arr']
            lw_arr_   = d['lw_arr'].astype(int)
            dZ_cx     = Z_lib[:, i_f] - Z_bl[i_f]
            dZ_2d     = np.column_stack([dZ_cx.real, dZ_cx.imag])
            labels    = [f'B_rb[K={k_rb_arr_[k]:.2g}_lw={lw_arr_[k]}]'
                         for k in range(len(dZ_cx))]
            libs[lname] = dict(
                dZ    = dZ_2d,
                labels= labels,
                dx_px = None,
                dy_px = None,
                k_rb  = k_rb_arr_,
                lw    = lw_arr_,
            )
            dz_mag = np.sqrt(dZ_2d[:, 0]**2 + dZ_2d[:, 1]**2)
            print(
                f'  {lname:20s}: {len(dZ_cx):4d} modes  '
                f'|dZ| in [{dz_mag.min():.1f}, {dz_mag.max():.1f}] Ohm'
            )
            continue

        # ── Library D_rb: 3-parameter (alpha, Rb, Cm) perturbation format ─────
        if lname == 'D_rb' and 'delta_alpha' in d:
            Z_lib   = d['Z'].astype(complex)       # (N_modes, n_freq)
            da_arr  = d['delta_alpha']             # (N_modes,)
            dRb_arr = d['delta_Rb']
            dCm_arr = d['delta_Cm']
            dZ_cx   = Z_lib[:, i_f] - Z_bl[i_f]
            dZ_2d   = np.column_stack([dZ_cx.real, dZ_cx.imag])
            labels  = [
                f'D_rb[da={da_arr[k]:+.3f}_dRb={dRb_arr[k]:+.3f}_dCm={dCm_arr[k]:+.3f}]'
                for k in range(len(dZ_cx))
            ]
            libs[lname] = dict(
                dZ      = dZ_2d,
                labels  = labels,
                dx_px   = None,
                dy_px   = None,
                k_rb    = None,
                lw      = None,
                rb_scale= None,
                d_alpha = da_arr,
                d_Rb    = dRb_arr,
                d_Cm    = dCm_arr,
            )
            dz_mag = np.sqrt(dZ_2d[:, 0]**2 + dZ_2d[:, 1]**2)
            print(
                f'  {lname:20s}: {len(dZ_cx):4d} modes  '
                f'|dZ| in [{dz_mag.min():.1f}, {dz_mag.max():.1f}] Ohm'
            )
            continue

        # ── All other libraries (old / C_brickwall format) ────────────────────
        Z_lib   = d['Z'].astype(complex)
        Z_flat  = Z_lib.reshape(-1, len(freqs))
        dZ_cx   = Z_flat[:, i_f] - Z_bl[i_f]
        dZ_2d   = np.column_stack([dZ_cx.real, dZ_cx.imag])
        labels  = [f'{lname}[{k}]' for k in range(len(dZ_cx))]
        libs[lname] = dict(
            dZ    = dZ_2d,
            labels= labels,
            dx_px = None,
            dy_px = None,
            k_rb  = None,
            lw    = None,
        )
        dz_mag = np.sqrt(dZ_2d[:, 0]**2 + dZ_2d[:, 1]**2)
        print(
            f'  {lname:20s}: {len(dZ_cx):4d} modes  '
            f'|dZ| in [{dz_mag.min():.1f}, {dz_mag.max():.1f}] Ohm'
        )

    if Z0_sim is None:
        Z0_sim = complex(0.0)

    # Two-electrode correction: Z_measured = 2 * Z_per_electrode.
    # Scale every library dZ to the two-electrode (measured) amplitude and
    # double Z0_sim so that cal = Z0_exp / Z0_sim is on the same scale.
    for lname in libs:
        libs[lname]['dZ'] = libs[lname]['dZ'] * 2.0
    Z0_sim = Z0_sim * 2.0

    print(f'  Total libraries loaded: {len(libs)}  |  |Z0_sim| (2-el) = {abs(Z0_sim):.2f} Ohm')
    return libs, d_um, Z0_sim


# ── Vectorized nearest-neighbour search ───────────────────────────────────────

def nearest_mode(dZ_lib, y, batch=4000):
    """
    Find the nearest library mode to each experimental point.

    Parameters
    ----------
    dZ_lib : (N_modes, 2)  [dR, dX] per mode
    y      : (N_t, 2)      experimental signal (or residual)

    Returns
    -------
    idx : (N_t,) int  index of nearest mode per time step
    """
    N_t = y.shape[0]
    idx = np.empty(N_t, dtype=np.intp)
    for s in range(0, N_t, batch):
        e = min(s + batch, N_t)
        # (batch, N_modes)
        d2 = np.sum((y[s:e, None, :] - dZ_lib[None, :, :]) ** 2, axis=2)
        idx[s:e] = np.argmin(d2, axis=1)
    return idx


# ── Greedy reconstruction ─────────────────────────────────────────────────────

def greedy_reconstruct(libs, dR, dX, order):
    """
    Apply greedy sequential state matching (same as v3).

    Parameters
    ----------
    libs  : dict from load_libraries()
    dR, dX: (N_t,) experimental fluctuation signals
    order : list of library names

    Returns
    -------
    result : dict with selected indices, reconstructed signal, residuals, etc.
    """
    N_t = len(dR)
    y   = np.column_stack([dR, dX])   # (N_t, 2)

    residual = y.copy()
    idx_dict = {}
    dZ_sel   = {}

    for step, lname in enumerate(order, start=1):
        if lname not in libs:
            print(f'  WARNING: library "{lname}" not loaded — skipping step {step}.')
            idx_dict[lname] = np.zeros(N_t, dtype=np.intp)
            dZ_sel[lname]   = np.zeros((N_t, 2), dtype=float)
            continue

        dZ_lib = libs[lname]['dZ']   # (N_modes, 2)
        print(f'Greedy {step}/{len(order)}  {lname:<12s} ... ', end='', flush=True)
        t0  = time.perf_counter()
        idx = nearest_mode(dZ_lib, residual)
        elapsed = time.perf_counter() - t0
        print(f'done  {elapsed:.1f} s')

        idx_dict[lname] = idx
        contrib         = dZ_lib[idx]           # (N_t, 2)
        dZ_sel[lname]   = contrib
        residual        = residual - contrib

    # Assemble results
    dR_rec   = sum(dZ_sel[n][:, 0] for n in order if n in dZ_sel)
    dX_rec   = sum(dZ_sel[n][:, 1] for n in order if n in dZ_sel)
    dR_resid = residual[:, 0]
    dX_resid = residual[:, 1]

    rmse_R     = float(np.sqrt(np.mean((dR_rec - dR)**2)))
    rmse_X     = float(np.sqrt(np.mean((dX_rec - dX)**2)))
    rmse_total = float(np.sqrt(np.mean((dR_rec - dR)**2 + (dX_rec - dX)**2)))

    return dict(
        idx_dict  = idx_dict,
        dZ_sel    = dZ_sel,
        dR_rec    = dR_rec,
        dX_rec    = dX_rec,
        dR_resid  = dR_resid,
        dX_resid  = dX_resid,
        rmse_R    = rmse_R,
        rmse_X    = rmse_X,
        rmse_total= rmse_total,
    )


# ── Null-mode injection ───────────────────────────────────────────────────────

def add_null_modes(libs):
    """
    Prepend a null mode (dZ=[0,0], all physical params=0) to each library.
    Modifies libs in-place.

    For A_shift: the first mode in the lower-triangle enumeration is (0,0),
    so mode 0 is already the null mode.  Confirmed by checking
    dx_px[0]==0 and dy_px[0]==0.  If confirmed, skip prepending.

    For all other libraries: always prepend.
    """
    for lname, lib in libs.items():
        dZ = lib['dZ']   # (M, 2) float64

        # ── A_shift: check if mode 0 is already (dx=0, dy=0) ─────────────────
        if lname == 'A_shift':
            dx_px = lib.get('dx_px')
            dy_px = lib.get('dy_px')
            if (dx_px is not None and dy_px is not None and
                    dx_px[0] == 0 and dy_px[0] == 0):
                print(f'  add_null_modes: {lname} — mode 0 already null (dx=0, dy=0), skipped')
                continue
            # Otherwise prepend null
            lib['dZ']     = np.vstack([np.zeros((1, 2), dtype=dZ.dtype), dZ])
            lib['labels'] = ['A_shift[null](0,0)'] + list(lib['labels'])
            if dx_px is not None:
                lib['dx_px'] = np.insert(dx_px, 0, 0)
            if dy_px is not None:
                lib['dy_px'] = np.insert(dy_px, 0, 0)
            print(f'  add_null_modes: {lname} — null mode prepended')
            continue

        # ── B_rb ──────────────────────────────────────────────────────────────
        if lname == 'B_rb':
            lib['dZ']     = np.vstack([np.zeros((1, 2), dtype=dZ.dtype), dZ])
            lib['labels'] = ['B_rb[null](k_rb=0,lw=0)'] + list(lib['labels'])
            if lib.get('k_rb') is not None:
                lib['k_rb'] = np.insert(lib['k_rb'].astype(float), 0, 0.0)
            if lib.get('lw') is not None:
                lib['lw'] = np.insert(lib['lw'].astype(int), 0, 0)
            print(f'  add_null_modes: {lname} — null mode prepended (k_rb=0, lw=0)')
            continue

        # ── D_rb ──────────────────────────────────────────────────────────────
        if lname == 'D_rb':
            lib['dZ']     = np.vstack([np.zeros((1, 2), dtype=dZ.dtype), dZ])
            lib['labels'] = ['D_rb[null](da=0,dRb=0,dCm=0)'] + list(lib['labels'])
            if lib.get('d_alpha') is not None:
                lib['d_alpha'] = np.insert(lib['d_alpha'].astype(float), 0, 0.0)
            if lib.get('d_Rb') is not None:
                lib['d_Rb'] = np.insert(lib['d_Rb'].astype(float), 0, 0.0)
            if lib.get('d_Cm') is not None:
                lib['d_Cm'] = np.insert(lib['d_Cm'].astype(float), 0, 0.0)
            print(f'  add_null_modes: {lname} — null mode prepended (da=0, dRb=0, dCm=0)')
            continue

        # ── C_brickwall (and any unrecognised library) ─────────────────────────
        lib['dZ']     = np.vstack([np.zeros((1, 2), dtype=dZ.dtype), dZ])
        lib['labels'] = [f'{lname}[null]'] + list(lib['labels'])
        print(f'  add_null_modes: {lname} — null mode prepended')


# ── Distance matrix builders ──────────────────────────────────────────────────

def build_dist_matrix_A(dx_px, dy_px):
    """
    Euclidean distance in pixel space for A_shift modes.
    Returns (M, M) float32.
    """
    dx = dx_px.astype(np.float32)
    dy = dy_px.astype(np.float32)
    diff_x = dx[:, None] - dx[None, :]   # (M, M)
    diff_y = dy[:, None] - dy[None, :]
    return np.sqrt(diff_x**2 + diff_y**2).astype(np.float32)


def build_dist_matrix_B(k_rb_arr, lw_arr):
    """
    Euclidean distance in normalized (k_rb / max_|k_rb|, lw / max_lw) space.
    Returns (M, M) float32.
    """
    k = k_rb_arr.astype(np.float32)
    l = lw_arr.astype(np.float32)

    max_k = float(np.max(np.abs(k)))
    max_l = float(np.max(np.abs(l)))
    if max_k == 0.0:
        max_k = 1.0
    if max_l == 0.0:
        max_l = 1.0

    k_norm = k / max_k
    l_norm = l / max_l

    diff_k = k_norm[:, None] - k_norm[None, :]   # (M, M)
    diff_l = l_norm[:, None] - l_norm[None, :]
    return np.sqrt(diff_k**2 + diff_l**2).astype(np.float32)


def build_dist_matrix_D(d_alpha, d_Rb, d_Cm):
    """
    Chebyshev distance in GRID INDEX space for D_rb modes.

    The D_rb library uses a non-uniform grid (e.g. fine steps ±0.001..±0.006
    plus outer points ±0.010).  In parameter-value space the gap between
    ±0.006 and ±0.010 is 0.004 — much larger than the fine step 0.001 — so
    a fixed value-based max_jump would isolate the outer modes.

    Using grid-index space every adjacent grid cell is exactly 1 step away
    regardless of the actual value spacing, so max_jump_D=1 gives 26
    Chebyshev neighbours in 3D and the outer ±1 % modes are always reachable
    in one transition step from the ±0.6 % boundary.

    Each parameter axis is mapped independently: unique sorted values become
    indices 0, 1, 2, ...  The null mode (prepended with d=0) is mapped to the
    closest index in the sorted grid, or inserted as a separate index=0 entry.

    Returns (M, M) float32  — Chebyshev distance in index units.
    """
    def _to_idx(vals):
        uniq = np.unique(vals)          # sorted unique values for this axis
        idx  = np.searchsorted(uniq, vals)  # index of each mode in that sorted list
        return idx.astype(np.float32)

    ia = _to_idx(d_alpha)
    ib = _to_idx(d_Rb)
    ic = _to_idx(d_Cm)

    n_uniq_a = len(np.unique(d_alpha))
    n_uniq_b = len(np.unique(d_Rb))
    n_uniq_c = len(np.unique(d_Cm))
    print(f'  D_rb index grid: alpha={n_uniq_a} vals, Rb={n_uniq_b} vals, Cm={n_uniq_c} vals')

    diff_a = np.abs(ia[:, None] - ia[None, :])   # (M, M)
    diff_b = np.abs(ib[:, None] - ib[None, :])
    diff_c = np.abs(ic[:, None] - ic[None, :])
    # Chebyshev = max of absolute differences along each axis
    return np.maximum(np.maximum(diff_a, diff_b), diff_c).astype(np.float32)


def build_dist_matrix_C(n_modes):
    """
    |i - j| distance for ordered C_brickwall modes.
    Returns (M, M) float32.
    """
    idx = np.arange(n_modes, dtype=np.float32)
    return np.abs(idx[:, None] - idx[None, :]).astype(np.float32)


# ── Neighbourhood structure ───────────────────────────────────────────────────

def build_nbrs_padded(dist_matrix, max_jump, label='lib'):
    """
    Build padded neighbour arrays for the vectorised Viterbi inner loop.

    For each mode i: neighbours j where dist_matrix[i, j] <= max_jump.
    Self-transition (j == i, dist = 0) is always included because 0 <= max_jump.

    Padding: rows shorter than max_k are padded with index 0 (safe — the
    padded entries are masked by trans_padded = inf, so they never win argmin).

    Returns
    -------
    nbrs_padded  : (M, max_k) int32    — neighbour indices, padded with 0
    trans_padded : (M, max_k) float32  — 0.0 for real neighbours, inf for padding
    """
    M    = dist_matrix.shape[0]
    mask = dist_matrix <= max_jump   # (M, M) bool  (self always True since dist=0)

    counts = mask.sum(axis=1)        # (M,) — each row has at least 1 (self)
    max_k  = int(counts.max())

    nbrs_padded  = np.zeros((M, max_k), dtype=np.int32)
    trans_padded = np.full((M, max_k), np.inf, dtype=np.float32)

    for i in range(M):
        nbr_i = np.where(mask[i])[0].astype(np.int32)
        k_i   = len(nbr_i)
        nbrs_padded[i, :k_i]  = nbr_i
        trans_padded[i, :k_i] = 0.0   # zero transition cost within max_jump

    mean_nbrs = float(counts.mean())
    print(f'  {label}: M={M}, max_k={max_k} (mean {mean_nbrs:.1f} neighbors per mode)')
    return nbrs_padded, trans_padded


# ── Viterbi core ──────────────────────────────────────────────────────────────

def _run_viterbi_forward(dZ_lib, residuals, nbrs_padded, trans_padded, V_init=None):
    """
    Forward-only Viterbi pass.  Does NOT store back-pointers.

    Used internally by viterbi_solve for checkpoint generation in chunked mode.

    Parameters
    ----------
    dZ_lib       : (M, 2) float32
    residuals    : (T, 2) float32
    nbrs_padded  : (M, max_k) int32
    trans_padded : (M, max_k) float32
    V_init       : (M,) float32 or None
                   If None, initialise from obs_cost[0] (no prior step).
                   If provided, use as V_prev and start forward from step 0.

    Returns
    -------
    V : (M,) float32  — final value function
    """
    T = residuals.shape[0]
    M = dZ_lib.shape[0]
    arange_M = np.arange(M, dtype=np.int32)

    # Pre-compute all observation costs: (T, M) float32
    diff_R   = residuals[:, 0:1] - dZ_lib[:, 0]   # (T, M)
    diff_X   = residuals[:, 1:2] - dZ_lib[:, 1]
    obs_cost = (diff_R**2 + diff_X**2).astype(np.float32)

    if V_init is None:
        V_prev  = obs_cost[0].copy()
        t_start = 1
    else:
        V_prev  = V_init.astype(np.float32)
        t_start = 0

    for t in range(t_start, T):
        obs_t  = obs_cost[t]                        # (M,)
        V_nbrs = V_prev[nbrs_padded]               # (M, max_k)
        cand   = V_nbrs + trans_padded              # (M, max_k)
        best_k = np.argmin(cand, axis=1)           # (M,)
        V_prev = obs_t + cand[arange_M, best_k]   # (M,)

    return V_prev


def viterbi_solve(dZ_lib, residuals, nbrs_padded, trans_padded, chunk_size=5000,
                  V_init=None, signal_batch_size=None):
    """
    Full Viterbi: forward pass + traceback.  Returns optimal state sequence.

    Automatically selects in-memory or chunked mode based on back-pointer
    array size (threshold: 500 MB = 500_000_000 bytes; back array is int32,
    4 bytes per element, shape (T, M)).

    Parameters
    ----------
    dZ_lib            : (M, 2) float32
    residuals         : (T, 2) float32
    nbrs_padded       : (M, max_k) int32
    trans_padded      : (M, max_k) float32
    chunk_size        : int   — steps per chunk in chunked mode
    V_init            : (M,) float32 or None
                        If provided, used as cost entering t=0 (batch chaining).
                        V_init[j] = 0 for the seed mode, inf for all others.
    signal_batch_size : int or None
                        If set and T > signal_batch_size, split residuals into
                        batches of this length.  Each batch is solved independently
                        but the terminal mode of batch k seeds batch k+1 via V_init,
                        enforcing continuity at batch boundaries.  obs_cost is
                        allocated per-batch, keeping peak memory to
                        signal_batch_size × M × 4 bytes.

    Returns
    -------
    seq : (T,) int32  — optimal state sequence (indices into dZ_lib)
    """
    T, M = residuals.shape[0], dZ_lib.shape[0]
    arange_M = np.arange(M, dtype=np.int32)

    # ── Signal batching (large-T memory reduction) ────────────────────────────
    if signal_batch_size is not None and T > signal_batch_size:
        n_batches = int(np.ceil(T / signal_batch_size))
        seq = np.empty(T, dtype=np.int32)
        V_batch = None   # None for first batch → normal initialisation
        for b in range(n_batches):
            bs = b * signal_batch_size
            be = min(bs + signal_batch_size, T)
            print(f'  Signal batch {b + 1}/{n_batches}  '
                  f'steps {bs}–{be - 1}  ({be - bs} pts)', flush=True)
            seq_b = viterbi_solve(
                dZ_lib, residuals[bs:be], nbrs_padded, trans_padded,
                chunk_size=chunk_size, V_init=V_batch, signal_batch_size=None,
            )
            seq[bs:be] = seq_b
            # Seed next batch: only the terminal mode has zero cost
            terminal = int(seq_b[-1])
            V_batch = np.full(M, np.inf, dtype=np.float32)
            V_batch[terminal] = 0.0
        return seq

    # Pre-compute all observation costs: (T, M) float32
    diff_R   = residuals[:, 0:1] - dZ_lib[:, 0]   # (T, M) broadcast
    diff_X   = residuals[:, 1:2] - dZ_lib[:, 1]
    obs_cost = (diff_R**2 + diff_X**2).astype(np.float32)

    # back array: (T, M) int32  — 4 bytes per element
    back_bytes = T * M * 4
    in_memory  = back_bytes < 500_000_000

    # ── In-memory mode ────────────────────────────────────────────────────────
    if in_memory:
        back_mb = back_bytes / 1024**2
        print(f'  In-memory Viterbi (back={back_mb:.0f} MB)')

        back = np.empty((T, M), dtype=np.int32)

        if V_init is None:
            V_prev  = obs_cost[0].copy()
            back[0] = arange_M          # t=0: self-pointer (no prior step)
            t_start = 1
        else:
            # Transition from V_init into t=0 (batch boundary)
            V_nbrs  = V_init[nbrs_padded]                    # (M, max_k)
            cand    = V_nbrs + trans_padded
            best_k  = np.argmin(cand, axis=1)
            back[0] = nbrs_padded[arange_M, best_k]
            V_prev  = obs_cost[0] + cand[arange_M, best_k]
            t_start = 1

        for t in range(t_start, T):
            obs_t   = obs_cost[t]
            V_nbrs  = V_prev[nbrs_padded]
            cand    = V_nbrs + trans_padded
            best_k  = np.argmin(cand, axis=1)
            back[t] = nbrs_padded[arange_M, best_k]
            V_prev  = obs_t + cand[arange_M, best_k]

        # Traceback
        seq      = np.empty(T, dtype=np.int32)
        seq[T-1] = int(np.argmin(V_prev))
        for t in range(T - 2, -1, -1):
            seq[t] = back[t + 1, seq[t + 1]]

        return seq

    # ── Chunked mode ──────────────────────────────────────────────────────────
    n_chunks = int(np.ceil(T / chunk_size))
    print(f'  Chunked Viterbi (C={chunk_size}, {n_chunks} chunks)')

    # Determine chunk boundaries
    chunk_starts = [i * chunk_size       for i in range(n_chunks)]
    chunk_ends   = [min((i+1)*chunk_size, T) for i in range(n_chunks)]

    # Forward pass: save V at entry of each chunk (i.e. after processing
    # all steps up to but not including chunk_starts[i]).
    # checkpoints[i] = V entering step chunk_starts[i].
    checkpoints = []

    # V entering step 0.
    # If V_init provided (batch boundary): transition from it into t=0.
    # Otherwise: initialise from obs_cost[0] with no prior step.
    if V_init is None:
        V_prev = obs_cost[0].copy()
    else:
        V_nbrs_0 = V_init[nbrs_padded]
        cand_0   = V_nbrs_0 + trans_padded
        best_k_0 = np.argmin(cand_0, axis=1)
        V_prev   = obs_cost[0] + cand_0[arange_M, best_k_0]
    checkpoints.append(V_prev.copy())   # checkpoint[0] = V at start of chunk 0

    for chunk_idx in range(n_chunks):
        cs = chunk_starts[chunk_idx]
        ce = chunk_ends[chunk_idx]
        # Steps processed in this chunk
        t_start_loop = cs + 1 if chunk_idx == 0 else cs
        # For chunk 0: step 0 is already consumed into V_prev above.
        # For chunk k>0: step cs is the first new step.
        if chunk_idx == 0:
            t_start_loop = 1
        else:
            # Transition from checkpoint into step cs
            obs_t  = obs_cost[cs]
            V_nbrs = V_prev[nbrs_padded]
            cand   = V_nbrs + trans_padded
            best_k = np.argmin(cand, axis=1)
            V_prev = obs_t + cand[arange_M, best_k]
            t_start_loop = cs + 1

        for t in range(t_start_loop, ce):
            obs_t  = obs_cost[t]
            V_nbrs = V_prev[nbrs_padded]
            cand   = V_nbrs + trans_padded
            best_k = np.argmin(cand, axis=1)
            V_prev = obs_t + cand[arange_M, best_k]

        # Save checkpoint at entry of next chunk (= V after processing this chunk)
        if chunk_idx < n_chunks - 1:
            checkpoints.append(V_prev.copy())

    # V_prev now holds V at T-1.
    terminal_state = int(np.argmin(V_prev))

    # Backward traceback: iterate chunks in reverse, re-running forward for
    # each chunk to recover back-pointers, then trace within the chunk.
    seq = np.empty(T, dtype=np.int32)
    seq[T - 1] = terminal_state
    carry_state = terminal_state   # state at the END of current chunk

    for chunk_idx in range(n_chunks - 1, -1, -1):
        cs = chunk_starts[chunk_idx]
        ce = chunk_ends[chunk_idx]
        chunk_len = ce - cs

        # Re-run forward for this chunk, storing back-pointers locally.
        V_chunk_init = checkpoints[chunk_idx]   # V entering step cs

        back_chunk = np.empty((chunk_len, M), dtype=np.int32)

        if chunk_idx == 0:
            # Step 0: either free start (V_init=None) or batch-boundary transition
            if V_init is None:
                back_chunk[0] = arange_M   # self-pointer, no prior step
                V_prev_c      = obs_cost[0].copy()
            else:
                V_nbrs_0      = V_init[nbrs_padded]
                cand_0        = V_nbrs_0 + trans_padded
                best_k_0      = np.argmin(cand_0, axis=1)
                back_chunk[0] = nbrs_padded[arange_M, best_k_0]
                V_prev_c      = obs_cost[0] + cand_0[arange_M, best_k_0]
            t_local_start = 1
        else:
            # First step of this chunk (global index cs) transitions from V_chunk_init
            obs_t         = obs_cost[cs]
            V_nbrs        = V_chunk_init[nbrs_padded]
            cand          = V_nbrs + trans_padded
            best_k        = np.argmin(cand, axis=1)
            back_chunk[0] = nbrs_padded[arange_M, best_k]
            V_prev_c      = obs_t + cand[arange_M, best_k]
            t_local_start = 1

        for t_local in range(t_local_start, chunk_len):
            t_global      = cs + t_local
            obs_t         = obs_cost[t_global]
            V_nbrs        = V_prev_c[nbrs_padded]
            cand          = V_nbrs + trans_padded
            best_k        = np.argmin(cand, axis=1)
            back_chunk[t_local] = nbrs_padded[arange_M, best_k]
            V_prev_c      = obs_t + cand[arange_M, best_k]

        # Traceback within this chunk.
        # seq[ce-1] is already set (either T-1 via terminal_state or from carry).
        if ce - 1 < T - 1:
            seq[ce - 1] = carry_state

        for t_local in range(chunk_len - 2, -1, -1):
            t_global      = cs + t_local
            seq[t_global] = back_chunk[t_local + 1, seq[t_global + 1]]

        carry_state = seq[cs]

    return seq


# ── Viterbi reconstruction ────────────────────────────────────────────────────

def viterbi_reconstruct(libs, dR, dX, order, max_jumps, chunk_size=5000,
                        signal_batch_size=None):
    """
    Apply Viterbi temporal regularization per library.

    Same sequential residual-subtraction structure as greedy_reconstruct.
    Also computes diagnostic statistics per library.

    Parameters
    ----------
    libs      : dict from load_libraries() (with null modes already added)
    dR, dX    : (N_t,) experimental fluctuation signals
    order     : list of library names
    max_jumps : dict {lname: float}  — max_jump per library
    chunk_size: int  — chunk size for large-T Viterbi

    Returns
    -------
    result : dict (same keys as greedy_reconstruct) plus 'vit_stats'
    """
    N_t = len(dR)
    y   = np.column_stack([dR, dX]).astype(np.float32)   # (N_t, 2)

    residual = y.copy()
    idx_dict = {}
    dZ_sel   = {}
    vit_stats = {}

    for step, lname in enumerate(order, start=1):
        if lname not in libs:
            print(f'  WARNING: library "{lname}" not loaded — skipping step {step}.')
            idx_dict[lname] = np.zeros(N_t, dtype=np.int32)
            dZ_sel[lname]   = np.zeros((N_t, 2), dtype=np.float32)
            continue

        lib    = libs[lname]
        dZ_lib = lib['dZ'].astype(np.float32)    # (M, 2)
        M      = dZ_lib.shape[0]
        mj     = max_jumps.get(lname, 3.0)

        print(f'\nViterbi {step}/{len(order)}  {lname:<12s}  M={M}  max_jump={mj}')

        # Build distance matrix for this library
        if lname == 'A_shift':
            dist_mat = build_dist_matrix_A(lib['dx_px'], lib['dy_px'])
        elif lname == 'B_rb':
            dist_mat = build_dist_matrix_B(lib['k_rb'], lib['lw'])
        elif lname == 'D_rb':
            dist_mat = build_dist_matrix_D(lib['d_alpha'], lib['d_Rb'], lib['d_Cm'])
        else:
            # C_brickwall or any unrecognised library
            dist_mat = build_dist_matrix_C(M)

        nbrs_p, trans_p = build_nbrs_padded(dist_mat, mj, label=lname)

        t0  = time.perf_counter()
        seq = viterbi_solve(dZ_lib, residual, nbrs_p, trans_p, chunk_size=chunk_size,
                            signal_batch_size=signal_batch_size)
        elapsed = time.perf_counter() - t0
        print(f'  Viterbi done  {elapsed:.1f} s')

        idx_dict[lname] = seq.astype(np.intp)
        contrib         = dZ_lib[seq]             # (N_t, 2) float32
        dZ_sel[lname]   = contrib.astype(np.float64)

        # Diagnostic: jump distances in this library's metric space
        if M > 1 and N_t > 1:
            jumps           = dist_mat[seq[:-1], seq[1:]]   # (N_t-1,)
            mean_jump       = float(jumps.mean())
            max_jump_actual = float(jumps.max())
        else:
            mean_jump       = 0.0
            max_jump_actual = 0.0

        vit_stats[lname] = dict(
            mean_jump       = mean_jump,
            max_jump_actual = max_jump_actual,
            elapsed         = elapsed,
        )

        residual = (residual - contrib).astype(np.float32)

    # Assemble results
    dR_f = dR.astype(np.float64)
    dX_f = dX.astype(np.float64)

    dR_rec   = sum(dZ_sel[n][:, 0] for n in order if n in dZ_sel)
    dX_rec   = sum(dZ_sel[n][:, 1] for n in order if n in dZ_sel)
    dR_resid = dR_f - dR_rec
    dX_resid = dX_f - dX_rec

    rmse_R     = float(np.sqrt(np.mean((dR_rec - dR_f)**2)))
    rmse_X     = float(np.sqrt(np.mean((dX_rec - dX_f)**2)))
    rmse_total = float(np.sqrt(np.mean((dR_rec - dR_f)**2 + (dX_rec - dX_f)**2)))

    return dict(
        idx_dict   = idx_dict,
        dZ_sel     = dZ_sel,
        dR_rec     = dR_rec,
        dX_rec     = dX_rec,
        dR_resid   = dR_resid,
        dX_resid   = dX_resid,
        rmse_R     = rmse_R,
        rmse_X     = rmse_X,
        rmse_total = rmse_total,
        vit_stats  = vit_stats,
    )


# ── Physical parameter extraction ─────────────────────────────────────────────

def extract_physical(libs, idx_dict, d_um):
    """
    Extract physical parameters from selected mode indices.

    Returns
    -------
    dx_um    : (N_t,) µm displacement x  (A_shift)
    dy_um    : (N_t,) µm displacement y  (A_shift)
    k_rb_seq : (N_t,) float  K_RB per step  (B_rb)
    lw_seq   : (N_t,) int    lw per step    (B_rb)
    da_seq   : (N_t,) float  delta_alpha    (D_rb)
    dRb_seq  : (N_t,) float  delta_Rb       (D_rb)
    dCm_seq  : (N_t,) float  delta_Cm       (D_rb)
    c_idx_seq: (N_t,) int    mode index     (C_brickwall)
    """
    N_t = len(next(iter(idx_dict.values())))

    # A_shift displacement
    dx_um = np.zeros(N_t, dtype=float)
    dy_um = np.zeros(N_t, dtype=float)
    if 'A_shift' in libs and 'A_shift' in idx_dict:
        lib_A  = libs['A_shift']
        idx_A  = idx_dict['A_shift']
        if lib_A['dx_px'] is not None and d_um > 0.0:
            dx_um = lib_A['dx_px'][idx_A].astype(float) * d_um
            dy_um = lib_A['dy_px'][idx_A].astype(float) * d_um

    # B_rb K_RB and lw sequences
    k_rb_seq = np.full(N_t, np.nan)
    lw_seq   = np.full(N_t, -1, dtype=int)
    if 'B_rb' in libs and 'B_rb' in idx_dict:
        lib_B = libs['B_rb']
        idx_B = idx_dict['B_rb']
        if lib_B['k_rb'] is not None:
            k_rb_seq = lib_B['k_rb'][idx_B].astype(float)
        if lib_B['lw'] is not None:
            lw_seq = lib_B['lw'][idx_B].astype(int)
        else:
            # Fallback: parse from labels using regex
            k_rb_vals_ = []
            lw_vals_   = []
            for i in idx_B:
                lbl = lib_B['labels'][i]
                mk  = re.search(r'K=([+-]?\d+\.?\d*(?:e[+-]?\d+)?)', lbl)
                ml  = re.search(r'lw=(\d+)', lbl)
                k_rb_vals_.append(float(mk.group(1)) if mk else np.nan)
                lw_vals_.append(int(ml.group(1)) if ml else -1)
            k_rb_seq = np.array(k_rb_vals_)
            lw_seq   = np.array(lw_vals_, dtype=int)

    # C_brickwall mode index sequence
    c_idx_seq = np.zeros(N_t, dtype=int)
    if 'C_brickwall' in idx_dict:
        c_idx_seq = idx_dict['C_brickwall'].astype(int)

    # D_rb parameter sequences (delta_alpha, delta_Rb, delta_Cm)
    da_seq  = np.full(N_t, np.nan)
    dRb_seq = np.full(N_t, np.nan)
    dCm_seq = np.full(N_t, np.nan)
    if 'D_rb' in libs and 'D_rb' in idx_dict:
        lib_D = libs['D_rb']
        idx_D = idx_dict['D_rb']
        if lib_D.get('d_alpha') is not None:
            da_seq  = lib_D['d_alpha'][idx_D].astype(float)
            dRb_seq = lib_D['d_Rb'][idx_D].astype(float)
            dCm_seq = lib_D['d_Cm'][idx_D].astype(float)

    return dx_um, dy_um, k_rb_seq, lw_seq, da_seq, dRb_seq, dCm_seq, c_idx_seq


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary(stem, order, libs, Z0_exp, Z0_sim, cal_active, N_t, dt, res):
    arrow_order = ' -> '.join(order)
    cal_str = 'ON' if cal_active else 'OFF'
    print(f'\n=== Viterbi reconstruction v4: {stem} ===')
    print(f'  Order: {arrow_order}')
    print(f'  Z0_exp={Z0_exp:.1f} Ohm   Z0_sim={Z0_sim:.1f} Ohm   cal={cal_str}')
    print(f'  {"Library":<15}  {"N_modes":>7}  {"dZ_range [Ohm]"}')
    for lname in order:
        if lname not in libs:
            print(f'  {lname:<15}  {"N/A":>7}  (not loaded)')
            continue
        dZ   = libs[lname]['dZ']
        mags = np.sqrt(dZ[:, 0]**2 + dZ[:, 1]**2)
        print(f'  {lname:<15}  {len(dZ):>7d}  [{mags.min():.1f}, {mags.max():.1f}]')
    print(f'  N_steps = {N_t}   dt = {dt:.3f} s')
    print()


# ── Figures ───────────────────────────────────────────────────────────────────

def plot_fit(t, dR, dX, dR_rec, dX_rec, dR_resid, dX_resid, stem, res, savefig, show):
    plt.rcParams.update({'font.size': 9})
    t_min = t / 60.0
    rmse_total = res['rmse_total']

    fig, axes = plt.subplots(3, 2, figsize=(14, 9), constrained_layout=True)
    fig.suptitle(
        f'Viterbi reconstruction v4 — {stem}  RMSE={rmse_total:.1f} Ohm',
        fontsize=11)

    # Row 1: dR
    ax = axes[0, 0]
    ax.plot(t_min, dR, color='k', lw=0.7, alpha=0.9, label='dR_exp')
    ax.plot(t_min, dR_rec, color='tomato', lw=0.6, alpha=0.8, label='dR_rec')
    ax.set_ylabel('dR [Ohm]')
    ax.legend(fontsize=7, loc='upper right')
    ax.set_title('Resistance fluctuation', fontsize=8)

    ax = axes[0, 1]
    ax.plot(t_min, dR - dR_rec, color='tomato', lw=0.5, alpha=0.7)
    ax.axhline(0, color='k', lw=0.4, ls=':')
    ax.set_ylabel('dR residual [Ohm]')
    ax.set_title(f'dR residual  RMSE={res["rmse_R"]:.2f} Ohm', fontsize=8)

    # Row 2: dX
    ax = axes[1, 0]
    ax.plot(t_min, dX, color='k', lw=0.7, alpha=0.9, label='dX_exp')
    ax.plot(t_min, dX_rec, color='steelblue', lw=0.6, alpha=0.8, label='dX_rec')
    ax.set_ylabel('dX [Ohm]')
    ax.legend(fontsize=7, loc='upper right')
    ax.set_title('Reactance fluctuation', fontsize=8)

    ax = axes[1, 1]
    ax.plot(t_min, dX - dX_rec, color='steelblue', lw=0.5, alpha=0.7)
    ax.axhline(0, color='k', lw=0.4, ls=':')
    ax.set_ylabel('dX residual [Ohm]')
    ax.set_title(f'dX residual  RMSE={res["rmse_X"]:.2f} Ohm', fontsize=8)

    # Row 3: |residual|
    resid_mag = np.sqrt(dR_resid**2 + dX_resid**2)
    ax = axes[2, 0]
    ax.fill_between(t_min, 0, resid_mag, color='darkorange', alpha=0.5, linewidth=0)
    ax.plot(t_min, resid_mag, color='darkorange', lw=0.5)
    ax.set_ylabel('|residual| [Ohm]')
    ax.set_xlabel('Time [min]')
    ax.set_title('Total residual magnitude', fontsize=8)

    ax = axes[2, 1]
    ax.hist(resid_mag, bins=80, color='darkorange', alpha=0.7, edgecolor='none')
    ax.set_xlabel('|residual| [Ohm]')
    ax.set_ylabel('Count')
    ax.set_title('Residual magnitude distribution', fontsize=8)

    if savefig:
        fpath = os.path.join(RESULTS, f'reconstruct_v4_{stem}_fit.png')
        fig.savefig(fpath, dpi=150, bbox_inches='tight')
        print(f'Saved: {fpath}')
    if show:
        plt.show()
    plt.close(fig)


def plot_phasespace(t, dR, dX, dR_rec, dX_rec, libs, order, stem, savefig, show):
    plt.rcParams.update({'font.size': 9})
    t_norm = (t - t[0]) / max(t[-1] - t[0], 1.0)

    sub = slice(None, None, max(1, len(t) // 4000))

    fig, ax = plt.subplots(figsize=(7, 7))

    # Experimental scatter colored by normalised time
    sc = ax.scatter(dR[sub], dX[sub], c=t_norm[sub], cmap='viridis',
                    s=1, zorder=1, rasterized=True, label='Experimental')
    plt.colorbar(sc, ax=ax, label='Normalised time (0=start, 1=end)', shrink=0.8)

    # Reconstructed scatter
    ax.scatter(dR_rec[sub], dX_rec[sub], color='grey', s=1, alpha=0.4,
               zorder=2, rasterized=True, label='Reconstructed')

    # Library mode scatter
    lib_plot_params = {
        'B_rb':        dict(color='red',    s=8,  label='B_rb modes'),
        'A_shift':     dict(color='blue',   s=4,  label='A_shift modes'),
        'C_brickwall': dict(color='purple', s=6,  label='C_brickwall modes'),
    }
    for lname in order:
        if lname not in libs:
            continue
        dZ = libs[lname]['dZ']
        kw = lib_plot_params.get(lname, dict(color='grey', s=4, label=lname))
        ax.scatter(dZ[:, 0], dZ[:, 1], zorder=3, rasterized=True, **kw)

    ax.axhline(0, color='k', lw=0.4, ls=':')
    ax.axvline(0, color='k', lw=0.4, ls=':')
    ax.set_xlabel('dR [Ohm]')
    ax.set_ylabel('dX [Ohm]')
    ax.set_title(f'Phase space — {stem}', fontsize=10)
    ax.legend(fontsize=7, loc='lower left', markerscale=4)
    fig.tight_layout()

    if savefig:
        fpath = os.path.join(RESULTS, f'reconstruct_v4_{stem}_phasespace.png')
        fig.savefig(fpath, dpi=150, bbox_inches='tight')
        print(f'Saved: {fpath}')
    if show:
        plt.show()
    plt.close(fig)


def plot_states(t, dx_um, dy_um, k_rb_seq, lw_seq,
                da_seq, dRb_seq, dCm_seq,
                c_idx_seq, stem, savefig, show):
    plt.rcParams.update({'font.size': 9})
    t_min = t / 60.0

    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True,
                             constrained_layout=True)
    fig.suptitle(f'State sequences — {stem}', fontsize=11)

    # Row 1: A_shift displacement magnitude + dx/dy on secondary axis
    ax1 = axes[0]
    disp_mag = np.sqrt(dx_um**2 + dy_um**2)
    ax1.fill_between(t_min, 0, disp_mag, color='steelblue', alpha=0.4, linewidth=0)
    ax1.plot(t_min, disp_mag, color='steelblue', lw=0.6, label='|disp|')
    ax1.set_ylabel('|displacement| [µm]', color='steelblue')
    ax1.tick_params(axis='y', colors='steelblue')
    ax1.set_title('A_shift: displacement magnitude', fontsize=8)

    ax1b = ax1.twinx()
    ax1b.plot(t_min, dx_um, color='grey', lw=0.5, alpha=0.7, label='dx_um')
    ax1b.plot(t_min, dy_um, color='olive', lw=0.5, alpha=0.7, label='dy_um')
    ax1b.set_ylabel('dx / dy [µm]', fontsize=7)
    ax1b.tick_params(axis='y', labelsize=6)
    lines1, labs1   = ax1.get_legend_handles_labels()
    lines1b, labs1b = ax1b.get_legend_handles_labels()
    ax1.legend(lines1 + lines1b, labs1 + labs1b, fontsize=7, loc='upper right')

    # Row 2: B_rb K_RB (left) and lw (right)
    ax2 = axes[1]
    valid_krb = ~np.isnan(k_rb_seq)
    if valid_krb.any():
        ax2.step(t_min, np.where(valid_krb, k_rb_seq, np.nan),
                 color='red', lw=0.7, alpha=0.9, where='post', label='K_RB')
    ax2.set_ylabel('K_RB', color='red')
    ax2.tick_params(axis='y', colors='red')
    ax2.set_title('B_rb: K_RB and lw state sequence', fontsize=8)

    ax2b = ax2.twinx()
    valid_lw = lw_seq >= 0
    if valid_lw.any():
        ax2b.step(t_min, np.where(valid_lw, lw_seq, np.nan),
                  color='navy', lw=0.7, alpha=0.8, where='post', label='lw')
    ax2b.set_ylabel('lw', color='navy', fontsize=7)
    ax2b.tick_params(axis='y', labelsize=6, colors='navy')
    lines2, labs2   = ax2.get_legend_handles_labels()
    lines2b, labs2b = ax2b.get_legend_handles_labels()
    ax2.legend(lines2 + lines2b, labs2 + labs2b, fontsize=7, loc='upper right')

    # Row 3: D_rb — delta_alpha, delta_Rb, delta_Cm evolution
    ax3 = axes[2]

    valid_da = ~np.isnan(da_seq)
    if valid_da.any():
        ax3.step(t_min, np.where(valid_da, da_seq * 100, np.nan),
                 color='#2ca02c', lw=0.7, alpha=0.9, where='post', label=r'$\delta\alpha$ [%]')
        ax3.step(t_min, np.where(valid_da, dRb_seq * 100, np.nan),
                 color='#d62728', lw=0.7, alpha=0.9, where='post', label=r'$\delta R_b$ [%]')
        ax3.step(t_min, np.where(valid_da, dCm_seq * 100, np.nan),
                 color='#1f77b4', lw=0.7, alpha=0.9, where='post', label=r'$\delta C_m$ [%]')

    ax3.axhline(0, color='#aaaaaa', lw=0.6, ls=':')
    ax3.set_ylabel(r'$\delta$ param [%]', color='teal')
    ax3.tick_params(axis='y', labelsize=7)
    ax3.set_title(r'D_rb: $\delta\alpha$, $\delta R_b$, $\delta C_m$ state sequence', fontsize=8)
    ax3.legend(fontsize=7, loc='upper right', ncol=3)
    ax3.grid(True, alpha=0.15)

    # Row 4: C_brickwall mode index
    ax4 = axes[3]
    ax4.step(t_min, c_idx_seq, color='purple', lw=0.7, where='post')
    ax4.fill_between(t_min, 0, c_idx_seq, step='post',
                     color='purple', alpha=0.3, linewidth=0)
    ax4.set_ylabel('C_brickwall mode index', color='purple')
    ax4.tick_params(axis='y', colors='purple')
    ax4.set_title('C_brickwall: mode index sequence', fontsize=8)
    ax4.set_xlabel('Time [min]')

    if savefig:
        fpath = os.path.join(RESULTS, f'reconstruct_v4_{stem}_states.png')
        fig.savefig(fpath, dpi=150, bbox_inches='tight')
        print(f'Saved: {fpath}')
    if show:
        plt.show()
    plt.close(fig)


def plot_displacement(t, dx_um, dy_um, stem, savefig, show):
    plt.rcParams.update({'font.size': 9})
    t_min  = t / 60.0
    t_norm = (t - t[0]) / max(t[-1] - t[0], 1.0)
    sub    = slice(None, None, max(1, len(t) // 4000))

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(13, 5),
                                             constrained_layout=True)
    fig.suptitle(f'A_shift displacement trajectory — {stem}', fontsize=11)

    # Left: dx and dy vs time
    ax_left.plot(t_min, dx_um, color='steelblue', lw=0.6, alpha=0.8, label='dx_um')
    ax_left.plot(t_min, dy_um, color='coral',     lw=0.6, alpha=0.8, label='dy_um')
    ax_left.axhline(0, color='k', lw=0.4, ls=':')
    ax_left.set_xlabel('Time [min]')
    ax_left.set_ylabel('Displacement [µm]')
    ax_left.set_title('dx and dy vs time', fontsize=9)
    ax_left.legend(fontsize=8)

    # Right: 2D path colored by time, start/end markers
    sc = ax_right.scatter(dx_um[sub], dy_um[sub], c=t_norm[sub], cmap='viridis',
                          s=2, alpha=0.7, rasterized=True)
    plt.colorbar(sc, ax=ax_right, label='Normalised time', shrink=0.8)
    ax_right.plot(dx_um[0], dy_um[0], 'go', ms=8, label='Start', zorder=5)
    ax_right.plot(dx_um[-1], dy_um[-1], 'rs', ms=8, label='End', zorder=5)
    ax_right.set_xlabel('dx [µm]')
    ax_right.set_ylabel('dy [µm]')
    ax_right.set_title('2D displacement path', fontsize=9)
    ax_right.legend(fontsize=8, loc='best')
    ax_right.axhline(0, color='k', lw=0.4, ls=':')
    ax_right.axvline(0, color='k', lw=0.4, ls=':')

    if savefig:
        fpath = os.path.join(RESULTS, f'reconstruct_v4_{stem}_displacement.png')
        fig.savefig(fpath, dpi=150, bbox_inches='tight')
        print(f'Saved: {fpath}')
    if show:
        plt.show()
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Viterbi sequential state matching (v4)')
    parser.add_argument('--stem',    required=True,
                        help='Experiment stem (eb11, eb04, ...)')
    parser.add_argument('--savefig', action='store_true',
                        help='Save figures to results/')
    parser.add_argument('--show',    action='store_true',
                        help='Call plt.show() after each figure')
    parser.add_argument('--cal',     action='store_true',
                        help='Enable Z0 calibration (default OFF)')
    parser.add_argument('--no_reg',  action='store_true',
                        help='Disable Viterbi, use greedy (same as v3)')
    parser.add_argument('--max_jump_A', type=float, default=3.0,
                        help='A_shift max jump in pixels (default 3.0)')
    parser.add_argument('--max_jump_B', type=float, default=0.3,
                        help='B_rb max jump normalized (default 0.3)')
    parser.add_argument('--max_jump_D', type=float, default=1.0,
                        help='D_rb max jump in grid index steps (default 1 = 26 Chebyshev neighbours in 3D)')
    parser.add_argument('--max_jump_C', type=float, default=3.0,
                        help='C_brickwall max jump in mode steps (default 3.0)')
    parser.add_argument('--chunk',   type=int, default=5000,
                        help='Chunk size for large-T Viterbi traceback')
    parser.add_argument('--batch_size', type=int, default=50000,
                        help='Signal batch size to limit peak memory for large recordings '
                             '(default 50000; 0 = no batching). Each batch seeds the next '
                             'from its terminal mode, enforcing continuity at boundaries.')
    parser.add_argument('--order',   nargs='+', default=DEFAULT_ORDER,
                        help='Matching order (library names)')
    args = parser.parse_args()

    stem  = args.stem
    order = args.order

    print(f'\n=== Viterbi reconstruction v4: {stem} ===\n')

    # ── Load experimental data ────────────────────────────────────────────────
    exp_npz = os.path.join(RESULTS, 'phase1_experimental.npz')
    if not os.path.exists(exp_npz):
        print('ERROR: phase1_experimental.npz not found. Run phase1_analysis.py first.')
        sys.exit(1)

    exp = np.load(exp_npz, allow_pickle=True)
    try:
        t      = exp[f'{stem}_t']
        dR     = exp[f'{stem}_dR']
        dX     = exp[f'{stem}_dX']
        f_meas = float(np.asarray(exp[f'{stem}_fmeas']).flat[0])
        mean_R = float(np.asarray(exp[f'{stem}_mean_R']).flat[0])
        mean_X = float(np.asarray(exp[f'{stem}_mean_X']).flat[0])
    except KeyError:
        print(f'ERROR: stem "{stem}" not found in phase1_experimental.npz.')
        sys.exit(1)

    dt     = float(np.median(np.diff(t)))
    Z0_exp = float(np.sqrt(mean_R**2 + mean_X**2))
    N_t    = len(t)

    print(f'  Experiment: {stem}   f={f_meas:.1f} Hz   N_t={N_t}   dt={dt:.3f} s')

    # ── Load libraries ────────────────────────────────────────────────────────
    print('\nLoading libraries at f={:.1f} Hz:'.format(f_meas))
    libs, d_um, Z0_sim_cx = load_libraries(f_meas)
    Z0_sim = abs(Z0_sim_cx)

    # ── Validate order ────────────────────────────────────────────────────────
    valid_order = []
    for lname in order:
        if lname not in LIBRARY_FILES:
            print(f'  WARNING: "{lname}" is not a known library name — skipping.')
        elif lname not in libs:
            print(f'  WARNING: "{lname}" was not loaded (file missing) — skipping.')
        else:
            valid_order.append(lname)
    if not valid_order:
        print('ERROR: no valid libraries in matching order.')
        sys.exit(1)
    # Pad to match requested order length if needed
    while len(valid_order) < len(order):
        valid_order.append(valid_order[-1])

    # ── Calibration ──────────────────────────────────────────────────────────
    cal_active = args.cal
    if cal_active and Z0_sim > 0.0:
        cal_factor = Z0_exp / Z0_sim
        print(f'\n  Calibration ACTIVE: cal={cal_factor:.4f}'
              f'  (Z0_exp={Z0_exp:.1f}, Z0_sim={Z0_sim:.1f})')
        for lname in libs:
            libs[lname]['dZ'] = libs[lname]['dZ'] * cal_factor
    else:
        cal_factor = 1.0
        print(f'\n  Calibration OFF  (Z0_exp={Z0_exp:.1f} Ohm, Z0_sim={Z0_sim:.1f} Ohm)')

    # ── Add null modes ────────────────────────────────────────────────────────
    print('\nAdding null modes:')
    add_null_modes(libs)

    # ── Print summary header ──────────────────────────────────────────────────
    print_summary(stem, valid_order, libs, Z0_exp, Z0_sim, cal_active, N_t, dt, None)

    # ── Greedy reconstruction (always run — fast baseline) ────────────────────
    print('Running greedy baseline:')
    greedy_res = greedy_reconstruct(libs, dR, dX, valid_order)

    std_dR = float(dR.std()) if dR.std() > 0 else 1.0
    std_dX = float(dX.std()) if dX.std() > 0 else 1.0

    print(
        f'\nGreedy baseline:  '
        f'RMSE dR={greedy_res["rmse_R"]:.2f} Ohm ({100*greedy_res["rmse_R"]/std_dR:.1f}% std)   '
        f'RMSE dX={greedy_res["rmse_X"]:.2f} Ohm ({100*greedy_res["rmse_X"]/std_dX:.1f}% std)'
    )

    # ── Viterbi reconstruction ────────────────────────────────────────────────
    if args.no_reg:
        print('\n--no_reg: using greedy result (Viterbi skipped).')
        res = greedy_res
        vit_stats = {}
    else:
        max_jumps = {
            'A_shift':     args.max_jump_A,
            'B_rb':        args.max_jump_B,
            'D_rb':        args.max_jump_D,
            'C_brickwall': args.max_jump_C,
        }
        print('\nRunning Viterbi reconstruction:')
        res = viterbi_reconstruct(libs, dR, dX, valid_order, max_jumps,
                                  chunk_size=args.chunk,
                                  signal_batch_size=args.batch_size if args.batch_size > 0 else None)
        vit_stats = res.get('vit_stats', {})

        print(
            f'\nViterbi v4:       '
            f'RMSE dR={res["rmse_R"]:.2f} Ohm ({100*res["rmse_R"]/std_dR:.1f}% std)   '
            f'RMSE dX={res["rmse_X"]:.2f} Ohm ({100*res["rmse_X"]/std_dX:.1f}% std)'
        )

        # Per-library diagnostics including pct_changed vs greedy
        if vit_stats:
            print('\nPer-library Viterbi diagnostics:')
            for lname in valid_order:
                if lname not in vit_stats:
                    continue
                st         = vit_stats[lname]
                vit_idx    = res['idx_dict'].get(lname)
                greedy_idx = greedy_res['idx_dict'].get(lname)
                if vit_idx is not None and greedy_idx is not None:
                    pct_changed = 100.0 * float(np.mean(vit_idx != greedy_idx))
                else:
                    pct_changed = float('nan')
                print(
                    f'  {lname:<12s}: '
                    f'mean_jump={st["mean_jump"]:.3f}  '
                    f'max_jump={st["max_jump_actual"]:.3f}  '
                    f'changed={pct_changed:.1f}%   '
                    f'{st["elapsed"]:.1f}s'
                )

    dR_rec   = res['dR_rec']
    dX_rec   = res['dX_rec']
    dR_resid = res['dR_resid']
    dX_resid = res['dX_resid']
    idx_dict = res['idx_dict']

    # ── Extract physical parameters ───────────────────────────────────────────
    dx_um, dy_um, k_rb_seq, lw_seq, da_seq, dRb_seq, dCm_seq, c_idx_seq = extract_physical(
        libs, idx_dict, d_um)

    # ── Save results ──────────────────────────────────────────────────────────
    idx_A = idx_dict.get('A_shift',     np.zeros(N_t, dtype=np.intp))
    idx_B = idx_dict.get('B_rb',        np.zeros(N_t, dtype=np.intp))
    idx_C = idx_dict.get('C_brickwall', np.zeros(N_t, dtype=np.intp))
    idx_D = idx_dict.get('D_rb',        np.zeros(N_t, dtype=np.intp))

    out_path = os.path.join(RESULTS, f'reconstruct_v4_{stem}.npz')
    np.savez(
        out_path,
        t          = t,
        dR_exp     = dR,
        dX_exp     = dX,
        dR_rec     = dR_rec,
        dX_rec     = dX_rec,
        dR_resid   = dR_resid,
        dX_resid   = dX_resid,
        idx_A      = idx_A,
        idx_B      = idx_B,
        idx_C      = idx_C,
        idx_D      = idx_D,
        da_seq     = da_seq,
        dRb_seq    = dRb_seq,
        dCm_seq    = dCm_seq,
        dx_um      = dx_um,
        dy_um      = dy_um,
        k_rb_seq   = k_rb_seq,
        lw_seq     = lw_seq,
        c_idx_seq  = c_idx_seq,
        rmse_R     = np.array(res['rmse_R']),
        rmse_X     = np.array(res['rmse_X']),
        rmse_total = np.array(res['rmse_total']),
        order      = np.array(','.join(valid_order)),
    )
    print(f'\nData saved: {out_path}')

    # ── Figures ───────────────────────────────────────────────────────────────
    plot_fit(t, dR, dX, dR_rec, dX_rec, dR_resid, dX_resid,
             stem, res, args.savefig, args.show)

    plot_phasespace(t, dR, dX, dR_rec, dX_rec, libs, valid_order,
                    stem, args.savefig, args.show)

    plot_states(t, dx_um, dy_um, k_rb_seq, lw_seq,
                da_seq, dRb_seq, dCm_seq,
                c_idx_seq, stem, args.savefig, args.show)

    plot_displacement(t, dx_um, dy_um, stem, args.savefig, args.show)

    print(f'\nRun: python reconstruct_v4.py --stem {stem} --savefig --show')


if __name__ == '__main__':
    main()
