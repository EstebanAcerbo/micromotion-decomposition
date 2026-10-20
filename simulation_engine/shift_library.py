"""
shift_library.py  —  Library A: Electrode Offset Sweep (lower-triangle)
=========================================================================
Sweeps all topologically distinct (dx, dy) electrode-centre offsets in a
lower-triangle pattern from (0,0) to (max_shift, max_shift), where
max_shift = 40 (two full cell-spacings; spacing_x ≈ 20 px).

Rationale for max_shift = 40
-----------------------------
  spacing_x ≈ 20 px → one full cell period.
  max_shift = 23 (old) covered up to the centre of the adjacent cell.
  max_shift = 40 extends to the centre of the diagonal next-nearest cell,
  expanding library coverage in (dR, dX) phase space to match larger
  experimental fluctuations.  Beyond 40 px the geometry repeats (period 20).

Sweep definition
----------------
  dx in [0, max_shift] inclusive
  dy in [0, dx] inclusive (lower triangle including diagonal)
  Total modes: (max_shift+1)*(max_shift+2)/2 = 41*42/2 = 861

Storage: flat 1-D mode index k
  Z_lib   : (N_modes, n_freq)
  dx_px   : (N_modes,)  int
  dy_px   : (N_modes,)  int

Symmetry check
--------------
Three extra points beyond the diagonal are computed and stored separately:
  dx = max_shift+1, max_shift+2, max_shift+3  with dy = dx  (diagonal)
  Saved as Z_sym_check (3 x n_freq) and sym_dx_vals ([41,42,43]).

Parallelisation: joblib with N_JOBS processes.
Set N_JOBS x MKL_THREADS equal to your physical core count.

Run from Micromotion_Simul/:
    python shift_library.py
"""

import os

# MKL thread count must be set before numpy/scipy is imported.
N_JOBS      = 7   # parallel worker processes
MKL_THREADS = 2   # MKL threads per worker  (N_JOBS x MKL_THREADS = 14 physical cores)
os.environ.setdefault('MKL_NUM_THREADS', str(MKL_THREADS))
os.environ.setdefault('OMP_NUM_THREADS', str(MKL_THREADS))

import numpy as np
from joblib import Parallel, delayed
from tqdm   import tqdm

from engine import build_geometry, substrate_offset, solve_Z, FREQS

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
os.makedirs(RESULTS, exist_ok=True)
OUT = os.path.join(RESULTS, 'library_A_shift.npz')


# ── Worker (runs in a child process) ─────────────────────────────────────────

def _worker(k, dx, dy, geo, f):
    """Solve Z for one electrode offset. Returns (k, Z)."""
    Sub = substrate_offset(geo, x_px=dx, y_px=dy)
    Z   = solve_Z(geo, f, Sub=Sub)
    return k, Z


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print('Building confluent geometry ...')
    geo = build_geometry(extra_pad=2)
    sx  = geo['stencil'].spacing_x   # horizontal cell spacing [px]
    sy  = geo['stencil'].spacing_y   # vertical cell spacing   [px]

    # Fixed max_shift: two full cell-spacings (spacing_x ≈ 20 px)
    # Old: n + 3 = 23 (centre of adjacent cell only)
    # New: 40 = centre of diagonal next-nearest cell, doubles phase-space coverage
    n         = geo['p']['n']
    max_shift = 40

    # Build lower-triangle mode list: dx in [0, max_shift], dy in [0, dx]
    modes = [(dx, dy) for dx in range(max_shift + 1) for dy in range(dx + 1)]
    N_modes = len(modes)   # = (max_shift+1)*(max_shift+2)//2

    dx_arr = np.array([m[0] for m in modes], dtype=int)
    dy_arr = np.array([m[1] for m in modes], dtype=int)

    print(
        f'Lower-triangle sweep: max_shift={max_shift}  N_modes={N_modes}\n'
        f'  dx in [0, {max_shift}] px  '
        f'({0:.1f} to {max_shift * geo["d"] * 1e7:.1f} nm)\n'
        f'  dy in [0, dx] px'
    )
    print(f'Running {N_JOBS} workers x {MKL_THREADS} MKL threads ...')

    # ── Main sweep ────────────────────────────────────────────────────────────
    tasks = [(k, int(dx), int(dy)) for k, (dx, dy) in enumerate(modes)]

    Z_lib = np.zeros((N_modes, len(FREQS)), dtype=complex)

    results = Parallel(n_jobs=N_JOBS)(
        delayed(_worker)(k, dx, dy, geo, FREQS)
        for k, dx, dy in tqdm(tasks, desc='Shift sweep', ncols=80)
    )

    for k, Z in results:
        Z_lib[k, :] = Z

    # ── Symmetry check points (dx=24,25,26 along diagonal) ───────────────────
    sym_dx_vals = np.array([max_shift + 1, max_shift + 2, max_shift + 3], dtype=int)
    sym_tasks   = [(i, int(v), int(v)) for i, v in enumerate(sym_dx_vals)]

    print(f'Computing {len(sym_dx_vals)} symmetry-check points ...')
    sym_results = Parallel(n_jobs=min(N_JOBS, len(sym_dx_vals)))(
        delayed(_worker)(i, dx, dy, geo, FREQS)
        for i, dx, dy in tqdm(sym_tasks, desc='Sym check', ncols=80)
    )

    Z_sym_check = np.zeros((len(sym_dx_vals), len(FREQS)), dtype=complex)
    for i, Z in sym_results:
        Z_sym_check[i, :] = Z

    # ── Baseline ──────────────────────────────────────────────────────────────
    # Always recompute from the same extra_pad=2 geometry so that
    # dZ = Z_mode - Z_baseline is consistent (results/baseline.npz uses the
    # standard domain and must not be used here).
    print('Computing baseline from extra_pad=2 geometry ...')
    Z_baseline = solve_Z(geo, FREQS)

    np.savez(
        OUT,
        dx_px        = dx_arr,
        dy_px        = dy_arr,
        freqs        = FREQS,
        Z            = Z_lib,
        Z_baseline   = Z_baseline,
        Z_sym_check  = Z_sym_check,
        sym_dx_vals  = sym_dx_vals,
        d            = geo['d'],
        spacing_x    = sx,
        spacing_y    = sy,
        n_param      = n,
        max_shift    = max_shift,
        extra_pad    = 2,
    )
    print(f'Saved -> {OUT}')

    # Summary at 1 kHz
    i1k = np.argmin(np.abs(FREQS - 1000))
    dZ  = Z_lib[:, i1k] - Z_baseline[i1k]
    print(
        f'\nN_modes = {N_modes}  (lower triangle 0..{max_shift})\n'
        f'  dx range: [0, {max_shift}] px\n'
        f'  dy range: [0, dx] px\n'
        f'\nAt 1 kHz  |Z0| = {abs(Z_baseline[i1k]):.2f} Ohm\n'
        f'  Max |dZ| = {np.abs(dZ).max():.3f} Ohm  '
        f'({100*np.abs(dZ).max()/abs(Z_baseline[i1k]):.4f} %)\n'
        f'  Mean |dZ| = {np.abs(dZ).mean():.3f} Ohm  '
        f'({100*np.abs(dZ).mean()/abs(Z_baseline[i1k]):.4f} %)'
    )


if __name__ == '__main__':
    main()
