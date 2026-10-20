"""
lib_verify.py  --  CPE-Cancellation Verification
=================================================
Verifies that changing the CPE coefficient A (from the current value, which
represents the two-electrode fit, to the correct per-electrode value A*2) does
NOT change the perturbation dZ = Z_perturbed - Z_baseline stored in each
library.

Theory: dZ = Z_pert(A) - Z_base(A).  Since the CPE term is identical in
both Z_pert and Z_base (the bare electrode admittance does not change with
cell perturbation), it cancels exactly.  This must hold for all libraries.

For each of the four active libraries (A_shift, B_rb, D_rb, C_brickwall) the
script:
  1. Randomly selects N_SAMPLE modes from the existing npz.
  2. Re-solves the FD system for each mode with both A_wrong and A_correct.
  3. Computes dZ_wrong = Z_pert_wrong - Z_base_wrong
             dZ_correct = Z_pert_correct - Z_base_correct
  4. Reports max |dZ_correct - dZ_wrong| across all 30 frequencies, normalised
     by the typical |dZ| magnitude (relative residual in %).

A new baseline is also computed with both A values so that the Z0_sim
discrepancy is quantified separately.

Usage
-----
    python lib_verify.py [--n N]   (default N=20)
"""

import os, sys, copy, argparse, time
import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))

from engine import (
    build_geometry, solve_Z, substrate_offset,
    build_l_corner_culture, apply_brick_offset, FREQS, P,
)

RESULTS = os.path.join(HERE, '..', 'results')

LIBRARY_FILES = {
    'A_shift':     os.path.join(RESULTS, 'library_A_shift.npz'),
    'B_rb':        os.path.join(RESULTS, 'library_B_rb.npz'),
    'D_rb':        os.path.join(RESULTS, 'library_D_rb.npz'),
    'C_brickwall': os.path.join(RESULTS, 'library_C_brickwall.npz'),
}

A_WRONG   = P['A']              # current value (2-electrode fit)
A_CORRECT = 2.0 * P['A']       # per-electrode value (double admittance = half CPE Z)

RNG_SEED  = 42

# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_geos():
    """Return (geo_wrong, geo_correct) — identical except for CPE parameter A."""
    geo_wrong   = build_geometry(extra_pad=2)   # uses engine.P as-is
    p_correct   = {**geo_wrong['p'], 'A': A_CORRECT}
    geo_correct = {**geo_wrong, 'p': p_correct}
    return geo_wrong, geo_correct


def _solve_baseline(geo, freqs):
    return solve_Z(geo, freqs)


def _dZ(Z_pert, Z_base):
    """dZ per frequency: shape (n_freq,) complex."""
    return Z_pert - Z_base


def _report_lib(name, diffs_dZ, dZ_wrong_stack, n_sample):
    """Print a summary table for one library."""
    # diffs_dZ    : (n_sample, n_freq) complex — dZ_correct - dZ_wrong
    # dZ_wrong_stack : (n_sample, n_freq) complex
    abs_diff  = np.abs(diffs_dZ)                      # (n_sample, n_freq)
    abs_dZ    = np.abs(dZ_wrong_stack)
    rel_diff  = 100.0 * abs_diff / (abs_dZ + 1e-12)  # %

    max_abs  = abs_diff.max()
    max_rel  = rel_diff.max()
    mean_abs = abs_diff.mean()
    mean_rel = rel_diff.mean()

    print(f'\n  {name}  ({n_sample} modes, {diffs_dZ.shape[1]} freqs)')
    print(f'    max  |dZ_correct - dZ_wrong|  = {max_abs:.4e} Ohm   ({max_rel:.4f} %)')
    print(f'    mean |dZ_correct - dZ_wrong|  = {mean_abs:.4e} Ohm   ({mean_rel:.4f} %)')

    # worst mode
    worst_m, worst_f = np.unravel_index(abs_diff.argmax(), abs_diff.shape)
    print(f'    worst mode idx={worst_m}  f={FREQS[worst_f]:.1f} Hz  '
          f'|diff|={abs_diff[worst_m, worst_f]:.4e} Ohm')


# ── Per-library verification ──────────────────────────────────────────────────

def verify_A(n_sample, geo_wrong, geo_correct, Z_base_wrong, Z_base_correct):
    d = np.load(LIBRARY_FILES['A_shift'], allow_pickle=True)
    dx_all = d['dx_px'].astype(int)
    dy_all = d['dy_px'].astype(int)
    n_modes = len(dx_all)

    rng = np.random.default_rng(RNG_SEED)
    idx = rng.choice(n_modes, size=min(n_sample, n_modes), replace=False)

    diffs = []
    dZ_w  = []
    for i in idx:
        Sub_w = substrate_offset(geo_wrong,   x_px=int(dx_all[i]), y_px=int(dy_all[i]))
        Sub_c = substrate_offset(geo_correct, x_px=int(dx_all[i]), y_px=int(dy_all[i]))
        Zw = solve_Z(geo_wrong,   FREQS, Sub=Sub_w)
        Zc = solve_Z(geo_correct, FREQS, Sub=Sub_c)
        dZw = _dZ(Zw, Z_base_wrong)
        dZc = _dZ(Zc, Z_base_correct)
        diffs.append(dZc - dZw)
        dZ_w.append(dZw)

    _report_lib('A_shift', np.array(diffs), np.array(dZ_w), len(idx))


def verify_B(n_sample, geo_wrong, geo_correct, Z_base_wrong, Z_base_correct):
    d = np.load(LIBRARY_FILES['B_rb'], allow_pickle=True)
    k_rb_all = d['k_rb_arr'].astype(float)
    lw_all   = d['lw_arr'].astype(int)
    n_modes  = len(k_rb_all)

    rng    = np.random.default_rng(RNG_SEED)
    idx    = rng.choice(n_modes, size=min(n_sample, n_modes), replace=False)
    corner = 1   # UR corner — same as library_B_rb.py
    max_lw = geo_wrong['stencil'].spacing_x - 2
    p_w    = geo_wrong['p']
    Rb_base = abs(p_w['Rb'])

    diffs = []
    dZ_w  = []
    for i in idx:
        k_rb = float(k_rb_all[i])
        lw   = int(lw_all[i])
        Rb_lw = max(Rb_base * (1.0 + k_rb * lw / max_lw), 1e-6)

        geo_lw_w = {**geo_wrong,   'p': {**geo_wrong['p'],   'l_width': lw, 'Rb': complex(Rb_lw)}}
        geo_lw_c = {**geo_correct, 'p': {**geo_correct['p'], 'l_width': lw, 'Rb': complex(Rb_lw)}}

        Cult_w, _ = build_l_corner_culture(geo_lw_w, corner=corner, all_cells=True)
        Cult_c, _ = build_l_corner_culture(geo_lw_c, corner=corner, all_cells=True)

        Zw = solve_Z(geo_lw_w, FREQS, Cult=Cult_w)
        Zc = solve_Z(geo_lw_c, FREQS, Cult=Cult_c)
        dZw = _dZ(Zw, Z_base_wrong)
        dZc = _dZ(Zc, Z_base_correct)
        diffs.append(dZc - dZw)
        dZ_w.append(dZw)

    _report_lib('B_rb', np.array(diffs), np.array(dZ_w), len(idx))


def verify_D(n_sample, geo_wrong, geo_correct, Z_base_wrong, Z_base_correct):
    d = np.load(LIBRARY_FILES['D_rb'], allow_pickle=True)
    da_all  = d['delta_alpha'].astype(float)
    dRb_all = d['delta_Rb'].astype(float)
    dCm_all = d['delta_Cm'].astype(float)
    n_modes = len(da_all)

    rng = np.random.default_rng(RNG_SEED)
    idx = rng.choice(n_modes, size=min(n_sample, n_modes), replace=False)

    p_w = geo_wrong['p']
    p_c = geo_correct['p']

    diffs = []
    dZ_w  = []
    for i in idx:
        geo_k_w = {**geo_wrong, 'p': {
            **p_w,
            'alpha': complex(abs(p_w['alpha']) * (1.0 + da_all[i])),
            'Rb':    complex(abs(p_w['Rb'])    * (1.0 + dRb_all[i])),
            'Cm':    complex(abs(p_w['Cm'])    * (1.0 + dCm_all[i])),
        }}
        geo_k_c = {**geo_correct, 'p': {
            **p_c,
            'alpha': complex(abs(p_c['alpha']) * (1.0 + da_all[i])),
            'Rb':    complex(abs(p_c['Rb'])    * (1.0 + dRb_all[i])),
            'Cm':    complex(abs(p_c['Cm'])    * (1.0 + dCm_all[i])),
        }}
        Zw = solve_Z(geo_k_w, FREQS)
        Zc = solve_Z(geo_k_c, FREQS)
        dZw = _dZ(Zw, Z_base_wrong)
        dZc = _dZ(Zc, Z_base_correct)
        diffs.append(dZc - dZw)
        dZ_w.append(dZw)

    _report_lib('D_rb', np.array(diffs), np.array(dZ_w), len(idx))


def verify_C(n_sample, geo_wrong, geo_correct, Z_base_wrong, Z_base_correct):
    d = np.load(LIBRARY_FILES['C_brickwall'], allow_pickle=True)
    delta_all = d['delta'].astype(float)
    n_modes   = len(delta_all)

    rng = np.random.default_rng(RNG_SEED)
    idx = rng.choice(n_modes, size=min(n_sample, n_modes), replace=False)

    diffs = []
    dZ_w  = []
    for i in idx:
        delta = float(delta_all[i])
        Cult_w = apply_brick_offset(geo_wrong,   delta)
        Cult_c = apply_brick_offset(geo_correct, delta)
        Zw = solve_Z(geo_wrong,   FREQS, Cult=Cult_w)
        Zc = solve_Z(geo_correct, FREQS, Cult=Cult_c)
        dZw = _dZ(Zw, Z_base_wrong)
        dZc = _dZ(Zc, Z_base_correct)
        diffs.append(dZc - dZw)
        dZ_w.append(dZw)

    _report_lib('C_brickwall', np.array(diffs), np.array(dZ_w), len(idx))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='CPE-cancellation verification')
    parser.add_argument('--n', type=int, default=20,
                        help='Number of random modes to sample per library (default 20)')
    args = parser.parse_args()
    n_sample = args.n

    print('=' * 65)
    print('  lib_verify.py — CPE cancellation check')
    print('=' * 65)
    print(f'\n  A_wrong   = {A_WRONG:.4e}  (current engine.P, two-electrode fit)')
    print(f'  A_correct = {A_CORRECT:.4e}  (per-electrode = 2 × A_wrong)')
    print(f'  Sampling {n_sample} modes per library  (seed={RNG_SEED})')

    # ── Check library files ────────────────────────────────────────────────────
    missing = [k for k, v in LIBRARY_FILES.items() if not os.path.exists(v)]
    if missing:
        print(f'\nERROR: missing library files: {missing}')
        sys.exit(1)

    # ── Build geometries ───────────────────────────────────────────────────────
    print('\nBuilding geometries ...')
    t0 = time.perf_counter()
    geo_wrong, geo_correct = _build_geos()
    print(f'  done ({time.perf_counter()-t0:.1f} s)')

    # ── Baselines ─────────────────────────────────────────────────────────────
    print('\nComputing baselines (wrong A and correct A) ...')
    t0 = time.perf_counter()
    Z_base_wrong   = _solve_baseline(geo_wrong,   FREQS)
    Z_base_correct = _solve_baseline(geo_correct, FREQS)
    print(f'  done ({time.perf_counter()-t0:.1f} s)')

    i1k = int(np.argmin(np.abs(FREQS - 1e3)))
    dZ_base = Z_base_correct[i1k] - Z_base_wrong[i1k]
    print(f'\n  Baseline comparison at 1 kHz:')
    print(f'    |Z_base_wrong|   = {abs(Z_base_wrong[i1k]):.2f} Ohm')
    print(f'    |Z_base_correct| = {abs(Z_base_correct[i1k]):.2f} Ohm')
    print(f'    |Z_base_correct - Z_base_wrong| = {abs(dZ_base):.4f} Ohm')
    print(f'    (Expected: non-zero — baseline changes with A; dZ does not)')

    # ── Per-library verification ───────────────────────────────────────────────
    print(f'\n{"─"*65}')
    print('  Library dZ comparison (should be ≈ 0.0000 Ohm / 0.0000 %)')
    print(f'{"─"*65}')

    t0 = time.perf_counter()
    verify_A(n_sample, geo_wrong, geo_correct, Z_base_wrong, Z_base_correct)
    print(f'    A_shift done  ({time.perf_counter()-t0:.1f} s)')

    t0 = time.perf_counter()
    verify_B(n_sample, geo_wrong, geo_correct, Z_base_wrong, Z_base_correct)
    print(f'    B_rb done  ({time.perf_counter()-t0:.1f} s)')

    t0 = time.perf_counter()
    verify_D(n_sample, geo_wrong, geo_correct, Z_base_wrong, Z_base_correct)
    print(f'    D_rb done  ({time.perf_counter()-t0:.1f} s)')

    t0 = time.perf_counter()
    verify_C(n_sample, geo_wrong, geo_correct, Z_base_wrong, Z_base_correct)
    print(f'    C_brickwall done  ({time.perf_counter()-t0:.1f} s)')

    print(f'\n{"="*65}')
    print('  PASS if all max |diff| < 1e-6 Ohm and < 1e-4 %')
    print(f'{"="*65}\n')


if __name__ == '__main__':
    main()
