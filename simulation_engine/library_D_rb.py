"""
library_D_rb.py  --  Library D (3-Parameter GK Perturbation Sweep)
====================================================================
Physical motivation
-------------------
GK sensitivity analysis shows that individual parameter sweeps produce
directions in (dR, dX) space that are either wrong or have large coupling.
Small simultaneous perturbations of (alpha, Rb, Cm) can produce new useful
directions:

    (+Rb, -Cm)          ->  angle ~-21 deg  (nearly pure +dR)
    (-alpha, +Rb, -Cm)  ->  angle ~+20 deg  (first quadrant: +dR, +dX)

This library sweeps all Cartesian combinations of fractional perturbations
for each of the three GK model parameters.

    Z(d_alpha, d_Rb, d_Cm) = solve_Z(geo with scaled params, freqs)

14^3 = 2744 modes covering all sign/magnitude combinations of
+/-{0.1%, 0.2%, 0.3%, 0.4%, 0.5%, 0.6%, 1.0%} perturbations in alpha, Rb, and Cm.

Run from Micromotion_Simul/:
    PYTHONIOENCODING=utf-8 python library_D_rb.py
    PYTHONIOENCODING=utf-8 python library_D_rb.py --append
        (append mode: only computes modes not already in library_D_rb.npz)
"""

import os
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('OMP_NUM_THREADS', '2')

import argparse
from itertools import product

import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

from engine import build_geometry, solve_Z, FREQS

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
os.makedirs(RESULTS, exist_ok=True)
OUT = os.path.join(RESULTS, 'library_D_rb.npz')

DELTA_VALS = [-0.010, -0.009, -0.008, -0.007, -0.006, -0.005, -0.004, -0.003, -0.002, -0.001,
              0.000,
              0.001,  0.002,  0.003,  0.004,  0.005,  0.006,  0.007,  0.008,  0.009,  0.010]  # fractional perturbations (+-0.1% to +-1.0% + zero, 21 values -> ~9261 modes)
N_JOBS = 6


# ── Top-level worker (module-level for joblib pickling) ───────────────────────

def _worker(k, d_alpha, d_Rb, d_Cm, geo, freqs):
    p     = geo['p']
    geo_k = {**geo, 'p': {
        **p,
        'alpha': complex(abs(p['alpha']) * (1.0 + d_alpha)),
        'Rb':    complex(abs(p['Rb'])    * (1.0 + d_Rb)),
        'Cm':    complex(abs(p['Cm'])    * (1.0 + d_Cm)),
    }}
    Z = solve_Z(geo_k, freqs)
    return k, Z


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--append', action='store_true',
                        help='Append only missing modes to existing npz (default: full recompute)')
    args = parser.parse_args()
    append_mode = args.append

    modes_list = list(product(DELTA_VALS, DELTA_VALS, DELTA_VALS))
    # 20 x 20 x 20 = 8000 modes; each entry: (delta_alpha, delta_Rb, delta_Cm)
    n_total = len(modes_list)

    # ── Append mode: load existing and filter ─────────────────────────────────
    existing_da = existing_dRb = existing_dCm = existing_Z = None
    if append_mode and os.path.exists(OUT):
        ex = np.load(OUT, allow_pickle=True)
        existing_da  = ex['delta_alpha']
        existing_dRb = ex['delta_Rb']
        existing_dCm = ex['delta_Cm']
        existing_Z   = ex['Z'].astype(complex)
        existing_set = set(
            zip(np.round(existing_da, 8),
                np.round(existing_dRb, 8),
                np.round(existing_dCm, 8))
        )
        modes_list = [
            m for m in modes_list
            if (round(m[0], 8), round(m[1], 8), round(m[2], 8)) not in existing_set
        ]
        print(f'Append mode: {len(existing_da)} modes already in file, {len(modes_list)} new modes to compute.')
    else:
        existing_set = set()
        if append_mode:
            print('Append mode: no existing file found, running full compute.')

    n_new = len(modes_list)

    print('Building confluent geometry ...')
    geo = build_geometry()
    p   = geo['p']

    alpha_base = abs(p['alpha'])
    Rb_base    = abs(p['Rb'])
    Cm_base    = abs(p['Cm'])

    print(
        f'  {geo["n_cols"]} x {geo["n_rows"]} cells,  side={geo["side"]} px\n'
        f'  alpha_base = {alpha_base:.6g}\n'
        f'  Rb_base    = {Rb_base:.4f} Ohm*cm2\n'
        f'  Cm_base    = {Cm_base:.4e} F/cm2\n'
        f'  DELTA_VALS = {DELTA_VALS}\n'
        f'  Total modes: {n_total}  ({len(DELTA_VALS)}^3 Cartesian product)  |  New to compute: {n_new}'
        + ('  [append mode]' if append_mode else '  [full recompute]')
    )

    # ── Baseline (loaded before parallel loop so checkpoints are complete) ──────
    baseline_path = os.path.join(RESULTS, 'baseline.npz')
    if os.path.exists(baseline_path):
        Z_baseline = np.load(baseline_path)['Z'].astype(complex)
        print('Baseline loaded from results/baseline.npz')
    else:
        print('Baseline not found -- computing inline ...')
        Z_baseline = solve_Z(geo, FREQS)

    # ── Accumulator: start from any existing modes ────────────────────────────
    # (works for both fresh run and append mode)
    acc_da  = list(existing_da.tolist()  if existing_da  is not None else [])
    acc_dRb = list(existing_dRb.tolist() if existing_dRb is not None else [])
    acc_dCm = list(existing_dCm.tolist() if existing_dCm is not None else [])
    acc_Z   = list(existing_Z            if existing_Z   is not None
                   else np.zeros((0, len(FREQS)), dtype=complex))

    # ── Checkpoint helper — atomic write via temp file ────────────────────────
    CHECKPOINT_EVERY = 100  # save to disk after every N new modes

    def _checkpoint():
        # Write directly to OUT — no temp/rename needed on Windows
        np.savez(
            OUT,
            delta_alpha = np.array(acc_da),
            delta_Rb    = np.array(acc_dRb),
            delta_Cm    = np.array(acc_dCm),
            delta_vals  = np.array(DELTA_VALS),
            alpha_base  = alpha_base,
            Rb_base     = Rb_base,
            Cm_base     = Cm_base,
            freqs       = FREQS,
            Z           = np.array(acc_Z, dtype=complex),
            Z_baseline  = Z_baseline,
            d           = geo['d'],
            spacing_x   = geo['stencil'].spacing_x,
        )

    # ── Parallel solve in batches with periodic checkpointing ─────────────────
    batches   = [modes_list[i:i + CHECKPOINT_EVERY]
                 for i in range(0, n_new, CHECKPOINT_EVERY)]
    n_batches = len(batches)
    print(f'\nSweeping (alpha, Rb, Cm) simultaneously '
          f'({n_batches} batches of up to {CHECKPOINT_EVERY}, checkpoint after each):')

    with tqdm(total=n_new, desc='D_rb 3-param sweep', ncols=80) as pbar:
        for batch in batches:
            batch_results = Parallel(n_jobs=N_JOBS)(
                delayed(_worker)(k, m[0], m[1], m[2], geo, FREQS)
                for k, m in enumerate(batch)
            )
            batch_results.sort(key=lambda x: x[0])   # ensure order
            for k, Z in batch_results:
                acc_da.append(batch[k][0])
                acc_dRb.append(batch[k][1])
                acc_dCm.append(batch[k][2])
                acc_Z.append(Z)
            pbar.update(len(batch))
            _checkpoint()   # atomic save after every batch

    da_arr  = np.array(acc_da)
    dRb_arr = np.array(acc_dRb)
    dCm_arr = np.array(acc_dCm)
    Z_lib   = np.array(acc_Z, dtype=complex)
    print(f'Total modes in library: {len(da_arr)}')

    # ── Final save (canonical — checkpoints already written incrementally) ──────
    np.savez(
        OUT,
        delta_alpha = da_arr,                         # (N,) float -- fractional change in alpha
        delta_Rb    = dRb_arr,                        # (N,) float -- fractional change in Rb
        delta_Cm    = dCm_arr,                        # (N,) float -- fractional change in Cm
        delta_vals  = np.array(DELTA_VALS),           # (20,) the perturbation levels
        alpha_base  = alpha_base,                     # scalar
        Rb_base     = Rb_base,                        # scalar
        Cm_base     = Cm_base,                        # scalar
        freqs       = FREQS,
        Z           = Z_lib,                          # (N, n_freq) complex
        Z_baseline  = Z_baseline,
        d           = geo['d'],
        spacing_x   = geo['stencil'].spacing_x,
    )
    print(f'Saved -> {OUT}')

    # ── Summary at 1 kHz ───────────────────────────────────────────────────────
    i1k = np.argmin(np.abs(FREQS - 1000))
    dZ  = Z_lib[:, i1k] - Z_baseline[i1k]

    angles     = np.degrees(np.arctan2(dZ.imag, dZ.real))
    magnitudes = np.abs(dZ)

    sort_idx = np.argsort(angles)

    print(f'\ndZ at 1 kHz  (|Z0| = {abs(Z_baseline[i1k]):.2f} Ohm):')
    print(f'  {"da":>8}  {"dRb":>8}  {"dCm":>8}  {"dR [Ohm]":>12}  {"dX [Ohm]":>12}  {"angle [deg]":>11}  {"|dZ| [Ohm]":>10}')
    for i in sort_idx:
        print(
            f'  {da_arr[i]:+8.4f}  {dRb_arr[i]:+8.4f}  {dCm_arr[i]:+8.4f}  '
            f'{dZ[i].real:+12.4f}  {dZ[i].imag:+12.4f}  '
            f'{angles[i]:+11.3f}  {magnitudes[i]:10.4f}'
        )

    print("""
Library D physical interpretation
----------------------------------
This library explores the 3D GK parameter perturbation space near baseline.
Each mode is a simultaneous small change in alpha, Rb, and Cm.
Covers all 20^3=8000 sign/magnitude combinations of +/-{0.1,...,1.0}%
perturbations (uniform 0.1% step).

Key findings from gk_sensitivity:
  (+Rb, -Cm)          -> angle ~-21 deg  (near pure +dR)
  (-alpha, +Rb, -Cm)  -> angle ~+20 deg  (first quadrant, fills new direction)
""")


if __name__ == '__main__':
    main()
