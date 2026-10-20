"""
shift2_library.py  --  Library C: Brick-Wall Alternating Column Offset
========================================================================
Physical mechanism
------------------
In a confluent monolayer the cells pack in a square grid (all columns
aligned).  A brick-wall (staggered) offset shifts alternate cell-columns
by delta * spacing_y / 2 pixels in the y-direction, producing a
hexagonal-like packing geometry.  At delta = 0 the configuration is
identical to the baseline; at delta = 1 (= 0.5 * spacing_y) the
packing is maximally staggered.

This perturbation sweeps the structural degree of freedom that describes
how much the monolayer has "staggered" relative to the electrode.

Sweep: delta in {0.00, 0.05, 0.10, ..., 0.50}  (11 points)

Run from Micromotion_Simul/:
    PYTHONIOENCODING=utf-8 python shift2_library.py
"""

import os
import numpy as np
from tqdm import tqdm

from engine import build_geometry, apply_brick_offset, solve_Z, FREQS

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
os.makedirs(RESULTS, exist_ok=True)
OUT = os.path.join(RESULTS, 'library_C_brickwall.npz')

# Delta sweep: 0 to 0.5 in steps of 0.05 (11 points)
DELTA_VALS = np.round(np.arange(-0.50, 0.51, 0.05), 20)


def main():
    print('Building confluent geometry ...')
    geo = build_geometry()
    sx  = geo['stencil'].spacing_x
    sy  = geo['stencil'].spacing_y
    d   = geo['d']

    print(
        f'  {geo["n_cols"]} x {geo["n_rows"]} cells,  '
        f'side={geo["side"]} px,  '
        f'spacing_x={sx} px,  spacing_y={sy} px\n'
        f'  delta sweep: {DELTA_VALS}\n'
        f'  offset range: 0 to {0.5*sy:.1f} px '
        f'= 0 to {0.5*sy*d*1e7:.1f} nm'
    )

    n_delta = len(DELTA_VALS)
    Z_lib   = np.zeros((n_delta, len(FREQS)), dtype=complex)
    offset_px_actual = np.zeros(n_delta, dtype=int)

    for i, delta in enumerate(tqdm(DELTA_VALS, desc='Brick-wall sweep', ncols=70)):
        offset_px_actual[i] = int(round(delta * sy))
        Cult = apply_brick_offset(geo, delta)
        Z_lib[i] = solve_Z(geo, FREQS, Cult=Cult)

    # Load or compute baseline
    baseline_path = os.path.join(RESULTS, 'baseline.npz')
    if os.path.exists(baseline_path):
        Z_baseline = np.load(baseline_path)['Z'].astype(complex)
        print('Baseline loaded from results/baseline.npz')
    else:
        print('Baseline not found -- computing inline ...')
        Z_baseline = solve_Z(geo, FREQS)

    np.savez(
        OUT,
        delta      = DELTA_VALS,
        offset_px  = offset_px_actual,
        offset_nm  = offset_px_actual * d * 1e7,
        freqs      = FREQS,
        Z          = Z_lib,
        Z_baseline = Z_baseline,
        d          = d,
        spacing_x  = sx,
        spacing_y  = sy,
    )
    print(f'Saved -> {OUT}')

    # Summary at 1 kHz
    i1k = np.argmin(np.abs(FREQS - 1000))
    dZ  = Z_lib[:, i1k] - Z_baseline[i1k]
    print(f'\ndZ at 1 kHz  (|Z0| = {abs(Z_baseline[i1k]):.2f} Ohm):')
    print(f'  {"delta":>6}  {"offset_px":>9}  {"dR [Ohm]":>12}  {"dX [Ohm]":>12}  {"pct":>9}')
    for i, delta in enumerate(DELTA_VALS):
        pct = 100 * abs(dZ[i]) / abs(Z_baseline[i1k])
        print(
            f'  {delta:6.3f}  {offset_px_actual[i]:9d}  '
            f'{dZ[i].real:+12.4f}  {dZ[i].imag:+12.4f}  {pct:9.5f}%'
        )


if __name__ == '__main__':
    main()
