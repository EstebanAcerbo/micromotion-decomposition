"""
baseline.py
===========
Compute the confluent-culture baseline impedance spectrum Z_baseline(f)
and save it to results/baseline.npz.

This file must be run before shift_library.py or pivot_library.py so
those scripts can load the reference spectrum.

Run from Micromotion_Simul/:
    python baseline.py
"""

import os
import numpy as np

from engine import build_geometry, solve_Z, FREQS

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
os.makedirs(RESULTS, exist_ok=True)
OUT = os.path.join(RESULTS, 'baseline.npz')


def main():
    print('Building confluent geometry ...')
    geo = build_geometry()
    p   = geo['p']

    print(
        f"  de={p['de']*1e4:.0f} um,  "
        f"side={geo['side']} px,  "
        f"d={geo['d']*1e7:.2f} nm,  "
        f"n_cols x n_rows={geo['n_cols']}x{geo['n_rows']},  "
        f"dim={geo['dim']:,}"
    )
    print(f"  spacing_x={geo['stencil'].spacing_x} px, "
          f"spacing_y={geo['stencil'].spacing_y} px")
    print(f'Solving at {len(FREQS)} frequencies ...')

    Z = solve_Z(geo, FREQS)

    np.savez(
        OUT,
        freqs     = FREQS,
        Z         = Z,
        d         = geo['d'],
        side      = geo['side'],
        spacing_x = geo['stencil'].spacing_x,
        spacing_y = geo['stencil'].spacing_y,
        n_cols    = geo['n_cols'],
        n_rows    = geo['n_rows'],
        de        = p['de'],
        ac        = p['ac'],
        hw_ratio  = p['hw_ratio'],
        n         = p['n'],
        A         = p['A'],
        B_abs     = p.get('B_abs', 0.0),
        N         = p['N'],
    )
    print(f'Saved -> {OUT}')

    # Sanity print at key frequencies
    for f_target in [100, 1000, 4000, 40000, 1e6]:
        idx = np.argmin(np.abs(FREQS - f_target))
        z   = Z[idx]
        print(f'  Z({FREQS[idx]:.0f} Hz) = {z.real:.2f} + {z.imag:.2f}j Ohm')


if __name__ == '__main__':
    main()
