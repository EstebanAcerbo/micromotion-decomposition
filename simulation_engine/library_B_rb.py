"""
library_B_rb.py  --  Library B (Multi-Corner + Rb-Scaled Sweep)
================================================================
Physical mechanism
------------------
Five corner combinations (CCW filling order) are applied to every live cell,
each combined with an arm-length sweep and a K_RB-scaled Rb:

    combo 0: [0]           -- UL only
    combo 1: [0, 1]        -- UL + UR
    combo 2: [0, 3]        -- UL + LR  (diagonal)
    combo 3: [0, 1, 3]     -- UL + UR + LR
    combo 4: [0, 1, 2, 3]  -- all four corners

For each (combo, lw, K_RB) triplet:

    Rb(lw, K_RB) = max(Rb_base * (1 + K_RB * lw / max_lw), 1e-6)

K_RB=0 is pure geometry (Rb unchanged); K_RB>0 tightens junctions as arm
grows; K_RB<0 models cell retraction (looser junctions).

Total modes (fresh run): 5 combos x 5 lw x 16 K_RB = 400 modes.

Append mode (python library_B_rb.py --append):
  Loads existing npz, identifies already-computed (combo_idx, lw, k_rb)
  triplets, builds only missing tasks, concatenates, and saves.

Run from Micromotion_Simul/:
    PYTHONIOENCODING=utf-8 python library_B_rb.py [--append]
"""

import os
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('OMP_NUM_THREADS', '2')

import sys
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

from engine import (
    build_geometry, solve_Z,
    build_fixed_corners_culture,
    FREQS, SystemConst,
)

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
os.makedirs(RESULTS, exist_ok=True)
OUT = os.path.join(RESULTS, 'library_B_rb.npz')

# Corner combinations (CCW filling)
CORNER_COMBOS = [
#    [0],          # combo 0: UL only
    [0, 1],       # combo 1: UL + UR
#    [0, 3],       # combo 2: UL + LR (diagonal)
    [0, 1, 3],    # combo 3: UL + UR + LR
    [0, 1, 2, 3], # combo 4: all four
]

LW_MAX    = 10
#K_RB_VALS = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5,  0.6,  0.7,  0.8,  0.9,  1.0]
K_RB_VALS = [0,.5,1,1.5,2]
N_JOBS    = 8


# combo_def: (5, 4) int8, padded with -1
_COMBO_DEF = np.full((5, 4), -1, dtype=np.int8)
for _i, _c in enumerate(CORNER_COMBOS):
    _COMBO_DEF[_i, :len(_c)] = _c


# ── Top-level worker (module-level for joblib pickling) ───────────────────────

def _worker(task_idx, combo_corners, k_rb, lw, geo, freqs):
    """Solve Z for one (combo, K_RB, lw) combination."""
    p       = geo['p']
    Rb_base = abs(p['Rb'])
    max_lw  = geo['stencil'].spacing_x - 2
    Rb_lw   = max(Rb_base * (1.0 + k_rb * float(lw) / max_lw), 1e-6)
    geo_lw  = {**geo, 'p': {**p, 'l_width': int(lw), 'Rb': complex(Rb_lw)}}
    Cult, _ = build_fixed_corners_culture(geo_lw, combo_corners)
    Z       = solve_Z(geo_lw, freqs, Cult=Cult)
    return task_idx, Z


def _build_tasks(combo_idx_list, lw_list, k_rb_list):
    """Return flat task list as list of (task_idx, combo_idx, k_rb, lw)."""
    tasks = []
    idx   = 0
    for ci in combo_idx_list:
        for k in k_rb_list:
            for lw in lw_list:
                tasks.append((idx, ci, k, int(lw)))
                idx += 1
    return tasks


def main():
    append_mode = '--append' in sys.argv

    print('Building confluent geometry ...')
    geo = build_geometry()
    p   = geo['p']
    sx  = geo['stencil'].spacing_x
    max_lw  = sx - 2
    Rb_base = abs(p['Rb'])

    lw_range     = list(range(1, LW_MAX + 1))
    n_combos     = len(CORNER_COMBOS)
    n_k          = len(K_RB_VALS)
    n_modes_total = n_combos * LW_MAX * n_k

    print(
        f'  {geo["n_cols"]} x {geo["n_rows"]} cells,  side={geo["side"]} px\n'
        f'  max_lw (interior)={max_lw},  LW_MAX={LW_MAX}\n'
        f'  Rb_base={Rb_base:.2f} Ohm*cm2\n'
        f'  K_RB_VALS={K_RB_VALS}\n'
        f'  Corner combos: {n_combos}\n'
        f'  Total modes (full): {n_modes_total}  '
        f'({n_combos} combos x {LW_MAX} lw x {n_k} K_RB)'
    )

    # ── Baseline ───────────────────────────────────────────────────────────────
    baseline_path = os.path.join(RESULTS, 'baseline.npz')
    if os.path.exists(baseline_path):
        Z_baseline = np.load(baseline_path)['Z'].astype(complex)
        print('Baseline loaded from results/baseline.npz')
    else:
        print('Baseline not found -- computing inline ...')
        Z_baseline = solve_Z(geo, FREQS)

    # ── Determine which tasks to run (fresh vs append) ────────────────────────
    existing_Z         = None
    existing_combo_idx = None
    existing_lw        = None
    existing_k_rb      = None
    existing_Rb_vals   = None

    if append_mode and os.path.exists(OUT):
        print(f'\nAppend mode: loading existing {OUT} ...')
        npz = np.load(OUT)
        existing_combo_idx = npz['combo_idx_arr'].tolist()
        existing_lw        = npz['lw_arr'].tolist()
        existing_k_rb      = [round(float(v), 4) for v in npz['k_rb_arr']]
        existing_Z         = npz['Z']
        existing_Rb_vals   = npz['Rb_vals']

        done = set(zip(existing_combo_idx, existing_lw, existing_k_rb))
        print(f'  Already computed: {len(done)} modes')

        todo_combos = []
        todo_lw     = []
        todo_k_rb   = []
        for ci in range(n_combos):
            for k in K_RB_VALS:
                for lw in lw_range:
                    if (ci, lw, round(k, 4)) not in done:
                        todo_combos.append(ci)
                        todo_lw.append(lw)
                        todo_k_rb.append(k)

        print(f'  To compute: {len(todo_combos)} modes')
        if len(todo_combos) == 0:
            print('Nothing to do. Exiting.')
            return
    else:
        if append_mode:
            print('--append specified but output file not found; running fresh.')
        todo_combos = [ci for ci in range(n_combos)
                       for k in K_RB_VALS for lw in lw_range]
        todo_lw     = [lw for ci in range(n_combos)
                       for k in K_RB_VALS for lw in lw_range]
        todo_k_rb   = [k  for ci in range(n_combos)
                       for k in K_RB_VALS for lw in lw_range]

    # ── Build flat task list ───────────────────────────────────────────────────
    tasks = [
        (i, todo_combos[i], todo_k_rb[i], todo_lw[i])
        for i in range(len(todo_combos))
    ]

    # ── Parallel solve ─────────────────────────────────────────────────────────
    print(f'\nSolving {len(tasks)} modes with N_JOBS={N_JOBS} ...')
    results = Parallel(n_jobs=N_JOBS)(
        delayed(_worker)(
            idx, CORNER_COMBOS[ci], k, lw, geo, FREQS
        )
        for idx, ci, k, lw in tqdm(tasks, desc='B_rb sweep', ncols=70)
    )

    n_new    = len(tasks)
    n_freq   = len(FREQS)
    Z_new    = np.zeros((n_new, n_freq), dtype=complex)
    for task_idx, Z in results:
        Z_new[task_idx] = Z

    # Metadata for new modes
    new_combo_idx = np.array(todo_combos, dtype=np.int8)
    new_lw        = np.array(todo_lw, dtype=int)
    new_k_rb      = np.array(todo_k_rb, dtype=float)
    new_Rb_vals   = np.maximum(
        Rb_base * (1.0 + new_k_rb * new_lw / max_lw), 1e-6
    )

    # ── Concatenate with existing (append mode) ────────────────────────────────
    if existing_Z is not None:
        combo_idx_arr = np.concatenate([existing_combo_idx, new_combo_idx])
        lw_arr        = np.concatenate([existing_lw,        new_lw])
        k_rb_arr      = np.concatenate([existing_k_rb,      new_k_rb])
        Rb_vals_arr   = np.concatenate([existing_Rb_vals,   new_Rb_vals])
        Z_lib         = np.concatenate([existing_Z,         Z_new], axis=0)
    else:
        combo_idx_arr = new_combo_idx
        lw_arr        = new_lw
        k_rb_arr      = new_k_rb
        Rb_vals_arr   = new_Rb_vals
        Z_lib         = Z_new

    # ── Save ───────────────────────────────────────────────────────────────────
    np.savez(
        OUT,
        combo_idx_arr = combo_idx_arr.astype(np.int8),
        combo_def     = _COMBO_DEF,
        lw_arr        = lw_arr.astype(int),
        k_rb_arr      = k_rb_arr.astype(float),
        Rb_vals       = Rb_vals_arr.astype(float),
        Z             = Z_lib,
        Z_baseline    = Z_baseline,
        freqs         = FREQS,
        Rb_base       = float(Rb_base),
        lw_max        = int(LW_MAX),
        k_rb_vals     = np.array(K_RB_VALS, dtype=float),
    )
    print(f'Saved -> {OUT}  ({len(Z_lib)} total modes)')

    # ── Summary at 1 kHz (grouped by combo_idx and K_RB) ─────────────────────
    i1k = np.argmin(np.abs(FREQS - 1000))
    dZ  = Z_lib[:, i1k] - Z_baseline[i1k]
    print(f'\ndZ at 1 kHz  (|Z0| = {abs(Z_baseline[i1k]):.2f} Ohm):')
    print(
        f'  {"combo":>5}  {"K_RB":>6}  {"lw":>3}  '
        f'{"Rb":>8}  {"dR [Ohm]":>12}  {"dX [Ohm]":>12}  {"pct":>9}'
    )
    last_ci = None
    last_k  = None
    for i in range(len(combo_idx_arr)):
        ci = int(combo_idx_arr[i])
        k  = float(k_rb_arr[i])
        lw = int(lw_arr[i])
        Rb = float(Rb_vals_arr[i])
        if ci != last_ci:
            print()
            last_ci = ci
            last_k  = None
        if k != last_k:
            if last_k is not None:
                print()
            last_k = k
        pct = 100 * abs(dZ[i]) / abs(Z_baseline[i1k])
        print(
            f'  {ci:5d}  {k:6.2f}  {lw:3d}  {Rb:8.3f}  '
            f'{dZ[i].real:+12.4f}  {dZ[i].imag:+12.4f}  {pct:9.5f}%'
        )


if __name__ == '__main__':
    main()
