"""
run_libraries.py  --  Master script: run all simulation libraries in sequence
==============================================================================
Runs the full pipeline with the parameters currently set in engine.py:

    1. baseline.py          -- Z_baseline(f), saved to results/baseline.npz
    2. shift_library.py     -- Library A (lower-triangle sweep), results/library_A_shift.npz
    3. library_B_det.py     -- Library B deterministic, results/library_B_det.npz
    4. library_B_rand.py    -- Library B random,        results/library_B_rand.npz
    5. library_B_micro.py   -- Library B micro (lw={0,1,2}), results/library_B_micro.npz
    6. library_B_rb.py      -- Library B Rb-scaled (K_RB sweep [-0.5..5.0]), results/library_B_rb.npz
    7. library_D_rb.py      -- Library D pure Rb sweep (no geometry), results/library_D_rb.npz
    8. shift2_library.py    -- Library C brickwall,     results/library_C_brickwall.npz

Each script is launched as a subprocess to guarantee clean process isolation
and correct MKL thread-count setup.

Usage
-----
    python run_libraries.py                  # run all 8 steps
    python run_libraries.py --skip baseline  # skip baseline if already done
    python run_libraries.py --only B_micro   # run only B_micro library

Log file
--------
A timestamped log is written to results/run_log_<timestamp>.txt recording
the engine.py parameter snapshot and per-step timings.

Parameter snapshot
------------------
The current engine.py P dict is printed at startup so it is unambiguous which
parameters were used for this run.
"""

import os
import sys
import argparse
import subprocess
import time
from datetime import datetime

HERE    = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, '..', 'results')
os.makedirs(RESULTS, exist_ok=True)

# ── Step definitions ──────────────────────────────────────────────────────────
STEPS = [
    ('baseline',   'baseline.py'),
    ('A_shift',    'shift_library.py'),
    #('B_det',      'library_B_det.py'),
    #('B_rand',     'library_B_rand.py'),
    #('B_micro',    'library_B_micro.py'),
    ('B_rb',       'library_B_rb.py'),
    ('D_rb',       'library_D_rb.py'),
    ('C_brickwall','shift2_library.py'),
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_params():
    """Return engine.P dict as a formatted string for logging."""
    sys.path.insert(0, HERE)
    import importlib
    import engine as eng
    importlib.reload(eng)          # picks up any edits since process start
    lines = ['engine.P snapshot:']
    for k, v in eng.P.items():
        lines.append(f'  {k:12s} = {v}')
    return '\n'.join(lines)


def run_step(name, script, log_fh):
    script_path = os.path.join(HERE, script)
    print(f'\n{"="*60}')
    print(f'  STEP: {name}  ({script})')
    print(f'{"="*60}')
    log_fh.write(f'\n--- STEP: {name} ({script}) ---\n')

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, script_path],
        cwd=HERE,
        # stdout/stderr stream to console AND captured for log
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        errors='replace',
    )
    elapsed = time.time() - t0

    # print and log combined output
    print(result.stdout, end='')
    log_fh.write(result.stdout)

    status = 'OK' if result.returncode == 0 else f'FAILED (exit {result.returncode})'
    summary = f'  -> {status}  |  elapsed: {elapsed:.1f} s\n'
    print(summary)
    log_fh.write(summary)
    log_fh.flush()

    return result.returncode == 0


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='Run all simulation libraries.')
    parser.add_argument(
        '--skip', nargs='+', metavar='NAME',
        help='Step names to skip (e.g. --skip baseline)',
        default=[],
    )
    parser.add_argument(
        '--only', nargs='+', metavar='NAME',
        help='Run only these step names (e.g. --only B_det B_rand)',
        default=[],
    )
    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # resolve step list
    if args.only:
        steps = [(n, s) for n, s in STEPS if n in args.only]
        unknown = set(args.only) - {n for n, _ in STEPS}
    elif args.skip:
        steps = [(n, s) for n, s in STEPS if n not in args.skip]
        unknown = set(args.skip) - {n for n, _ in STEPS}
    else:
        steps  = STEPS
        unknown = set()

    if unknown:
        print(f'WARNING: unknown step names ignored: {unknown}')
        print(f'Valid names: {[n for n,_ in STEPS]}')

    if not steps:
        print('No steps to run. Exiting.')
        return

    # parameter snapshot
    params_str = load_params()
    print(params_str)

    # log file
    ts      = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(RESULTS, f'run_log_{ts}.txt')

    t_total = time.time()
    failed  = []

    with open(log_path, 'w', encoding='utf-8') as log_fh:
        log_fh.write(f'run_libraries.py  started {datetime.now().isoformat()}\n')
        log_fh.write(params_str + '\n')
        log_fh.write(f'\nSteps to run: {[n for n,_ in steps]}\n')

        for name, script in steps:
            ok = run_step(name, script, log_fh)
            if not ok:
                failed.append(name)

        elapsed_total = time.time() - t_total
        summary = (
            f'\n{"="*60}\n'
            f'  DONE  |  total: {elapsed_total:.1f} s\n'
            f'  Failed steps: {failed if failed else "none"}\n'
            f'{"="*60}\n'
        )
        print(summary)
        log_fh.write(summary)

    print(f'Log saved -> {log_path}')
    if failed:
        sys.exit(1)


if __name__ == '__main__':
    main()
