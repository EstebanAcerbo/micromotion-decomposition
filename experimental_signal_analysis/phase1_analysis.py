"""
phase1_analysis.py  --  Phase 1: Experimental Micromotion Characterisation
===========================================================================
Loads all available .dat experiments, applies group-adaptive detrending
(linear for stable experiments, moving-average window for healing/drug),
and produces four figure sets:

  Fig 1: R(t) raw + trend overlay  + detrended dR(t) — per experiment
  Fig 2: PSD of dR(t) — per experiment, grouped by type
  Fig 3: std_R vs detrend window W — window sensitivity per experiment
         Shows where std plateaus (fast micromotion) vs keeps falling
         (slow drift contaminates the estimate)
  Fig 4: FD comparison — recomputed from moving-average detrended signal
         vs stored NPZ values (old linear analysis)
  Fig 5: Summary std_R bar chart vs simulation library sensitivities

Adaptive detrending per group (GROUP_DETREND in load_experimental.py):
  blank / confluent: linear
  seeding:  5-min MA window
  healing:  5-min MA window

Run from Micromotion_Simul/:
    PYTHONIOENCODING=utf-8 python phase1_analysis.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams

rcParams.update({'font.size': 10, 'font.family': 'sans-serif'})

from load_experimental import (
    load_all, load_dat, EXPERIMENT_MAP, GROUP_ORDER,
    micromotion_stats, window_sensitivity, GROUP_DETREND,
    _detrend_moving_avg, _detrend_linear,
)

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
os.makedirs(RESULTS, exist_ok=True)

# Path to FD NPZ files (co-located with this script)
FD_NPZ_DIR = os.path.dirname(os.path.abspath(__file__))

GROUP_COLORS = {
    'blank':     '#888888',
    'seeding':   '#4CAF50',
    'confluent': '#2196F3',
    'healing':   '#FF9800',
}

# Window sweep for sensitivity analysis [seconds]
WINDOW_SWEEP = [15, 30, 60, 120, 300, 600, 900, 1800, 'linear']


# ── FD utilities ──────────────────────────────────────────────────────────────

def _fd_slope_from_npz(stem):
    """Load stored FD data from FDeb*_sc.npz; return dict of slopes."""
    fname = os.path.join(FD_NPZ_DIR, 'FDeb%s_sc.npz' % stem.replace('eb', ''))
    if not os.path.exists(fname):
        return None
    d = np.load(fname, allow_pickle=True)

    def slope(xk, yk):
        x = d.get(xk); y = d.get(yk)
        if x is None or y is None:
            return float('nan')
        mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 3:
            return float('nan')
        return float(np.polyfit(np.log10(x[mask]), np.log10(y[mask]), 1)[0])

    return dict(
        FD_R_HI  = slope('X_R_HI',  'Y_R_HI'),
        FD_C_HI  = slope('X_C_HI',  'Y_C_HI'),
        FD_R_MRL = slope('X_R_MRL', 'Y_R_MRL'),
        FD_C_MRL = slope('X_C_MRL', 'Y_C_MRL'),
        H_RS1_R  = float(d['Y_R_RS1'].mean()) if 'Y_R_RS1' in d else float('nan'),
    )


def higuchi_fd(sig, k_max=100):
    """Fast Higuchi fractal dimension estimate. Returns slope (FD = -slope)."""
    N = len(sig)
    k_arr = np.arange(1, min(k_max, N // 4) + 1)
    L_mean = np.zeros(len(k_arr))
    for ki, k in enumerate(k_arr):
        L = 0.0
        for m in range(1, k + 1):
            seq = sig[m - 1::k]
            a   = len(seq) - 1
            if a < 1:
                continue
            L += np.sum(np.abs(np.diff(seq))) * (N - 1) / (k * k * a)
        L_mean[ki] = L / k
    mask = L_mean > 0
    if mask.sum() < 3:
        return float('nan')
    slope = np.polyfit(np.log10(k_arr[mask]), np.log10(L_mean[mask]), 1)[0]
    return float(slope)


def _load_sim_sensitivity():
    libs = {}
    lib_files = {
        'A_shift':     'library_A_shift.npz',
        'B_pivot_old': 'library_B_pivot.npz',
        'B_det':       'library_B_det.npz',
        'B_rand':      'library_B_rand.npz',
        'C_brickwall': 'library_C_brickwall.npz',
    }
    for name, fname in lib_files.items():
        fpath = os.path.join(RESULTS, fname)
        if not os.path.exists(fpath):
            libs[name] = None
            continue
        d     = np.load(fpath, allow_pickle=True)
        Z     = d['Z'].astype(complex)
        Z_bl  = d['Z_baseline'].astype(complex)
        freqs = d['freqs']
        i1k   = np.argmin(np.abs(freqs - 1000))
        dZ    = (Z - Z_bl[i1k]).reshape(-1, len(freqs))[:, i1k]
        pcts  = 100 * np.abs(dZ) / np.abs(Z_bl[i1k])
        libs[name] = dict(
            mean_pct = pcts.mean(),
            max_pct  = pcts.max(),
            Z0       = abs(Z_bl[i1k]),
            mean_Ohm = pcts.mean() / 100.0 * abs(Z_bl[i1k]),
        )
    return libs


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    dat_dir = os.path.dirname(os.path.abspath(__file__))

    print('Loading experiments with group-adaptive detrend ...')
    import sys
    _stems_arg = [a for a in sys.argv[1:] if not a.startswith('-')]
    _stems_filter = _stems_arg if _stems_arg else None
    data = load_all(dat_dir=dat_dir, stems=_stems_filter, verbose=True)
    if not data:
        print('No .dat files found. Check path.')
        return

    # Ordered stem list
    groups_by_stem = {}
    for stem, d in data.items():
        g = d['meta'].get('group', 'unknown')
        groups_by_stem.setdefault(g, []).append(stem)
    ordered_stems = []
    for g in GROUP_ORDER:
        for stem in sorted(groups_by_stem.get(g, [])):
            ordered_stems.append(stem)

    # ── Table ─────────────────────────────────────────────────────────────────
    print('\n' + '='*110)
    hdr = ('%-12s %-12s %6s %7s %7s %9s %12s %12s %10s %10s' %
           ('Experiment', 'Group', 'f[Hz]', 'N', 'dur[h]',
            '<R>[Ohm]', 'std_R[Ohm]', 'std_X[Ohm]', 'std_R/R[%]', 'Detrend'))
    print(hdr)
    print('='*110)
    for stem in ordered_stems:
        d = data[stem]; s = d['stats']; m = d['meta']
        rel  = 100 * s['std_R'] / abs(s['mean_R']) if s['mean_R'] != 0 else 0
        w    = s.get('window_sec_used')
        dlbl = ('linear' if s['detrend_mode'] == 'linear' else 'MA%ds' % int(w))
        print('%-12s %-12s %6.0f %7d %7.2f %9.1f %12.4f %12.4f %10.5f %10s' %
              (stem, m.get('group','?'), d['f_meas'], s['n_pts'],
               s['duration_h'], s['mean_R'], s['std_R'], s['std_X'], rel, dlbl))
    print('='*110)

    # ── Simulation sensitivities ───────────────────────────────────────────────
    print('\nSimulation sensitivities at 1 kHz:')
    sim = _load_sim_sensitivity()
    for name, v in sim.items():
        if v:
            print('  %-20s mean=%7.4f%%  max=%7.4f%%  |dZ|_mean=%7.2f Ohm (sim scale)'
                  % (name, v['mean_pct'], v['max_pct'], v['mean_Ohm']))
        else:
            print('  %-20s NOT FOUND' % name)

    # ── Figure 1: R(t) raw + trend + dR ──────────────────────────────────────
    n = len(ordered_stems)
    fig1, axes = plt.subplots(n, 2, figsize=(14, 2.6 * n), constrained_layout=True)
    if n == 1:
        axes = axes[np.newaxis, :]

    for row, stem in enumerate(ordered_stems):
        d  = data[stem]; s = d['stats']; m = d['meta']
        col = GROUP_COLORS.get(m.get('group', ''), '#333333')
        t_h = s['t'] / 3600.0

        # Left: R(t) + trend overlay
        ax = axes[row, 0]
        ax.plot(t_h, d['R'], color=col, lw=0.5, alpha=0.8, label='R(t)')
        trend = d['R'] - s['dR']  # reconstruct trend = R - dR
        ax.plot(t_h, trend, 'k--', lw=1.2, alpha=0.7, label='trend')
        w = s.get('window_sec_used')
        dlbl = 'linear' if s['detrend_mode'] == 'linear' else 'MA%ds' % int(w)
        ax.set_title('%s [%s] — %s' % (stem, m.get('label', '?'), dlbl),
                     fontsize=8, loc='left')
        ax.set_ylabel('R [Ohm]', fontsize=8)
        ax.legend(fontsize=7, loc='upper right')
        ax.tick_params(labelsize=7)

        # Right: dR(t)
        ax2 = axes[row, 1]
        ax2.plot(t_h, s['dR'], color=col, lw=0.4, alpha=0.8)
        ax2.axhline(0, color='k', lw=0.5, ls=':')
        ax2.set_title('std_R=%.2f Ohm (%.4f%%)' %
                      (s['std_R'], 100 * s['std_R'] / abs(s['mean_R'])),
                      fontsize=8, loc='left')
        ax2.set_ylabel('dR [Ohm]', fontsize=8)
        ax2.tick_params(labelsize=7)

    axes[-1, 0].set_xlabel('Time [h]')
    axes[-1, 1].set_xlabel('Time [h]')
    fig1.suptitle('R(t) with adaptive detrend and residual dR(t)', fontsize=11)
    fig1.savefig(os.path.join(RESULTS, 'phase1_timeseries.png'), dpi=150)
    print('\nFig1 saved: phase1_timeseries.png')

    # ── Figure 2: PSD panel ───────────────────────────────────────────────────
    fig2, axes2 = plt.subplots(n, 1, figsize=(10, 2.4 * n),
                               sharex=False, constrained_layout=True)
    if n == 1:
        axes2 = [axes2]
    for row, stem in enumerate(ordered_stems):
        d = data[stem]; s = d['stats']; m = d['meta']
        col = GROUP_COLORS.get(m.get('group', ''), '#333333')
        ax  = axes2[row]
        fp  = s['freqs_psd'][1:]
        ax.loglog(fp, s['psd_R'][1:], color=col, lw=1.0, label=stem)
        if row == 0:
            ax.set_ylabel('PSD [(Ohm)^2/Hz]', fontsize=8)
        ax.set_title('%s' % stem, fontsize=7, loc='right')
        ax.tick_params(labelsize=7)
        ax.grid(True, which='both', alpha=0.2)
    axes2[-1].set_xlabel('Frequency [Hz]')
    fig2.suptitle('PSD of dR (adaptive detrend)', fontsize=11)
    fig2.savefig(os.path.join(RESULTS, 'phase1_psd.png'), dpi=150)
    print('Fig2 saved: phase1_psd.png')

    # ── Figure 3: Window sensitivity ──────────────────────────────────────────
    print('\nComputing window sensitivity (may take ~1 min) ...')
    fig3, ax3 = plt.subplots(figsize=(10, 5))

    win_labels = []
    win_x      = []
    for w in WINDOW_SWEEP:
        if w == 'linear':
            win_x.append(1e5)     # sentinel x-position for linear
            win_labels.append('linear')
        else:
            win_x.append(float(w) / 60.0)
            win_labels.append('%.0f' % (float(w) / 60.0))

    for stem in ordered_stems:
        d   = data[stem]
        m   = d['meta']
        col = GROUP_COLORS.get(m.get('group', ''), '#333333')
        ws  = window_sensitivity(d['R'], d['t'], windows_sec=WINDOW_SWEEP)

        x_plot = [v / 60.0 if isinstance(v, float) else 120.0 * 60.0
                  for v in ws['windows']]
        # Normalise by mean_R to show relative %
        stds_pct = 100.0 * ws['stds'] / abs(d['stats']['mean_R'])

        ax3.semilogx(x_plot, stds_pct, 'o-', color=col,
                     lw=1.3, ms=4, alpha=0.8,
                     label='%s (%s)' % (stem, m.get('group', '?')))

    ax3.axvline(5, color='k', ls=':', lw=0.8, alpha=0.6, label='5 min')
    ax3.axvline(15, color='k', ls='--', lw=0.8, alpha=0.6, label='15 min')
    ax3.set_xlabel('Moving-average window [min]  (rightmost = linear detrend)')
    ax3.set_ylabel('std_R / <R>  [%]')
    ax3.set_title('Window sensitivity: std(dR) vs detrend window size')
    ax3.legend(fontsize=7, ncol=2, loc='upper right')
    ax3.grid(True, which='both', alpha=0.25)
    fig3.tight_layout()
    fig3.savefig(os.path.join(RESULTS, 'phase1_window_sensitivity.png'), dpi=150)
    print('Fig3 saved: phase1_window_sensitivity.png')

    # ── Figure 4: FD comparison — stored NPZ vs recomputed ──────────────────
    print('Computing Higuchi FD from MA-detrended signals ...')
    fd_results = {}
    for stem in ordered_stems:
        d   = data[stem]
        m   = d['meta']
        # Recompute FD from properly detrended dR
        dR = d['stats']['dR']
        # Downsample to max 10000 pts for speed
        step = max(1, len(dR) // 10000)
        fd_new = higuchi_fd(dR[::step], k_max=100)
        # Load stored NPZ FD
        fd_npz = _fd_slope_from_npz(stem)
        fd_results[stem] = dict(
            fd_new  = fd_new,
            fd_old  = fd_npz.get('FD_R_HI', float('nan')) if fd_npz else float('nan'),
            fd_mrl  = fd_npz.get('FD_R_MRL', float('nan')) if fd_npz else float('nan'),
            group   = m.get('group', '?'),
            label   = m.get('label', stem),
        )
        print('  %s: FD_new(HI)=%+.4f  FD_stored(HI)=%+.4f  FD_stored(MRL)=%+.4f'
              % (stem, fd_new,
                 fd_results[stem]['fd_old'],
                 fd_results[stem]['fd_mrl']))

    # Plot: stored HI vs recomputed HI
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    x_pos = np.arange(len(ordered_stems))
    width = 0.28

    fd_old_vals = [fd_results[s]['fd_old'] for s in ordered_stems]
    fd_new_vals = [fd_results[s]['fd_new'] for s in ordered_stems]
    fd_mrl_vals = [fd_results[s]['fd_mrl'] for s in ordered_stems]

    bars1 = ax4.bar(x_pos - width, fd_old_vals, width, label='Stored HI (linear detrend)',
                    color='steelblue', alpha=0.8)
    bars2 = ax4.bar(x_pos,          fd_new_vals, width, label='New HI (adaptive detrend)',
                    color='darkorange', alpha=0.8)
    bars3 = ax4.bar(x_pos + width,  fd_mrl_vals, width, label='Stored MRL (linear detrend)',
                    color='seagreen', alpha=0.8)

    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(ordered_stems, rotation=45, ha='right', fontsize=8)
    ax4.set_ylabel('FD slope  (FD = −slope for HI;  FD = slope+2 for MRL)')
    ax4.set_title('Fractal dimension slopes: stored (linear) vs recomputed (adaptive detrend)')
    ax4.axhline(-2.0, color='grey', ls=':', lw=0.8, alpha=0.6, label='Slope −2 (white noise)')
    ax4.axhline(-1.5, color='grey', ls='--', lw=0.8, alpha=0.6, label='Slope −1.5 (Brownian)')
    ax4.legend(fontsize=8)
    ax4.grid(axis='y', alpha=0.25)
    fig4.tight_layout()
    fig4.savefig(os.path.join(RESULTS, 'phase1_fd_comparison.png'), dpi=150)
    print('Fig4 saved: phase1_fd_comparison.png')

    # ── Figure 5: Summary std_R vs simulation ─────────────────────────────────
    fig5, ax5 = plt.subplots(figsize=(11, 5))
    x = 0
    x_ticks = []; x_labels = []
    prev_g = None
    for stem in ordered_stems:
        d = data[stem]; s = d['stats']; m = d['meta']
        g = m.get('group', '')
        if g != prev_g and prev_g is not None:
            x += 0.4
        prev_g = g
        col = GROUP_COLORS.get(g, '#333')
        ax5.bar(x, s['std_R'], width=0.7, color=col, alpha=0.8)
        x_ticks.append(x); x_labels.append(stem)
        x += 1.0

    SIM_LINES = [
        ('A_shift',     'steelblue',  '--', 'Lib A shift (sim, mean)'),
        ('B_pivot_old', 'darkorange', '--', 'Lib B pivot-old (sim, mean)'),
        ('B_det',       'firebrick',  '--', 'Lib B det (sim, mean)'),
        ('C_brickwall', 'purple',     '--', 'Lib C brick (sim, mean)'),
    ]
    for lname, c, ls, lab in SIM_LINES:
        v = sim.get(lname)
        if v:
            ax5.axhline(v['mean_Ohm'], color=c, ls=ls, lw=1.4, alpha=0.7, label=lab)

    ax5.set_xticks(x_ticks)
    ax5.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
    ax5.set_ylabel('std(dR) [Ohm]  (adaptive detrend)')
    ax5.set_title('Micromotion amplitude vs simulation sensitivities (1 kHz, sim scale)')
    ax5.legend(fontsize=8, ncol=2)
    ax5.grid(axis='y', alpha=0.25)

    # Add group colour legend
    from matplotlib.patches import Patch
    patches = [Patch(color=c, label=g) for g, c in GROUP_COLORS.items()]
    ax5.legend(handles=patches +
               [plt.Line2D([0], [0], color=c, ls=ls, lw=1.4, label=lab)
                for _, c, ls, lab in SIM_LINES],
               fontsize=8, ncol=3, loc='upper left')
    fig5.tight_layout()
    fig5.savefig(os.path.join(RESULTS, 'phase1_std_comparison.png'), dpi=150)
    print('Fig5 saved: phase1_std_comparison.png')

    # ── Save npz ──────────────────────────────────────────────────────────────
    npz_out = {}
    for stem, d in data.items():
        s = d['stats']
        for k in ('dR','dX','t','psd_R','psd_X','freqs_psd'):
            npz_out['%s_%s' % (stem, k)] = s[k]
        for k in ('mean_R','std_R','mean_X','std_X'):
            npz_out['%s_%s' % (stem, k)] = np.array([s[k]])
        npz_out['%s_fmeas' % stem] = np.array([d['f_meas']])
        if stem in fd_results:
            npz_out['%s_fd_new' % stem] = np.array([fd_results[stem]['fd_new']])

    np.savez(os.path.join(RESULTS, 'phase1_experimental.npz'), **npz_out)
    print('Data saved: phase1_experimental.npz')

    plt.show()
    print('\nPhase 1 complete.')


if __name__ == '__main__':
    main()
