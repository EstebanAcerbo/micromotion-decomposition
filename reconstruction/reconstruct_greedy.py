"""
reconstruct_v3.py  --  Greedy Sequential State Matching
=========================================================
At each time step, select exactly ONE mode per library (one physical state)
such that their summed dZ best approximates the experimental (dR, dX).
No coefficients, no rescaling, no basis inversion.  The residual has
physical meaning.

Algorithm per time step t
-------------------------
    y(t) = [dR_exp(t), dX_exp(t)]

    Step 1: i*(t) = argmin_i  ||dZ_B[i] - y(t)||^2        # best B_rb mode
            r1(t) = y(t) - dZ_B[i*(t)]                    # residual after B_rb

    Step 2: j*(t) = argmin_j  ||dZ_A[j] - r1(t)||^2       # best A_shift mode
            r2(t) = r1(t) - dZ_A[j*(t)]                   # residual after A_shift

    Step 3: k*(t) = argmin_k  ||dZ_C[k] - r2(t)||^2       # best C_brickwall mode
            r3(t) = r2(t) - dZ_C[k*(t)]                   # final residual

Default order: B_rb -> A_shift -> C_brickwall

Usage
-----
    python reconstruct_v3.py --stem eb11 [--savefig] [--show] [--cal]
    python reconstruct_v3.py --stem eb11 --order B_rb A_shift C_brickwall --savefig --show
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

# ── Library definitions (subset used in v3) ───────────────────────────────────
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

DEFAULT_ORDER = ['B_rb',  'D_rb', 'A_shift', 'C_brickwall']


# ── Library loading ───────────────────────────────────────────────────────────

def load_libraries(freq_target=1000.0):
    """
    Load perturbation libraries for A_shift, B_rb, C_brickwall.

    Returns
    -------
    libs   : dict  name -> dict with keys:
                 dZ      : (N_modes, 2)  [dR, dX] at freq_target
                 labels  : list[str]     mode label per mode
                 dx_px   : (N_modes,) int or None   (A_shift only)
                 dy_px   : (N_modes,) int or None   (A_shift only)
                 k_rb    : (N_modes,) float or None  (B_rb only)
                 lw      : (N_modes,) int or None    (B_rb only)
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

        # ── Library A: new flat format ────────────────────────────────────────
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

        # ── Library D_rb: 3-parameter (alpha, Rb, Cm) perturbation format ────────
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

        # ── All other libraries (old format) ──────────────────────────────────
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


# ── Greedy reconstruction ──────────────────────────────────────────────────────

def greedy_reconstruct(libs, dR, dX, order):
    """
    Apply greedy sequential state matching.

    Parameters
    ----------
    libs  : dict from load_libraries()
    dR, dX: (N_t,) experimental fluctuation signals
    order : list of library names (3 elements)

    Returns
    -------
    result : dict with selected indices, reconstructed signal, residuals, etc.
    """
    N_t = len(dR)
    y   = np.column_stack([dR, dX])   # (N_t, 2)

    residual  = y.copy()
    idx_dict  = {}
    dZ_sel    = {}

    for step, lname in enumerate(order, start=1):
        if lname not in libs:
            print(f'  WARNING: library "{lname}" not loaded — skipping step {step}.')
            idx_dict[lname]  = np.zeros(N_t, dtype=np.intp)
            dZ_sel[lname]    = np.zeros((N_t, 2), dtype=float)
            continue

        dZ_lib = libs[lname]['dZ']   # (N_modes, 2)
        abbrev = lname[:6]
        print(f'Step {step}/3  {lname:<12s} ... ', end='', flush=True)
        t0  = time.perf_counter()
        idx = nearest_mode(dZ_lib, residual)
        elapsed = time.perf_counter() - t0
        print(f'done  {elapsed:.1f} s')

        idx_dict[lname] = idx
        contrib          = dZ_lib[idx]           # (N_t, 2)
        dZ_sel[lname]   = contrib
        residual         = residual - contrib

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


# ── Physical parameter extraction ────────────────────────────────────────────

def extract_physical(libs, idx_dict, d_um):
    """
    Extract physical parameters from selected mode indices.

    Returns
    -------
    dx_um    : (N_t,) µm displacement x  (A_shift)
    dy_um    : (N_t,) µm displacement y  (A_shift)
    k_rb_seq : (N_t,) float  K_RB per step  (B_rb)
    lw_seq   : (N_t,) int    lw per step    (B_rb)
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
    print(f'\n=== Greedy reconstruction: {stem} ===')
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
        f'Greedy reconstruction — {stem}  RMSE={rmse_total:.1f} Ohm',
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
        fpath = os.path.join(RESULTS, f'reconstruct_v3_{stem}_fit.png')
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
        fpath = os.path.join(RESULTS, f'reconstruct_v3_{stem}_phasespace.png')
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
    lines1, labs1 = ax1.get_legend_handles_labels()
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
        fpath = os.path.join(RESULTS, f'reconstruct_v3_{stem}_states.png')
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
        fpath = os.path.join(RESULTS, f'reconstruct_v3_{stem}_displacement.png')
        fig.savefig(fpath, dpi=150, bbox_inches='tight')
        print(f'Saved: {fpath}')
    if show:
        plt.show()
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Greedy sequential state matching (v3)')
    parser.add_argument('--stem',    required=True,
                        help='Experiment stem (eb11, eb04, ...)')
    parser.add_argument('--savefig', action='store_true',
                        help='Save figures to results/')
    parser.add_argument('--show',    action='store_true',
                        help='Call plt.show() after each figure')
    parser.add_argument('--cal',     action='store_true',
                        help='Enable Z0 calibration (default OFF)')
    parser.add_argument('--order',   nargs='+',
                        default=DEFAULT_ORDER,
                        metavar='LIB',
                        help='Greedy matching order (library names, default: B_rb D_rb A_shift C_brickwall)')
    args = parser.parse_args()

    stem  = args.stem
    order = args.order

    print(f'\n=== Greedy reconstruction v3: {stem} ===\n')

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
        # Two-electrode correction: Z_measured = 2 × Z_per_electrode.

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
    # Pad to 3 entries so the algorithm always has something to iterate
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

    # ── Print summary header ──────────────────────────────────────────────────
    print_summary(stem, order, libs, Z0_exp, Z0_sim, cal_active, N_t, dt, None)

    # ── Greedy reconstruction ─────────────────────────────────────────────────
    res = greedy_reconstruct(libs, dR, dX, order)

    dR_rec   = res['dR_rec']
    dX_rec   = res['dX_rec']
    dR_resid = res['dR_resid']
    dX_resid = res['dX_resid']
    idx_dict = res['idx_dict']

    # ── Extract physical parameters ───────────────────────────────────────────
    dx_um, dy_um, k_rb_seq, lw_seq, da_seq, dRb_seq, dCm_seq, c_idx_seq = extract_physical(
        libs, idx_dict, d_um)

    # ── RMSE printout ─────────────────────────────────────────────────────────
    rmse_R     = res['rmse_R']
    rmse_X     = res['rmse_X']
    rmse_total = res['rmse_total']
    std_dR     = float(dR.std()) if dR.std() > 0 else 1.0
    std_dX     = float(dX.std()) if dX.std() > 0 else 1.0

    print(f'\nRMSE dR   = {rmse_R:.2f} Ohm  ({100*rmse_R/std_dR:.1f}% of std)')
    print(f'RMSE dX   = {rmse_X:.2f} Ohm  ({100*rmse_X/std_dX:.1f}% of std)')
    print(f'RMSE |dZ| = {rmse_total:.2f} Ohm')

    # ── Save results ──────────────────────────────────────────────────────────
    idx_A = idx_dict.get('A_shift',     np.zeros(N_t, dtype=np.intp))
    idx_B = idx_dict.get('B_rb',        np.zeros(N_t, dtype=np.intp))
    idx_C = idx_dict.get('C_brickwall', np.zeros(N_t, dtype=np.intp))
    idx_D = idx_dict.get('D_rb',        np.zeros(N_t, dtype=np.intp))

    out_path = os.path.join(RESULTS, f'reconstruct_v3_{stem}.npz')
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
        rmse_R     = np.array(rmse_R),
        rmse_X     = np.array(rmse_X),
        rmse_total = np.array(rmse_total),
        order      = np.array(','.join(order)),
    )
    print(f'\nData saved: {out_path}')

    # ── Figures ───────────────────────────────────────────────────────────────
    plot_fit(t, dR, dX, dR_rec, dX_rec, dR_resid, dX_resid,
             stem, res, args.savefig, args.show)

    plot_phasespace(t, dR, dX, dR_rec, dX_rec, libs, order,
                    stem, args.savefig, args.show)

    plot_states(t, dx_um, dy_um, k_rb_seq, lw_seq,
                da_seq, dRb_seq, dCm_seq,
                c_idx_seq, stem, args.savefig, args.show)

    plot_displacement(t, dx_um, dy_um, stem, args.savefig, args.show)

    print(f'\nRun: python reconstruct_v3.py --stem {stem} --savefig --show')


if __name__ == '__main__':
    main()
