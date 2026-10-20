"""
load_experimental.py  --  Experimental .dat File Loader
=========================================================
Parses the custom ECIS .dat format from the Bariloche SR530 setup and
returns calibrated impedance time series at 1 kHz.

File structure (from RCT_Calculate.py, Comparacion/Datos/):
  - Lines 1-22:  text header
  - Line 23:     FLAG1
  - Line 24:     N_ref (number of reference-channel rows)
  - Lines 25-30: header scalars (time constant, meas/ch, n_ch, Rs [Ohm],
                 stabilization time, n_columns)
  - Line 31:     blank
  - Lines 32-35: column label comment + blank + FLAG2 + blank
  - Lines 37 .. 37+N_ref-1 : reference-channel rows [ch, t, f, V, phi]
  - Lines 37+N_ref .. end  : cell-channel rows       [ch, t, f, V, phi]

Columns (0-indexed):
  0: channel index
  1: time [s] since measurement start
  2: frequency [Hz]
  3: amplitude [V]
  4: phase [deg]

The function computes:
  Z(t) = Rs * V_cell / (V_ref_avg - V_cell)   (phasor form)
  R(t) = Re(Z)      [Ohm]
  C(t) = -1 / (2*pi*f * Im(Z))     [F]
  X(t) = Im(Z)      [Ohm]   (signed reactance)

Usage
-----
    from load_experimental import load_dat, micromotion_stats, EXPERIMENT_MAP

    t, R, X, C = load_dat('eb11_sc.dat')
    stats = micromotion_stats(R, X, t)
"""

import os
import numpy as np
from scipy.signal import welch

# Default location of the .dat files (co-located with this script)
_DEFAULT_DAT_DIR = os.path.dirname(os.path.abspath(__file__))

# Load resistance [Ohm] — hard-coded in header but fixed across experiments
RS_DEFAULT = 99715.0

# ── Experiment catalogue ──────────────────────────────────────────────────────
#  key  : filename stem (without _sc.dat)
#  value: dict with label, group, frequency string, notes

EXPERIMENT_MAP = {
    'eb18': dict(label='Blank (no cells) LF-1',    group='blank',     freq='LF'),
    'eb19': dict(label='Blank (no cells) LF-2',    group='blank',     freq='LF'),
    'eb20': dict(label='Blank (no cells) HF',      group='blank',     freq='HF'),
    'eb04': dict(label='Seeding LF',               group='seeding',   freq='LF'),
    'eb06': dict(label='Confluence HF',            group='confluent', freq='HF'),
    'eb11': dict(label='Confluence LF',            group='confluent', freq='LF'),
    'eb13': dict(label='Wound healing LF',         group='healing',   freq='LF'),
    'eb15': dict(label='Post-healing (1 day) LF',  group='confluent', freq='LF'),
}

GROUP_ORDER = ['blank', 'seeding', 'confluent', 'healing']


# ── Core loader ───────────────────────────────────────────────────────────────

def load_dat(stem, dat_dir=None, Rs=RS_DEFAULT):
    """
    Load and calibrate one ECIS .dat file.

    Parameters
    ----------
    stem    : str — file stem, e.g. 'eb11' or 'eb13_bis'.
              The function tries 'eb11_sc.dat' and 'eb11.dat' variants.
    dat_dir : str or None — directory containing .dat files.
              Defaults to the directory containing this script.
    Rs      : float — series (load) resistance [Ohm].

    Returns
    -------
    t : ndarray (N,)  — time [s] from file (monotonically increasing)
    R : ndarray (N,)  — resistance = Re(Z) [Ohm]
    X : ndarray (N,)  — reactance = Im(Z) [Ohm]
    C : ndarray (N,)  — capacitance = -1/(2*pi*f*X) [F]
    f_meas : float    — measurement frequency [Hz]
    """
    if dat_dir is None:
        dat_dir = _DEFAULT_DAT_DIR

    # Try both naming conventions
    for fname in (f'{stem}_sc.dat', f'{stem}.dat', f'{stem}_sc.DAT'):
        fpath = os.path.join(dat_dir, fname)
        if os.path.exists(fpath):
            break
    else:
        raise FileNotFoundError(
            f'Cannot find {stem}_sc.dat or {stem}.dat in {dat_dir}'
        )

    with open(fpath, 'r', encoding='latin-1') as fh:
        lines = fh.readlines()

    # --- Parse header ---
    # Line 24 (0-indexed: 23): N_ref
    N_ref = int(lines[23].strip())

    # --- Read reference rows ---
    # skiprows=37 matches RCT_Calculate.py convention: 0-indexed skip of 37 lines
    # leaves the cursor at 0-indexed line 37 = 1-indexed line 38 = first data row.
    ref_data = np.loadtxt(fpath, skiprows=37, max_rows=N_ref,
                          encoding='latin-1')
    if ref_data.ndim == 1:
        ref_data = ref_data[np.newaxis, :]

    # Phasor mean of reference channel
    V_ref_phasor = ref_data[:, 3] * np.exp(1j * ref_data[:, 4] * np.pi / 180.0)
    V_ref_mean   = V_ref_phasor.mean()

    # --- Read cell rows ---
    # Cell data immediately follows the N_ref reference rows at 0-indexed line 37+N_ref.
    cell_data = np.loadtxt(fpath, skiprows=37 + N_ref, encoding='latin-1')
    if cell_data.ndim == 1:
        cell_data = cell_data[np.newaxis, :]

    t      = cell_data[:, 1]
    f_meas = float(cell_data[0, 2])

    V_cell = cell_data[:, 3] * np.exp(1j * cell_data[:, 4] * np.pi / 180.0)
    Z      = Rs * V_cell / (V_ref_mean - V_cell)

    R = Z.real
    X = Z.imag
    C = -1.0 / (2.0 * np.pi * f_meas * X)

    return t, R, X, C, f_meas


# ── Detrending utilities ──────────────────────────────────────────────────────

def _detrend_linear(sig, t):
    """Remove best-fit straight line from sig(t)."""
    t_n = (t - t.mean()) / (t[-1] - t[0] + 1e-12)
    coeffs = np.polyfit(t_n, sig, 1)
    return sig - np.polyval(coeffs, t_n)


def _detrend_moving_avg(sig, dt, window_sec):
    """
    Remove local moving average (centred uniform filter) from sig.

    Window is rounded to an odd number of samples.  Edge pixels use
    nearest-neighbour extension to avoid ringing artefacts.
    """
    from scipy.ndimage import uniform_filter1d
    n_win = max(3, int(round(window_sec / dt)))
    if n_win % 2 == 0:
        n_win += 1
    trend = uniform_filter1d(sig.astype(float), size=n_win, mode='nearest')
    return sig - trend


# ── Statistical characterisation ──────────────────────────────────────────────

# Per-group recommended detrend window (seconds).
# 'linear' → global linear fit.  float → moving-average window in seconds.
GROUP_DETREND = {
    'blank':     'linear',   # short recordings, stable
    'seeding':   300,        # 5-min window: non-confluent, active attachment dynamics
    'confluent': 900,        # 15-min window: stable, small drift
    'healing':   300,        # 5-min window: sigmoid recovery over hours
}


def micromotion_stats(R, X, t,
                      detrend='auto',
                      window_sec=None,
                      group=None,
                      nperseg_sec=300.0):
    """
    Compute micromotion statistics from R(t) and X(t) time series.

    Parameters
    ----------
    R, X        : ndarray (N,) — resistance and reactance [Ohm]
    t           : ndarray (N,) — time [s]
    detrend     : 'auto' | 'linear' | 'moving_avg'
                  'auto' uses GROUP_DETREND[group] if group is given,
                  otherwise 'linear'.
    window_sec  : float or None — moving-average window [s].
                  Only used when detrend='moving_avg' (or 'auto' resolves to it).
                  If None, uses GROUP_DETREND value or 900 s.
    group       : str or None — experiment group key (for 'auto' mode).
    nperseg_sec : float — Welch segment length [s].

    Returns
    -------
    dict with keys:
        mean_R, mean_X, std_R, std_X [Ohm]
        dR, dX           : detrended fluctuation arrays
        t                : time array
        freqs_psd        : PSD frequency axis [Hz]
        psd_R, psd_X     : Welch PSD [(Ohm)^2 / Hz]
        dt               : mean sample interval [s]
        n_pts, duration_h
        detrend_mode     : str ('linear' or 'moving_avg')
        window_sec_used  : float or None
    """
    dt = float(np.median(np.diff(t)))

    # --- Resolve detrend mode ---
    if detrend == 'auto':
        spec = GROUP_DETREND.get(group, 'linear')
        if spec == 'linear':
            mode = 'linear'
            w    = None
        else:
            mode = 'moving_avg'
            w    = float(spec) if window_sec is None else float(window_sec)
    elif detrend == 'linear':
        mode = 'linear'
        w    = None
    else:  # 'moving_avg'
        mode = 'moving_avg'
        w    = float(window_sec) if window_sec is not None else 900.0

    if mode == 'linear':
        dR = _detrend_linear(R, t)
        dX = _detrend_linear(X, t)
    else:
        dR = _detrend_moving_avg(R, dt, w)
        dX = _detrend_moving_avg(X, dt, w)

    # Welch PSD
    fs      = 1.0 / dt
    nperseg = max(256, int(nperseg_sec / dt))
    nperseg = min(nperseg, len(dR))
    fpsd, psd_R = welch(dR, fs=fs, nperseg=nperseg)
    _,    psd_X = welch(dX, fs=fs, nperseg=nperseg)

    return dict(
        mean_R         = float(R.mean()),
        mean_X         = float(X.mean()),
        std_R          = float(dR.std()),
        std_X          = float(dX.std()),
        dR             = dR,
        dX             = dX,
        t              = t,
        freqs_psd      = fpsd,
        psd_R          = psd_R,
        psd_X          = psd_X,
        dt             = dt,
        n_pts          = len(R),
        duration_h     = (t[-1] - t[0]) / 3600.0,
        detrend_mode   = mode,
        window_sec_used= w,
    )


def window_sensitivity(R, t, windows_sec=None):
    """
    Compute std(dR) as a function of moving-average window length.

    This reveals whether a given experiment has a slow biological drift
    (std decreases as window shortens) vs pure fast micromotion
    (std plateaus at short windows).

    Parameters
    ----------
    R           : ndarray (N,) — resistance [Ohm]
    t           : ndarray (N,) — time [s]
    windows_sec : list of float — window lengths to sweep [s]
                  Default: [30, 60, 120, 300, 600, 900, 1800, 3600, 'linear']

    Returns
    -------
    dict: windows -> list (matching windows_sec),
          stds    -> ndarray (N_windows,)
    """
    dt = float(np.median(np.diff(t)))
    if windows_sec is None:
        windows_sec = [30, 60, 120, 300, 600, 900, 1800, 3600, 'linear']

    stds = []
    labels = []
    for w in windows_sec:
        if w == 'linear':
            dR = _detrend_linear(R, t)
            labels.append('linear')
        else:
            dR = _detrend_moving_avg(R, dt, float(w))
            labels.append(float(w))
        stds.append(float(dR.std()))

    return dict(windows=labels, stds=np.array(stds))


# ── Convenience: load all available experiments ───────────────────────────────

def load_all(dat_dir=None, stems=None, verbose=True):
    """
    Load all experiments listed in EXPERIMENT_MAP (or a subset via `stems`).

    Returns
    -------
    dict: stem -> dict with keys: meta, t, R, X, C, f_meas, stats
    """
    if stems is None:
        stems = list(EXPERIMENT_MAP.keys())

    out = {}
    for stem in stems:
        try:
            t, R, X, C, f_meas = load_dat(stem, dat_dir=dat_dir)
            meta  = EXPERIMENT_MAP.get(stem, {})
            group = meta.get('group', None)
            stats = micromotion_stats(R, X, t, detrend='auto', group=group)
            out[stem] = dict(
                meta   = meta,
                t=t, R=R, X=X, C=C,
                f_meas = f_meas,
                stats  = stats,
            )
            if verbose:
                m = out[stem]['meta']
                w = stats.get('window_sec_used')
                detrend_lbl = ('linear' if stats['detrend_mode'] == 'linear'
                               else 'MA%ds' % int(w))
                print(
                    f'  {stem:10s} [{m.get("group","?"):10s}] '
                    f'{stats["n_pts"]:6d} pts  '
                    f'{stats["duration_h"]:.2f} h  '
                    f'dt={stats["dt"]:.3f} s  '
                    f'f={f_meas:.0f} Hz  '
                    f'R={stats["mean_R"]:.0f} Ohm  '
                    f'std_R={stats["std_R"]:.2f} Ohm  '
                    f'[{detrend_lbl}]'
                )
        except FileNotFoundError as e:
            if verbose:
                print(f'  {stem:10s} -- NOT FOUND: {e}')
    return out
