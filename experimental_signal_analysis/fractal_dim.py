"""
fractal_dim.py — Fractal dimension estimation for 1D signals.

Four methods matching acerbo2024determining:
  'higuchi' : Higuchi algorithm (time-domain curve length at varying lags)
  'mrlb'    : Multiresolution Length-Based
  'mrbc'    : Multiresolution Box-Counting
  'rs'      : Rescaled Range (Hurst exponent → df = 2 − H)

For all methods the log-log regression slope β gives:
  df = −β          (Higuchi, MRLB, MRBC)
  df = 2 − β       (R/S: β = H, Hurst exponent)

Public API
----------
    result = estimate_fd(signal, dt, method,
                         fit_range=None, remove_ma=None, plot=False)
    # result keys: df, df_err, slope, intercept, x_fit, y_fit

    bp = find_breakpoint(x_fit, y_fit, min_pts=4)
    # bp keys: x_break, df_short, df_long, df_short_err, df_long_err,
    #          ssr_two, ssr_one, improvement, idx_break

Core algorithms are ported from FractalDimEstimator.py with minimal changes.
"""

import numpy as np
from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt

# ─── log-log linear fit ───────────────────────────────────────────────────────

def _lin_fit_loglog(x, y):
    """Fit log10(y) ~ slope * log10(x) + intercept. Returns (slope, intercept, slope_stderr)."""
    lx = np.log10(x)
    ly = np.log10(y)
    coeffs, cov = np.polyfit(lx, ly, 1, cov=True)
    slope, intercept = float(coeffs[0]), float(coeffs[1])
    slope_stderr = float(np.sqrt(cov[0, 0]))
    return slope, intercept, slope_stderr

# ─── core algorithms ──────────────────────────────────────────────────────────

def _higuchi(signal, k_max=100):
    """Higuchi algorithm. Returns (k_array, L_mean_array); L ~ k^(-df)."""
    k_max = min(k_max, len(signal) // 4)
    N = len(signal)
    k_ = np.arange(1, k_max + 1, dtype=int)
    L_mean = np.zeros(len(k_))
    for i, k in enumerate(k_):
        L = np.zeros(k)
        for m in range(k):
            a = int((N - m) / k)
            diffs = np.abs(np.diff(signal[m::k]))
            L[m] = ((N - 1) / (k * k * a)) * np.sum(diffs)
        L_mean[i] = np.mean(L)
    return k_.astype(float), L_mean


def _mrlb(t, signal):
    """Multiresolution Length-Based. Returns (lags_s, normalised_length); measure ~ lag^(-df).
    Signal is amplitude-normalised so mean(|dy|) = 1e4 * dx, matching the paper's condition."""
    dx = float(np.mean(np.diff(t)))
    N  = len(signal)
    # Amplitude normalisation (paper: acerbo2024determining, Sec. Methods)
    dy_mean = float(np.mean(np.abs(np.diff(signal))))
    y = signal * (1e4 * dx / dy_mean) if dy_mean > 0 else signal.copy()
    Skip = np.unique(np.geomspace(N / 2500, N / 20, 100).astype(int))
    Skip = Skip[Skip > 0]
    D_ = np.zeros(len(Skip))
    for i, s in enumerate(Skip):
        d_ = np.zeros(s)
        for j in range(s):
            d_[j] = np.sum(
                np.sqrt((y[s + j::s] - y[j:-s:s]) ** 2 + (dx * s) ** 2)
            )
        D_[i] = np.mean(d_)
    lags = Skip * dx
    return lags, D_ / lags


def _mrbc(t, signal):
    """Multiresolution Box-Counting. Returns (lags_s, box_count); measure ~ lag^(-df).
    Signal is amplitude-normalised so mean(|dy|) = 1e4 * dx."""
    dx = float(np.mean(np.diff(t)))
    N  = len(signal)
    dy_mean = float(np.mean(np.abs(np.diff(signal))))
    y = signal * (1e4 * dx / dy_mean) if dy_mean > 0 else signal.copy()
    Skip = np.unique(np.geomspace(N / 2500, N / 20, 100).astype(int))
    Skip = Skip[Skip > 0]
    B_ = np.zeros(len(Skip))
    for i, s in enumerate(Skip):
        b_ = np.zeros(s)
        dt_sub = dx * s
        for j in range(s):
            b_[j] = np.sum(
                np.ceil(np.abs(y[s + j::s] - y[j:-s:s]) / dt_sub)
            )
        B_[i] = np.mean(b_)
    return Skip * dx, B_


def _rs(signal):
    """Rescaled Range analysis. Returns (lags_samples, RS_values); RS ~ lag^H."""
    intervalos = np.unique(np.linspace(10, 100, 50).astype(int))
    RS     = np.zeros(len(intervalos))
    N      = len(signal)
    for i, n_int in enumerate(intervalos):
        Puntos = int(N / n_int)
        if Puntos < 10:
            RS[i] = np.nan
            continue
        rs_vals = []
        for k in range(n_int):
            seg = signal[k * Puntos: (k + 1) * Puntos]
            I   = np.diff(seg)
            AI  = I - np.mean(I)
            Cum = np.cumsum(AI)
            R   = np.max(Cum) - np.min(Cum)
            S   = np.std(I, ddof=1)
            if S > 0:
                rs_vals.append(R / S)
        if rs_vals:
            RS[i] = float(np.mean(rs_vals))
        else:
            RS[i] = np.nan
    lags = N / intervalos.astype(float)
    valid = np.isfinite(RS) & (RS > 0) & (lags > 0)
    return lags[valid], RS[valid]

# ─── public API ───────────────────────────────────────────────────────────────

def estimate_fd(signal, dt, method,
                fit_range=None, remove_ma=None, plot=False):
    """
    Estimate fractal dimension of a 1D signal.

    Parameters
    ----------
    signal     : 1D array-like — δR or δX fluctuations [Ω]
    dt         : float — sampling interval [s]
    method     : str — 'higuchi' | 'mrlb' | 'mrbc' | 'rs'
    fit_range  : (t_min, t_max) in seconds (or samples for Higuchi);
                 restricts the regression window; None = full range
    remove_ma  : float — optional MA window [s] subtracted before estimation
    plot       : bool — show diagnostic log-log scatter with fitted line

    Returns
    -------
    dict with keys:
        df        : float — fractal dimension
        df_err    : float — 1-σ std error on df (from regression)
        slope     : float — raw regression slope
        intercept : float — regression intercept (log10 scale)
        x_fit     : ndarray — x values used in the fit
        y_fit     : ndarray — y values used in the fit
    """
    sig = np.asarray(signal, dtype=float).copy()

    # Optional moving-average removal
    if remove_ma is not None:
        w = max(3, int(round(remove_ma / dt)))
        if w % 2 == 0:
            w += 1
        sig -= uniform_filter1d(sig, size=w, mode='nearest')

    t = np.arange(len(sig)) * dt

    # Compute (x_arr, y_arr) depending on method
    if method == 'higuchi':
        x_arr, y_arr = _higuchi(sig)
        # x_arr is in samples; convert fit_range from seconds to samples
        if fit_range is not None:
            k_lo = fit_range[0] / dt
            k_hi = fit_range[1] / dt
            mask = (x_arr >= k_lo) & (x_arr <= k_hi)
        else:
            mask = np.ones(len(x_arr), bool)

    elif method == 'mrlb':
        x_arr, y_arr = _mrlb(t, sig)
        if fit_range is not None:
            mask = (x_arr >= fit_range[0]) & (x_arr <= fit_range[1])
        else:
            mask = np.ones(len(x_arr), bool)

    elif method == 'mrbc':
        x_arr, y_arr = _mrbc(t, sig)
        if fit_range is not None:
            mask = (x_arr >= fit_range[0]) & (x_arr <= fit_range[1])
        else:
            mask = np.ones(len(x_arr), bool)

    elif method == 'rs':
        x_arr, y_arr = _rs(sig)
        if fit_range is not None:
            mask = (x_arr >= fit_range[0]) & (x_arr <= fit_range[1])
        else:
            mask = np.ones(len(x_arr), bool)

    else:
        raise ValueError(
            f"Unknown method {method!r}. Choose: 'higuchi', 'mrlb', 'mrbc', 'rs'."
        )

    # Apply mask and positivity filter
    valid = mask & (x_arr > 0) & (y_arr > 0) & np.isfinite(y_arr)
    x_fit = x_arr[valid]
    y_fit = y_arr[valid]

    _nan = dict(df=np.nan, df_err=np.nan, slope=np.nan,
                intercept=np.nan, x_fit=x_fit, y_fit=y_fit)
    if len(x_fit) < 3:
        return _nan

    slope, intercept, slope_stderr = _lin_fit_loglog(x_fit, y_fit)

    if method == 'rs':
        # Hurst exponent H = slope → df = 2 - H
        df     =  2.0 - slope
        df_err = slope_stderr
    else:
        # L ~ lag^(-df) → df = -slope
        df     = -slope
        df_err = slope_stderr

    if plot:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.loglog(x_arr[x_arr > 0], y_arr[x_arr > 0],
                  'o', ms=3, alpha=0.5, label='all data')
        ax.loglog(x_fit, y_fit, 's', ms=4, color='tab:orange', label='fit range')
        x_line = np.logspace(np.log10(x_fit.min()), np.log10(x_fit.max()), 200)
        ax.loglog(x_line,
                  10 ** (intercept + slope * np.log10(x_line)),
                  'r--', lw=1.5,
                  label=fr'$d_f = {df:.3f} \pm {df_err:.3f}$')
        ax.set_xlabel('Lag')
        ax.set_ylabel('Measure')
        ax.set_title(f'{method.upper()}')
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.show()

    return dict(df=df, df_err=df_err, slope=slope,
                intercept=intercept, x_fit=x_fit, y_fit=y_fit)


# ─── piecewise breakpoint detection ───────────────────────────────────────────

def find_breakpoint(x_fit, y_fit, min_pts=4):
    """
    Find the single breakpoint in a log-log curve that best splits it into
    two linear segments (piecewise regression in log10-log10 space).

    Parameters
    ----------
    x_fit, y_fit : arrays — the (x, y) values used in the fractal fit
                   (positive; will be log10-transformed internally)
    min_pts      : int — minimum points required in each segment

    Returns
    -------
    dict with keys:
        x_break    : float — breakpoint x value (same units as x_fit)
        df_short   : float — fractal dimension for x < x_break (short scale)
        df_long    : float — fractal dimension for x >= x_break (long scale)
        df_short_err, df_long_err : float — regression std errors
        ssr_two    : float — total SSR of two-segment fit
        ssr_one    : float — SSR of single-segment fit (for comparison)
        improvement: float — fractional SSR reduction: 1 - ssr_two/ssr_one
        idx_break  : int — index of breakpoint in x_fit
    """
    x_fit = np.asarray(x_fit, dtype=float)
    y_fit = np.asarray(y_fit, dtype=float)

    lx = np.log10(x_fit)
    ly = np.log10(y_fit)
    n  = len(lx)

    _nan = dict(x_break=np.nan, df_short=np.nan, df_long=np.nan,
                df_short_err=np.nan, df_long_err=np.nan,
                ssr_two=np.nan, ssr_one=np.nan,
                improvement=np.nan, idx_break=None)

    if n < 2 * min_pts:
        return _nan

    # Single-segment SSR
    c1 = np.polyfit(lx, ly, 1)
    res_one = ly - np.polyval(c1, lx)
    ssr_one = float(np.dot(res_one, res_one))

    # Scan candidate breakpoints
    best_ssr = np.inf
    best_i   = None
    for i in range(min_pts, n - min_pts + 1):
        c_left  = np.polyfit(lx[:i], ly[:i], 1)
        c_right = np.polyfit(lx[i:], ly[i:], 1)
        res_l   = ly[:i] - np.polyval(c_left,  lx[:i])
        res_r   = ly[i:] - np.polyval(c_right, lx[i:])
        ssr = float(np.dot(res_l, res_l) + np.dot(res_r, res_r))
        if ssr < best_ssr:
            best_ssr = ssr
            best_i   = i

    if best_i is None:
        return _nan

    # Recompute optimal split with covariance to get std errors
    coeffs_l, cov_l = np.polyfit(lx[:best_i], ly[:best_i], 1, cov=True)
    coeffs_r, cov_r = np.polyfit(lx[best_i:], ly[best_i:], 1, cov=True)

    slope_left  = float(coeffs_l[0])
    slope_right = float(coeffs_r[0])
    err_left    = float(np.sqrt(cov_l[0, 0]))
    err_right   = float(np.sqrt(cov_r[0, 0]))

    improvement = 1.0 - best_ssr / ssr_one if ssr_one > 0 else 0.0

    return dict(
        x_break     = float(x_fit[best_i]),
        df_short    = -slope_left,
        df_long     = -slope_right,
        df_short_err= err_left,
        df_long_err = err_right,
        ssr_two     = best_ssr,
        ssr_one     = ssr_one,
        improvement = float(improvement),
        idx_break   = int(best_i),
    )


# ─── windowed fractal analysis ────────────────────────────────────────────────

def windowed_fractal_analysis(signal, dt, method, max_windows=10, target_win_pts=12500):
    """
    Windowed fractal analysis: split signal into equal windows, run estimate_fd
    on each window, average the log-log curves, detect breakpoint on the average,
    and refit each window's curve at the forced split to obtain per-window
    fast/slow slope statistics.

    Parameters
    ----------
    signal         : 1D array — signal to analyse
    dt             : float — sampling interval [s]
    method         : str — 'higuchi' | 'mrlb' | 'mrbc' | 'rs'
    max_windows    : int — maximum number of windows (default 10)
    target_win_pts : int — target points per window (default 12500)

    Returns
    -------
    dict with keys:
        x_arr, mean_y          — averaged log-log curve arrays
        n_win, w_size          — number of windows and window length [pts]
        x_break, idx_break     — breakpoint x value and index (idx_break may be None)
        df_single_mean/std     — windowed single-slope df statistics
        df_fast_mean/std       — windowed fast-segment df statistics
        df_slow_mean/std       — windowed slow-segment df statistics
        slope_avg, intercept_avg          — single-slope fit on mean_y
        slope_fast_avg, intercept_fast_avg — fast-segment fit on mean_y
        slope_slow_avg, intercept_slow_avg — slow-segment fit on mean_y
        df_avg                 — df from single-slope fit on averaged curve
        df_fast_avg            — df from fast-segment fit on averaged curve
        df_slow_avg            — df from slow-segment fit on averaged curve
    """
    sig = np.asarray(signal, dtype=float)
    N   = len(sig)

    # 1. Determine window count and size
    n_win  = max(1, min(max_windows, N // target_win_pts))
    w_size = N // n_win
    sig    = sig[:n_win * w_size]   # truncate so all windows are identical length

    # 2. Run estimate_fd on each window; collect x_arr and y_arr per window
    x_arr_ref = None
    y_arrs    = []
    df_vals   = []

    for k in range(n_win):
        win = sig[k * w_size: (k + 1) * w_size]
        res = estimate_fd(win, dt, method)
        x_k = res['x_fit']
        y_k = res['y_fit']
        if len(x_k) < 3 or not np.isfinite(res['df']):
            continue
        if x_arr_ref is None:
            x_arr_ref = x_k
        else:
            # Interpolate y_k onto the reference x grid so all arrays align
            if not np.array_equal(x_k, x_arr_ref):
                y_k = np.interp(np.log10(x_arr_ref), np.log10(x_k), np.log10(y_k))
                y_k = 10.0 ** y_k
        y_arrs.append(y_k)
        df_vals.append(res['df'])

    if x_arr_ref is None or len(y_arrs) == 0:
        # Return NaN dict
        return dict(
            x_arr=np.array([]), mean_y=np.array([]),
            n_win=n_win, w_size=w_size,
            x_break=np.nan, idx_break=None,
            df_single_mean=np.nan, df_single_std=np.nan,
            df_fast_mean=np.nan, df_fast_std=np.nan,
            df_slow_mean=np.nan, df_slow_std=np.nan,
            slope_avg=np.nan, intercept_avg=np.nan,
            slope_fast_avg=np.nan, intercept_fast_avg=np.nan,
            slope_slow_avg=np.nan, intercept_slow_avg=np.nan,
            df_avg=np.nan, df_fast_avg=np.nan, df_slow_avg=np.nan,
        )

    x_arr  = x_arr_ref
    # Clip y_arrs to the shortest length (in case of any length mismatch)
    min_len = min(len(y) for y in y_arrs)
    x_arr   = x_arr[:min_len]
    y_arrs  = [y[:min_len] for y in y_arrs]

    # 3. Average y_arr across windows (in linear space)
    mean_y = np.mean(np.array(y_arrs), axis=0)

    # 4. Find breakpoint on the averaged curve
    bp = find_breakpoint(x_arr, mean_y)
    idx_break = bp['idx_break']   # None if not found
    x_break   = bp['x_break']     # float or NaN

    # 5. Per-window two-segment fits at the forced split
    df_fast_vals = []
    df_slow_vals = []

    if idx_break is not None and idx_break >= 2 and idx_break <= min_len - 2:
        lx = np.log10(x_arr)
        for y_k in y_arrs:
            ly_k = np.log10(y_k)
            # Fast segment (left)
            c_fast = np.polyfit(lx[:idx_break], ly_k[:idx_break], 1)
            slope_fast_k = float(c_fast[0])
            # Slow segment (right)
            c_slow = np.polyfit(lx[idx_break:], ly_k[idx_break:], 1)
            slope_slow_k = float(c_slow[0])

            if method == 'rs':
                # slope is H (positive) → df = 2 - H
                df_fast_vals.append(2.0 - slope_fast_k)
                df_slow_vals.append(2.0 - slope_slow_k)
            else:
                df_fast_vals.append(-slope_fast_k)
                df_slow_vals.append(-slope_slow_k)
    else:
        # No valid breakpoint — fill with NaN
        df_fast_vals = [np.nan] * len(y_arrs)
        df_slow_vals = [np.nan] * len(y_arrs)

    # 6. Statistics
    df_single_mean = float(np.mean(df_vals))
    df_single_std  = float(np.std(df_vals))
    df_fast_mean   = float(np.nanmean(df_fast_vals)) if df_fast_vals else np.nan
    df_fast_std    = float(np.nanstd(df_fast_vals))  if df_fast_vals else np.nan
    df_slow_mean   = float(np.nanmean(df_slow_vals)) if df_slow_vals else np.nan
    df_slow_std    = float(np.nanstd(df_slow_vals))  if df_slow_vals else np.nan

    # 7. Fit averaged curve: single-slope, fast segment, slow segment
    lx_all = np.log10(x_arr)
    ly_all = np.log10(mean_y)

    c_avg  = np.polyfit(lx_all, ly_all, 1)
    slope_avg, intercept_avg = float(c_avg[0]), float(c_avg[1])

    if method == 'rs':
        df_avg = 2.0 - slope_avg
    else:
        df_avg = -slope_avg

    if idx_break is not None and idx_break >= 2 and idx_break <= min_len - 2:
        c_fast_avg = np.polyfit(lx_all[:idx_break], ly_all[:idx_break], 1)
        c_slow_avg = np.polyfit(lx_all[idx_break:], ly_all[idx_break:], 1)
        slope_fast_avg, intercept_fast_avg = float(c_fast_avg[0]), float(c_fast_avg[1])
        slope_slow_avg, intercept_slow_avg = float(c_slow_avg[0]), float(c_slow_avg[1])
        if method == 'rs':
            df_fast_avg = 2.0 - slope_fast_avg
            df_slow_avg = 2.0 - slope_slow_avg
        else:
            df_fast_avg = -slope_fast_avg
            df_slow_avg = -slope_slow_avg
    else:
        slope_fast_avg = intercept_fast_avg = np.nan
        slope_slow_avg = intercept_slow_avg = np.nan
        df_fast_avg = df_slow_avg = np.nan

    # 8. Return full dict
    return dict(
        x_arr             = x_arr,
        mean_y            = mean_y,
        n_win             = n_win,
        w_size            = w_size,
        x_break           = float(x_break) if np.isfinite(x_break) else np.nan,
        idx_break         = idx_break,
        df_single_mean    = df_single_mean,
        df_single_std     = df_single_std,
        df_fast_mean      = df_fast_mean,
        df_fast_std       = df_fast_std,
        df_slow_mean      = df_slow_mean,
        df_slow_std       = df_slow_std,
        slope_avg         = slope_avg,
        intercept_avg     = intercept_avg,
        slope_fast_avg    = slope_fast_avg,
        intercept_fast_avg= intercept_fast_avg,
        slope_slow_avg    = slope_slow_avg,
        intercept_slow_avg= intercept_slow_avg,
        df_avg            = df_avg,
        df_fast_avg       = df_fast_avg,
        df_slow_avg       = df_slow_avg,
    )
