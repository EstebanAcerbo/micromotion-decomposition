"""
check_shift_boundary.py
=======================
Diagnostic: does the insulating-wall boundary condition contaminate
Library A (electrode shift) at large dx offsets?

Geometry
--------
  domain side     : 441 px
  electrode radius: (de/2) / d  [px]  (computed from baseline.npz)
  electrode centre: dom_cx + dx_px  (dom_cx = side/2 = 220.5 px)
  right margin    : (side - 1) - (dom_cx + dx_px + elec_r_px)

Threshold lines
---------------
  margin = 20 px  -> dx_px ~ 24.28  (orange dashed)
  margin = 10 px  -> dx_px ~ 34.28  (red dashed)
  dx = 40 px gives margin ~ 4.28 px (deep into artifact-risk zone)

Output
------
  results/check_shift_boundary.png
  Console table: dR, dX, |dZ| for dx = 0, 5, 10, ..., 40 at ~924 Hz
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE    = os.path.dirname(os.path.abspath(__file__))
RESULTS  = os.path.join(_HERE, '..', 'results')
LIB_PATH = os.path.join(RESULTS, 'library_A_shift.npz')
BL_PATH  = os.path.join(RESULTS, 'baseline.npz')
OUT_PATH = os.path.join(RESULTS, 'check_shift_boundary.png')


# ── Load data ─────────────────────────────────────────────────────────────────
print('Loading library_A_shift.npz ...')
lib = np.load(LIB_PATH, allow_pickle=True)

dx_px_all = lib['dx_px']          # (861,) int
dy_px_all = lib['dy_px']          # (861,) int
freqs     = lib['freqs']           # (30,)  float
Z_lib     = lib['Z']               # (861, 30) complex
d         = float(lib['d'])        # pixel size [cm]

print('Loading baseline.npz ...')
bl        = np.load(BL_PATH, allow_pickle=True)
Z_baseline = bl['Z']               # (30,) complex -- from baseline.npz
side      = int(bl['side'])        # 441 px
de        = float(bl['de'])        # electrode diameter [cm]

# Electrode radius in pixels
elec_r_px = (de / 2.0) / d        # ~175.22 px
dom_cx    = side / 2.0             # 220.5 px
d_um_per_px = d * 1e4              # µm per pixel


# ── Geometry thresholds ────────────────────────────────────────────────────────
# right_margin = (side-1) - (dom_cx + dx_px + elec_r_px)
# Set right_margin = M  ->  dx_px = (side-1) - dom_cx - elec_r_px - M
def dx_at_margin(M):
    return (side - 1) - dom_cx - elec_r_px - M

dx_thresh_10 = dx_at_margin(10)   # ~34.28 px
dx_thresh_20 = dx_at_margin(20)   # ~24.28 px

dx_thresh_10_um = dx_thresh_10 * d_um_per_px
dx_thresh_20_um = dx_thresh_20 * d_um_per_px

print(f'\nGeometry:')
print(f'  domain side     = {side} px')
print(f'  electrode radius= {elec_r_px:.2f} px')
print(f'  pixel size      = {d_um_per_px:.6f} µm/px')
print(f'  margin=10 px at dx = {dx_thresh_10:.2f} px  ({dx_thresh_10_um:.2f} µm)')
print(f'  margin=20 px at dx = {dx_thresh_20:.2f} px  ({dx_thresh_20_um:.2f} µm)')


# ── Select dy=0 slice ─────────────────────────────────────────────────────────
mask_dy0 = (dy_px_all == 0)
idx_dy0  = np.where(mask_dy0)[0]           # indices into Z_lib
dx_vals  = dx_px_all[idx_dy0]             # 0,1,...,40
sort_order = np.argsort(dx_vals)
idx_dy0  = idx_dy0[sort_order]
dx_vals  = dx_vals[sort_order]

Z_dy0    = Z_lib[idx_dy0, :]               # (41, 30)


# ── Frequency selection: closest to 924 Hz ────────────────────────────────────
target_freq = 924.0   # Hz
i_freq = int(np.argmin(np.abs(freqs - target_freq)))
actual_freq = freqs[i_freq]
print(f'\nTarget frequency: {target_freq} Hz -> closest bin: {actual_freq:.2f} Hz  (index {i_freq})')


# ── dZ at selected frequency ──────────────────────────────────────────────────
# Use Z_baseline from baseline.npz (not the one embedded in the library)
dZ = Z_dy0[:, i_freq] - Z_baseline[i_freq]

dR  = dZ.real
dX  = dZ.imag
absdZ = np.abs(dZ)

dx_um = dx_vals * d_um_per_px   # convert to µm


# ── Console table ─────────────────────────────────────────────────────────────
print(f'\n{"dx(px)":>7}  {"dx(um)":>8}  {"right_margin(px)":>17}  '
      f'{"dR(ohm)":>12}  {"dX(ohm)":>12}  {"|dZ|(ohm)":>12}')
print('-' * 80)

report_dx = [0, 5, 10, 15, 20, 25, 30, 35, 40]
for dx_val in report_dx:
    idx = np.where(dx_vals == dx_val)[0]
    if len(idx) == 0:
        continue
    i = idx[0]
    rm = (side - 1) - (dom_cx + dx_val + elec_r_px)
    flag = ''
    if rm < 10:
        flag = ' << margin<10'
    elif rm < 20:
        flag = ' < margin<20'
    print(f'{dx_val:>7d}  {dx_um[i]:>8.3f}  {rm:>17.2f}  '
          f'{dR[i]:>12.6f}  {dX[i]:>12.6f}  {absdZ[i]:>12.6f}{flag}')


# ── Monotonicity check ────────────────────────────────────────────────────────
def check_monotonic(arr, label, dx_array, thresh_px):
    """Return list of (dx_px, dx_px+1) where arr is non-monotonic past thresh."""
    past = np.where(dx_array > thresh_px)[0]
    reversals = []
    for j in past:
        if j == 0:
            continue
        if arr[j] < arr[j-1] - 0.0 and arr[j-1] > arr[j-2]:
            reversals.append(int(dx_array[j]))
        # simpler: sign change in first difference
    diffs = np.diff(arr[past])
    sign_changes = np.where(np.diff(np.sign(diffs)))[0]
    return sign_changes, past

_, past_dR  = check_monotonic(dR,    'dR',   dx_vals, dx_thresh_10)
_, past_dX  = check_monotonic(dX,    'dX',   dx_vals, dx_thresh_10)
_, past_abs = check_monotonic(absdZ, '|dZ|', dx_vals, dx_thresh_10)

# Detect sign changes in first differences past margin=10 line
def sign_change_count(arr, past_idx):
    if len(past_idx) < 2:
        return 0
    d = np.diff(arr[past_idx])
    return int(np.sum(np.diff(np.sign(d)) != 0))

sc_dR  = sign_change_count(dR,    past_dR)
sc_dX  = sign_change_count(dX,    past_dX)
sc_abs = sign_change_count(absdZ, past_abs)

artifact_visible = sc_dR > 0 or sc_dX > 0 or sc_abs > 0

if artifact_visible:
    title_note = (f'NON-MONOTONIC behavior past margin=10 px line detected '
                  f'(dR: {sc_dR}, dX: {sc_dX}, |dZ|: {sc_abs} sign changes)')
else:
    title_note = 'No clear non-monotonic inflection past margin=10 px line'

print(f'\nMonotonicity check past dx_thresh_10 ({dx_thresh_10:.1f} px):')
print(f'  sign changes: dR={sc_dR}  dX={sc_dX}  |dZ|={sc_abs}')
print(f'  -> {title_note}')


# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

datasets = [
    (axes[0], dR,    'dR  [Ω]',   'tab:blue'),
    (axes[1], dX,    'dX  [Ω]',   'tab:orange'),
    (axes[2], absdZ, '|dZ|  [Ω]', 'tab:green'),
]

for ax, data, ylabel, color in datasets:
    ax.plot(dx_um, data, color=color, lw=1.5, marker='o', ms=3.5,
            markerfacecolor='white', markeredgecolor=color, markeredgewidth=1.2)

    # Vertical threshold lines
    ax.axvline(dx_thresh_10_um, color='red',    lw=1.5, ls='--',
               label='margin = 10 px', zorder=4)
    ax.axvline(dx_thresh_20_um, color='darkorange', lw=1.5, ls='--',
               label='margin = 20 px', zorder=4)

    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, lw=0.4, alpha=0.5)
    ax.legend(fontsize=8, loc='best')

    # Shade artifact-risk region (past margin=10)
    xlim_right = dx_um[-1] * 1.03
    ax.axvspan(dx_thresh_10_um, xlim_right, color='red', alpha=0.06, zorder=0)

axes[-1].set_xlabel('dx  [µm]', fontsize=11)
axes[-1].set_xlim(dx_um[0] - 0.5, dx_um[-1] * 1.03)

fig.suptitle(
    f'Library A shift — dy=0 slice  |  f ≈ {actual_freq:.1f} Hz\n'
    f'{title_note}',
    fontsize=10, y=0.995, va='top'
)
fig.tight_layout(rect=[0, 0, 1, 0.975])
fig.savefig(OUT_PATH, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'\nSaved -> {OUT_PATH}')
