import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

RESULTS = 'C:/Users/esteb/Archivos/Workbench/Micromotion_Simul/results'

greedy  = np.load(f'{RESULTS}/reconstruct_v3_eb11.npz', allow_pickle=True)
viterbi = np.load(f'{RESULTS}/reconstruct_v4_eb11.npz', allow_pickle=True)

t       = greedy['t']
dR_exp  = greedy['dR_exp']
dX_exp  = greedy['dX_exp']

dR_rec_g = greedy['dR_rec']
dX_rec_g = greedy['dX_rec']

dR_rec_v = viterbi['dR_rec']
dX_rec_v = viterbi['dX_rec']

std_dR = np.std(dR_exp)
std_dX = np.std(dX_exp)

rmse_R_greedy  = float(greedy['rmse_R'])
rmse_X_greedy  = float(greedy['rmse_X'])
rmse_R_viterbi = float(viterbi['rmse_R'])
rmse_X_viterbi = float(viterbi['rmse_X'])

pct_R_greedy  = rmse_R_greedy  / std_dR * 100
pct_X_greedy  = rmse_X_greedy  / std_dX * 100
pct_R_viterbi = rmse_R_viterbi / std_dR * 100
pct_X_viterbi = rmse_X_viterbi / std_dX * 100

# Fixed window: steps 1000 to 1600
sl = slice(1000, 1600)
t_win = t[sl]

dt = float(np.mean(np.diff(t)))
fs = 1.0 / dt

nperseg = 4096

f_R, psd_R_exp = welch(dR_exp,  fs=fs, nperseg=nperseg, scaling='density')
f_R, psd_R_g   = welch(dR_rec_g, fs=fs, nperseg=nperseg, scaling='density')
f_R, psd_R_v   = welch(dR_rec_v, fs=fs, nperseg=nperseg, scaling='density')

f_X, psd_X_exp = welch(dX_exp,  fs=fs, nperseg=nperseg, scaling='density')
f_X, psd_X_g   = welch(dX_rec_g, fs=fs, nperseg=nperseg, scaling='density')
f_X, psd_X_v   = welch(dX_rec_v, fs=fs, nperseg=nperseg, scaling='density')

# Skip DC bin (index 0)
f_R_plot = f_R[1:]
f_X_plot = f_X[1:]
psd_R_exp_plot = psd_R_exp[1:]
psd_R_g_plot   = psd_R_g[1:]
psd_R_v_plot   = psd_R_v[1:]
psd_X_exp_plot = psd_X_exp[1:]
psd_X_g_plot   = psd_X_g[1:]
psd_X_v_plot   = psd_X_v[1:]

# 1/f^1.7 reference line anchored at 5th frequency bin (index 4 of non-DC array)
anchor_idx = 4
slope = 1.7

def ref_line(f_arr, psd_ref, anchor_i):
    anchor_f   = f_arr[anchor_i]
    anchor_psd = psd_ref[anchor_i]
    return anchor_psd * (f_arr / anchor_f) ** (-slope)

ref_R = ref_line(f_R_plot, psd_R_exp_plot, anchor_idx)
ref_X = ref_line(f_X_plot, psd_X_exp_plot, anchor_idx)

# colors: experimental black, greedy tab:blue, viterbi tab:orange, ref gray
# alpha: experimental 1.0, greedy 0.85, viterbi 0.70
# linestyles: experimental solid, greedy dashed, viterbi dotted

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
ax1, ax2, ax3, ax4 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

# --- Top-left: dR time series ---
ax1.plot(t_win, dR_rec_v[sl], color='tab:orange', lw=1.2, ls='dotted',  alpha=0.70, zorder=1,
         label=f'Viterbi (RMSE {rmse_R_viterbi:.0f} \u03a9, {pct_R_viterbi:.0f}%)')
ax1.plot(t_win, dR_rec_g[sl], color='tab:blue',   lw=1.2, ls='dashed',  alpha=0.85, zorder=2,
         label=f'Greedy (RMSE {rmse_R_greedy:.0f} \u03a9, {pct_R_greedy:.0f}%)')
ax1.plot(t_win, dR_exp[sl],   color='black',      lw=1.0, ls='solid',   alpha=1.00, zorder=3,
         label='Experimental')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('\u03b4R (\u03a9)')
ax1.legend(loc='upper right', fontsize=8)
ax1.text(0.02, 0.97, '(a)', transform=ax1.transAxes,
         fontsize=10, fontweight='bold', va='top')

# --- Top-right: dX time series ---
ax2.plot(t_win, dX_rec_v[sl], color='tab:orange', lw=1.2, ls='dotted',  alpha=0.70, zorder=1,
         label=f'Viterbi (RMSE {rmse_X_viterbi:.0f} \u03a9, {pct_X_viterbi:.0f}%)')
ax2.plot(t_win, dX_rec_g[sl], color='tab:blue',   lw=1.2, ls='dashed',  alpha=0.85, zorder=2,
         label=f'Greedy (RMSE {rmse_X_greedy:.0f} \u03a9, {pct_X_greedy:.0f}%)')
ax2.plot(t_win, dX_exp[sl],   color='black',      lw=1.0, ls='solid',   alpha=1.00, zorder=3,
         label='Experimental')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('\u03b4X (\u03a9)')
ax2.legend(loc='upper right', fontsize=8)
ax2.text(0.02, 0.97, '(b)', transform=ax2.transAxes,
         fontsize=10, fontweight='bold', va='top')

# --- Bottom-left: PSD dR ---
ax3.loglog(f_R_plot, psd_R_v_plot,   color='tab:orange', lw=1.2, ls='dotted',  alpha=0.70, zorder=1, label='Viterbi v4')
ax3.loglog(f_R_plot, psd_R_g_plot,   color='tab:blue',   lw=1.2, ls='dashed',  alpha=0.85, zorder=2, label='Greedy v3')
ax3.loglog(f_R_plot, psd_R_exp_plot, color='black',      lw=1.0, ls='solid',   alpha=1.00, zorder=3, label='Experimental')
ax3.loglog(f_R_plot, ref_R,          color='gray',       lw=1.0, ls=(0,(3,1,1,1)), alpha=0.6, zorder=0, label=r'$1/f^{1.7}$')
ax3.set_xlabel('Frequency (Hz)')
ax3.set_ylabel(r'PSD (\u03a9\u00b2/Hz)')
ax3.legend(loc='upper right', fontsize=8)
ax3.text(0.02, 0.97, '(c)', transform=ax3.transAxes,
         fontsize=10, fontweight='bold', va='top')

# --- Bottom-right: PSD dX ---
ax4.loglog(f_X_plot, psd_X_v_plot,   color='tab:orange', lw=1.2, ls='dotted',  alpha=0.70, zorder=1, label='Viterbi v4')
ax4.loglog(f_X_plot, psd_X_g_plot,   color='tab:blue',   lw=1.2, ls='dashed',  alpha=0.85, zorder=2, label='Greedy v3')
ax4.loglog(f_X_plot, psd_X_exp_plot, color='black',      lw=1.0, ls='solid',   alpha=1.00, zorder=3, label='Experimental')
ax4.loglog(f_X_plot, ref_X,          color='gray',       lw=1.0, ls=(0,(3,1,1,1)), alpha=0.6, zorder=0, label=r'$1/f^{1.7}$')
ax4.set_xlabel('Frequency (Hz)')
ax4.set_ylabel(r'PSD (\u03a9\u00b2/Hz)')
ax4.legend(loc='upper right', fontsize=8)
ax4.text(0.02, 0.97, '(d)', transform=ax4.transAxes,
         fontsize=10, fontweight='bold', va='top')

fig.suptitle('Electrode eb11 — Confluent monolayer — Greedy vs. Viterbi reconstruction',
             fontsize=11)
plt.tight_layout(rect=[0, 0, 1, 0.96])

out_path = f'{RESULTS}/report_eb11_comparison.png'
fig.savefig(out_path, dpi=150)
plt.close(fig)

print('Saved: results/report_eb11_comparison.png')
print(f'Greedy  dR RMSE: {rmse_R_greedy:.1f} Ohm ({pct_R_greedy:.1f}%)   dX RMSE: {rmse_X_greedy:.1f} Ohm ({pct_X_greedy:.1f}%)')
print(f'Viterbi dR RMSE: {rmse_R_viterbi:.1f} Ohm ({pct_R_viterbi:.1f}%)   dX RMSE: {rmse_X_viterbi:.1f} Ohm ({pct_X_viterbi:.1f}%)')
