"""
gk_sensitivity.py
=================
GK parameter sensitivity using the FD engine (solve_Z) at n=10.
Sweeps +-15% in alpha, Rb, Cm for three reference parameter sets:
  - CH3 GK fit    (alpha=19.4, Rb=14.4, Cm=4.95e-6)  <- engine default
  - Classifier CH2 (alpha=5,  Rb=64,   Cm=3.70e-6)
  - Old engine    (alpha=13.3, Rb=11.0, Cm=5.67e-6)

Workflow
--------
  Run:    python gk_sensitivity.py            -- computes + saves data + plots
  Replot: python gk_sensitivity.py --plot-only -- loads saved npz, skips FD solve

Outputs
-------
  results/gk_sensitivity_data.npz   -- ALL computed Z arrays (raw data)
  results/gk_sensitivity_*.png
  results/gk_sensitivity_comparison.png
"""

import sys, os, time, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm as mplcm

_HERE = os.path.dirname(os.path.abspath(__file__))

from engine import FREQS

N_RES     = 20
SWEEP_N   = 11
SWEEP_PCT = 0.05

DATA_OUT = os.path.join(_HERE, '..', 'results', 'gk_sensitivity_data.npz')

# ── Reference parameter sets ─────────────────────────────────────────────────
REFS = [
    dict(label='CH3_fit',
         title='CH3 GK fit  (alpha=19.4, Rb=14.4, Cm=4.95 uF/cm2)',
         alpha=19.4, Rb=14.4, Cm=4.95e-6),
    dict(label='classifier_CH2',
         title='Classifier CH2  (alpha=5, Rb=64, Cm=3.70 uF/cm2)',
         alpha=5.0,  Rb=64.0, Cm=3.70e-6),
    dict(label='old_engine_default',
         title='Old engine default  (alpha=13.3, Rb=11, Cm=5.67 uF/cm2)',
         alpha=13.3, Rb=11.0, Cm=5.67e-6),
]

def fc(Rb, Cm):
    return 1.0 / (2 * np.pi * Rb * Cm)

i_1kHz  = int(np.argmin(np.abs(FREQS - 1e3)))
i_64kHz = int(np.argmin(np.abs(FREQS - 64e3)))


# ── Compute ───────────────────────────────────────────────────────────────────

def compute_and_save():
    from engine import build_geometry, solve_Z

    def run_spectrum(alpha, Rb, Cm):
        geo = build_geometry(dict(n=N_RES, alpha=alpha+0j, Rb=Rb+0j,
                                  Cm=Cm+0j, solver='direct'))
        return solve_Z(geo, FREQS)

    print(f"Building geometry at n={N_RES} ...")
    t0 = time.time()
    build_geometry(dict(n=N_RES, solver='direct'))   # warm-up / size check
    print(f"  done in {time.time()-t0:.1f} s")
    print(f"FREQS[{i_1kHz}] = {FREQS[i_1kHz]:.0f} Hz (1 kHz)")

    save_dict = dict(freqs=FREQS, sweep_pct=SWEEP_PCT, sweep_n=SWEEP_N)

    for ref in REFS:
        a0, Rb0, Cm0 = ref['alpha'], ref['Rb'], ref['Cm']
        lbl = ref['label']
        print(f"\n{'='*55}\n{lbl}  alpha={a0} Rb={Rb0} Cm={Cm0:.2e}")

        Z0 = run_spectrum(a0, Rb0, Cm0)
        save_dict[f'{lbl}_Z0'] = Z0
        print(f"  ref: |Z|@1kHz={abs(Z0[i_1kHz])/1e3:.1f} kOhm  "
              f"phi={np.degrees(np.angle(Z0[i_1kHz])):.1f} deg")

        for pname in ('alpha', 'Rb', 'Cm'):
            p0   = dict(alpha=a0, Rb=Rb0, Cm=Cm0)[pname]
            vals = np.linspace(p0*(1-SWEEP_PCT), p0*(1+SWEEP_PCT), SWEEP_N)
            print(f"  sweeping {pname} ({SWEEP_N} pts) ...")
            Zarr = []
            for i, pval in enumerate(vals):
                a  = pval if pname == 'alpha' else a0
                Rb = pval if pname == 'Rb'    else Rb0
                Cm = pval if pname == 'Cm'    else Cm0
                Zarr.append(run_spectrum(a, Rb, Cm))
                if i % 5 == 0:
                    print(f"    [{i+1}/{SWEEP_N}] {pname}={pval:.4g}  "
                          f"|Z|@1kHz={abs(Zarr[-1][i_1kHz])/1e3:.1f} kOhm")
            save_dict[f'{lbl}_{pname}_vals'] = vals
            save_dict[f'{lbl}_{pname}_Z']    = np.array(Zarr)   # (SWEEP_N, n_freqs)

    os.makedirs(os.path.dirname(DATA_OUT), exist_ok=True)
    np.savez(DATA_OUT, **save_dict)
    print(f"\nData saved -> {DATA_OUT}")


# ── Plot ──────────────────────────────────────────────────────────────────────

def make_plots():
    if not os.path.exists(DATA_OUT):
        raise FileNotFoundError(f"Data file not found: {DATA_OUT}\n"
                                f"Run without --plot-only first.")
    D = np.load(DATA_OUT, allow_pickle=False)
    freqs = D['freqs']

    # ── GKS analytical setup ───────────────────────────────────────────────────
    from CPE_GK_Funs import GKSpectre

    ae    = np.pi * (0.035) ** 2 / 4           # cm²  electrode area (r=350 µm)
    A_cpe = 6.98e-9                             # CPE A [Ω⁻¹·s^N]  eb17 mean
    N_cpe = 0.928                               # CPE N              eb17 mean
    ZN    = ae / (A_cpe * (1j * 2 * np.pi * freqs) ** N_cpe)   # specific Ω·cm²

    def gks_eval(alpha, Rb, Cm):
        Z_cov, R, C = GKSpectre(alpha, Rb, Cm, ZN, freqs, 0.0, ae)
        return Z_cov, R, C

    # ── Reference: CH3_fit ────────────────────────────────────────────────────
    ref          = REFS[0]
    a0, Rb0, Cm0 = ref['alpha'], ref['Rb'], ref['Cm']
    f_c_ref      = fc(Rb0, Cm0)
    suptitle_ref = (f'CH3 GK fit  (\u03b1={a0}, Rb={Rb0}, '
                    f'Cm={Cm0*1e6:.2f}\u202f\u00b5F/cm\u00b2)   '
                    f'f_c={f_c_ref:.0f}\u202fHz')

    Z0_gks, R0_gks, C0_gks = gks_eval(a0, Rb0, Cm0)
    Z0_fd  = D['CH3_fit_Z0'].astype(complex)
    R0_fd  = np.real(Z0_fd)
    C0_fd  = -1.0 / (2 * np.pi * freqs * np.imag(Z0_fd))

    # ── Parameter sweep metadata ───────────────────────────────────────────────
    params_meta = [
        ('alpha', r'$\alpha$  $[\Omega^{1/2}$cm$]$',   mplcm.Reds),
        ('Rb',    r'$R_b$  $[\Omega\cdot$cm$^2]$',     mplcm.Blues),
        ('Cm',    r'$C_m$  $[\mathrm{F\,cm}^{-2}]$',  mplcm.Greens),
    ]
    pcts = np.linspace(-SWEEP_PCT * 100, SWEEP_PCT * 100, SWEEP_N)

    # ── Pre-compute GKS sweeps (analytical → fast) ────────────────────────────
    gks_sw = {}
    print('Computing GKS analytical sweeps ...')
    for pname, _, _ in params_meta:
        vals = D[f'CH3_fit_{pname}_vals']
        Zlist, Rlist, Clist = [], [], []
        for pval in vals:
            a  = pval if pname == 'alpha' else a0
            Rb = pval if pname == 'Rb'    else Rb0
            Cm = pval if pname == 'Cm'    else Cm0
            Zc, Rv, Cv = gks_eval(a, Rb, Cm)
            Zlist.append(Zc); Rlist.append(Rv); Clist.append(Cv)
        gks_sw[pname] = dict(vals=vals,
                              Z=np.array(Zlist),
                              R=np.array(Rlist),
                              C=np.array(Clist))
        print(f'  {pname}: done')

    # ── Two normalized 2×2 figures ─────────────────────────────────────────────
    from matplotlib.lines import Line2D

    fig_configs = [
        ('Zphase',
         '|Z| and Phase \u2014 normalised by reference',
         '|Z| / |Z$_0$|',
         r'$\varphi$ / $\varphi_0$'),
        ('RC',
         'R$_{eq}$ and C$_{eq}$ \u2014 normalised by reference',
         'R$_{eq}$ / R$_0$',
         'C$_{eq}$ / C$_0$'),
    ]

    for fig_key, fig_title, ylabel_top, ylabel_bot in fig_configs:
        fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True,
                                  gridspec_kw=dict(hspace=0.07, wspace=0.12))
        fig.suptitle(f'{fig_title}\n{suptitle_ref}', fontsize=10)

        axes[0, 0].set_title('Analytical GK',          fontsize=10, fontweight='bold')
        axes[0, 1].set_title(f'FD model  (n={N_RES})', fontsize=10, fontweight='bold')
        axes[0, 0].set_ylabel(ylabel_top, fontsize=9)
        axes[1, 0].set_ylabel(ylabel_bot, fontsize=9)

        for col, model in enumerate(('gks', 'fd')):
            Z0 = Z0_gks if model == 'gks' else Z0_fd
            R0 = R0_gks if model == 'gks' else R0_fd
            C0 = C0_gks if model == 'gks' else C0_fd
            ax_t = axes[0, col]
            ax_b = axes[1, col]

            for pname, _, cmap in params_meta:
                if model == 'gks':
                    Z_arr = gks_sw[pname]['Z']          # (SWEEP_N, n_freqs)
                    R_arr = gks_sw[pname]['R']
                    C_arr = gks_sw[pname]['C']
                else:
                    Z_arr = D[f'CH3_fit_{pname}_Z'].astype(complex)
                    R_arr = np.real(Z_arr)
                    C_arr = (-1.0 / (2 * np.pi
                             * freqs[np.newaxis, :] * np.imag(Z_arr)))

                colors = cmap(np.linspace(0.30, 0.95, len(pcts)))
                for i, (pct, color) in enumerate(zip(pcts, colors)):
                    lw = 1.4 if abs(pct) < 0.5 else 0.7
                    ls = '--' if abs(pct) < 0.5 else '-'
                    if fig_key == 'Zphase':
                        top_y = np.abs(Z_arr[i])    / np.abs(Z0)
                        bot_y = np.angle(Z_arr[i])  / np.angle(Z0)
                    else:
                        top_y = R_arr[i] / R0
                        bot_y = C_arr[i] / C0
                    ax_t.plot(freqs, top_y, color=color, lw=lw, ls=ls)
                    ax_b.plot(freqs, bot_y, color=color, lw=lw, ls=ls)

            # Reference horizontal
            ax_t.axhline(1.0, color='k', ls='--', lw=1.4, alpha=0.55)
            ax_b.axhline(1.0, color='k', ls='--', lw=1.4, alpha=0.55)

            # Frequency markers (1 kHz solid, 64 kHz dashed, f_c dotted orange)
            for f_mark, ls_m in [(1e3, (0, (5, 4))), (64e3, (0, (2, 3)))]:
                for ax in (ax_t, ax_b):
                    ax.axvline(f_mark, color='steelblue', lw=0.7,
                               linestyle=ls_m, alpha=0.45)
            for ax in (ax_t, ax_b):
                ax.axvline(f_c_ref, color='darkorange', lw=0.9, ls=':', alpha=0.8)
                ax.set_xscale('log')
                ax.grid(True, which='both', ls='--', alpha=0.18)
            ax_b.set_xlabel('Frequency [Hz]', fontsize=9)

        # ── Legend (top-left axes) ─────────────────────────────────────────────
        handles, labels = [], []
        for pname, plabel, cmap in params_meta:
            colors = cmap(np.linspace(0.30, 0.95, len(pcts)))
            for i, (pct, color) in enumerate(zip(pcts, colors)):
                lw = 1.4 if abs(pct) < 0.5 else 0.7
                ls = '--' if abs(pct) < 0.5 else '-'
                handles.append(Line2D([0], [0], color=color, lw=lw, linestyle=ls))
                labels.append(f'{plabel}  {pct:+.0f}%')

        axes[0, 0].legend(handles, labels, fontsize=5.5, ncol=3,
                          loc='upper left', framealpha=0.85,
                          handlelength=1.8, borderpad=0.5, labelspacing=0.25)

        out = os.path.join(_HERE, '..', 'results', f'gk_sensitivity_norm_{fig_key}.png')
        fig.savefig(out, dpi=130, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {out}')

    # ── Comparison overlay (absolute spectra, all refs) ────────────────────────
    fig, (ax_Z, ax_phi) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    fig.suptitle('GK Parameter Sets — Impedance Spectra Comparison\n'
                 'FD model n=10, ×2 for two electrodes', fontsize=11)

    colors_ref = ['#c0392b', '#2980b9', '#27ae60']
    for ref_i, col_i in zip(REFS, colors_ref):
        lbl_i = ref_i['label']
        Z_i   = D[f'{lbl_i}_Z0']
        fc_i  = fc(ref_i['Rb'], ref_i['Cm'])
        lstr  = (f"\u03b1={ref_i['alpha']}  Rb={ref_i['Rb']}  "
                 f"Cm={ref_i['Cm']:.2e}  f_c={fc_i:.0f} Hz  "
                 f"\u03c6(1kHz)={np.degrees(np.angle(Z_i[i_1kHz])):.0f}\u00b0")
        ax_Z.plot(freqs,   np.abs(Z_i) * 2,             color=col_i, lw=2, label=lstr)
        ax_phi.plot(freqs, np.degrees(np.angle(Z_i)),   color=col_i, lw=2)

    ax_Z.axhspan(38e3, 116e3, xmin=0, xmax=0.5, alpha=0.13, color='gold',
                 label='Exp. |Z| range @ 1 kHz (LF)')
    ax_phi.axhspan(-70, -48, xmin=0, xmax=0.5, alpha=0.13, color='gold',
                   label='Exp. phase range @ 1 kHz')
    for f_mark, ls_m, lab in [(1e3, '-', '1 kHz'), (64e3, '--', '64 kHz')]:
        ax_Z.axvline(f_mark,   color='grey', lw=0.8, ls=ls_m, alpha=0.6, label=lab)
        ax_phi.axvline(f_mark, color='grey', lw=0.8, ls=ls_m, alpha=0.6)

    ax_Z.set_xscale('log');  ax_Z.set_yscale('log')
    ax_phi.set_xscale('log')
    ax_Z.set_ylabel('|Z| [Ohm]  (\u00d72 electrodes)', fontsize=10)
    ax_phi.set_ylabel('Phase [deg]',                   fontsize=10)
    ax_phi.set_xlabel('Frequency [Hz]',                fontsize=10)
    for ax in [ax_Z, ax_phi]:
        ax.grid(True, which='both', ls='--', alpha=0.25)
    ax_Z.legend(fontsize=7)
    ax_phi.legend(fontsize=8, loc='lower right')
    fig.tight_layout()
    out = os.path.join(_HERE, '..', 'results', 'gk_sensitivity_comparison.png')
    fig.savefig(out, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print(f'Comparison: {out}')


# ── Entry point ───────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument('--plot-only', action='store_true',
                    help='Skip FD computation; load existing npz and replot.')
args = parser.parse_args()

os.makedirs(os.path.join(_HERE, '..', 'results'), exist_ok=True)

if not args.plot_only:
    compute_and_save()

make_plots()
print('Done.')
