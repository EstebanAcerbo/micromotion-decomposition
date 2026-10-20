"""
baseline_tabla.py
=================
Figura de validación del espectro de impedancia de referencia (baseline)
y tablas LaTeX de diferencias FD vs. GK analítico para todas las frecuencias.

Genera:
  - baseline_tabla.png         (figura de tres paneles, etiquetas en español)
  - baseline_tabla_ZPh.tex     (tabla LaTeX: diferencias en |Z| y fase)
  - baseline_tabla_RC.tex      (tabla LaTeX: diferencias en R y C)

Uso:
    python baseline_tabla.py

Rutas absolutas — el script puede ejecutarse desde cualquier directorio.
"""

import os
import sys
import numpy as np

# ── Rutas absolutas ──────────────────────────────────────────────────────────
PROJECT_ROOT     = r"C:\Users\esteb\Archivos\Papers\Micromotion_Simul"
SIM_ENGINE       = os.path.join(PROJECT_ROOT, "simulation_engine")
SIMUL_SHEET_DIR  = os.path.join(PROJECT_ROOT, "Fractals", "ModelCellShape")
EXP_DIR          = os.path.join(PROJECT_ROOT, "experimental_signal_analysis")
RESULTS_DIR      = os.path.join(PROJECT_ROOT, "results")
OUT_DIR          = 'results/'   # guardar figuras y tablas en results/

for p in (SIM_ENGINE, SIMUL_SHEET_DIR, PROJECT_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

# ── Matplotlib ───────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size":        12,
    "font.family":      "sans-serif",
    "mathtext.fontset": "stix",
})

# ── Constantes del modelo ─────────────────────────────────────────────────────
ALPHA_FIT = 19.4
RB_FIT    = 14.4       # Ω·cm²
CM_FIT    = 4.95e-6    # F/cm²
A_CPE     = 6.98e-9    # Ω⁻¹·s^N  (valor de ajuste dos electrodos, para GK analítico)
N_CPE     = 0.928
R_CONSTR  = 1000.0     # Ω por electrodo (eb17, valor analítico)


# ─────────────────────────────────────────────────────────────────────────────
# Funciones auxiliares
# ─────────────────────────────────────────────────────────────────────────────

def _load_dat_ch(path, ch_select=3):
    """Parsea archivo .dat SR530; devuelve (freqs, Z_mat) para el canal indicado."""
    Rs = 99715.0
    with open(path) as fh:
        lines = fh.readlines()
    flag1_idx = next(i for i, ln in enumerate(lines) if ln.strip() == "FLAG1")
    nf_idx    = next(i for i in range(flag1_idx + 1, len(lines)) if lines[i].strip())
    N_frec    = int(lines[nf_idx].strip())
    flag2_idx = next(i for i, ln in enumerate(lines) if ln.strip() == "FLAG2")
    data_start = next(i for i in range(flag2_idx + 1, len(lines)) if lines[i].strip())
    data   = np.loadtxt(path, skiprows=data_start)
    ch0    = data[data[:, 0] == 0][:N_frec]
    freqs  = ch0[:, 2]
    V0     = ch0[:, 3] * np.exp(1j * ch0[:, 4] * np.pi / 180.0)
    ch_r   = data[data[:, 0] == ch_select]
    n_rows = (len(ch_r) // N_frec) * N_frec
    V_all  = ch_r[:n_rows, 3] * np.exp(1j * ch_r[:n_rows, 4] * np.pi / 180.0)
    n_sw   = n_rows // N_frec
    Z_mat  = Rs * V_all.reshape(n_sw, N_frec) / (V0[np.newaxis, :] - V_all.reshape(n_sw, N_frec))
    return freqs, Z_mat


def _fmt_freq(f):
    """Formato compacto para frecuencia: '100 Hz', '1.00 kHz', '64.0 kHz', '1.00 MHz'."""
    if f >= 1e6:
        return f"{f/1e6:.2g} MHz"
    if f >= 1e3:
        return f"{f/1e3:.3g} kHz"
    return f"{f:.0f} Hz"


def _capacitance_F(Z, omega):
    """Capacitancia equivalente [F] a partir de la parte imaginaria de Z.
    C = -1 / (omega * Im{Z}).  Im{Z} < 0 para sistema capacitivo.
    """
    return -1.0 / (omega * Z.imag)


# ─────────────────────────────────────────────────────────────────────────────
# Carga de datos
# ─────────────────────────────────────────────────────────────────────────────

def load_all():
    """Carga baseline FD, curva GK, CPE y datos experimentales.

    Returns
    -------
    dict con claves:
        freqs, Z_fd_2el, Z_gk_2el, Z_cpe_2el,
        Z_fd_1el, Z_gk_1el,
        f_exp, mag_mean, mag_std, ph_mean, ph_std,
        gk_available, exp_available
    """
    # ── Baseline FD ──────────────────────────────────────────────────────────
    bl_path = os.path.join(RESULTS_DIR, "baseline.npz")
    if not os.path.isfile(bl_path):
        raise FileNotFoundError(f"No se encontró baseline.npz en {RESULTS_DIR}")
    npz   = np.load(bl_path, allow_pickle=False)
    freqs = npz["freqs"]
    Z_fd  = npz["Z"].astype(complex)
    de_cm = float(npz["de"])

    # ── GK analítico ─────────────────────────────────────────────────────────
    gk_available = False
    Z_gk_1el     = None
    try:
        from CPE_GK_Funs import GKSpectre
        ae      = np.pi * de_cm ** 2 / 4 / 2   # área por electrodo [cm²]
        omega   = 2 * np.pi * freqs
        ZN_spec = ae / (A_CPE * (1j * omega) ** N_CPE)   # Ω·cm²
        Z_gk_1el, _, _ = GKSpectre(ALPHA_FIT, RB_FIT, CM_FIT, ZN_spec, freqs, 0.0, ae)
        gk_available = True
    except ImportError as exc:
        print(f"[advertencia] GKSpectre no disponible ({exc}); se omite curva GK.")

    # ── CPE desnudo ──────────────────────────────────────────────────────────
    ae      = np.pi * de_cm ** 2 / 4 / 2
    omega   = 2 * np.pi * freqs
    ZN_spec = ae / (A_CPE * (1j * omega) ** N_CPE)
    Z_cpe   = ZN_spec / ae    # Ω (un electrodo)

    # ── Corrección dos electrodos ─────────────────────────────────────────────
    Z_fd_2el  = 2 * (Z_fd  + R_CONSTR)
    Z_cpe_2el = Z_cpe + 2 * R_CONSTR
    Z_gk_2el  = None
    if gk_available:
        Z_gk_2el = Z_gk_1el + 2 * R_CONSTR

    # ── Datos experimentales CH3 ──────────────────────────────────────────────
    exp_available = False
    f_exp = mag_mean = mag_std = ph_mean = ph_std = None
    dat_path = os.path.join(EXP_DIR, "ConfluentCH3Data.dat")
    if os.path.isfile(dat_path):
        try:
            f_exp, Z_exp_mat = _load_dat_ch(dat_path, ch_select=3)
            mag_mean = np.abs(Z_exp_mat).mean(axis=0)
            mag_std  = np.abs(Z_exp_mat).std(axis=0)
            ph_mean  = np.angle(Z_exp_mat, deg=True).mean(axis=0)
            ph_std   = np.angle(Z_exp_mat, deg=True).std(axis=0)
            exp_available = True
        except Exception as exc:
            print(f"[advertencia] No se pudo cargar ConfluentCH3Data.dat: {exc}")

    return dict(
        freqs=freqs, omega=omega,
        Z_fd_1el=Z_fd, Z_gk_1el=Z_gk_1el,
        Z_fd_2el=Z_fd_2el, Z_gk_2el=Z_gk_2el, Z_cpe_2el=Z_cpe_2el,
        f_exp=f_exp, mag_mean=mag_mean, mag_std=mag_std,
        ph_mean=ph_mean, ph_std=ph_std,
        gk_available=gk_available, exp_available=exp_available,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Métricas FD vs. GK
# ─────────────────────────────────────────────────────────────────────────────

def _compute_metrics(d):
    """RMSLE para |Z|, R, C_series y RMSE para fase (FD vs. GK, dos electrodos).

    Returns dict with keys: rmsle_Z, rmsle_R, rmsle_C, rmse_ph.
    Returns None if GK is not available.
    """
    if not d["gk_available"]:
        return None

    freqs    = d["freqs"]
    omega    = d["omega"]
    Z_fd     = d["Z_fd_2el"]
    Z_gk     = d["Z_gk_2el"]

    mag_fd = np.abs(Z_fd)
    mag_gk = np.abs(Z_gk)
    R_fd   = Z_fd.real
    R_gk   = Z_gk.real
    C_fd   = -1.0 / (omega * Z_fd.imag)
    C_gk   = -1.0 / (omega * Z_gk.imag)
    ph_fd  = np.angle(Z_fd, deg=True)
    ph_gk  = np.angle(Z_gk, deg=True)

    # Guard against non-positive values before log (should not occur for this system)
    mask = (mag_fd > 0) & (mag_gk > 0) & (R_fd > 0) & (R_gk > 0) & (C_fd > 0) & (C_gk > 0)

    rmsle_Z  = np.sqrt(np.mean((np.log(mag_fd[mask]) - np.log(mag_gk[mask])) ** 2))
    rmsle_R  = np.sqrt(np.mean((np.log(R_fd[mask])   - np.log(R_gk[mask]))   ** 2))
    rmsle_C  = np.sqrt(np.mean((np.log(C_fd[mask])   - np.log(C_gk[mask]))   ** 2))
    rmse_ph  = np.sqrt(np.mean((ph_fd - ph_gk) ** 2))

    return dict(rmsle_Z=rmsle_Z, rmsle_R=rmsle_R, rmsle_C=rmsle_C, rmse_ph=rmse_ph)


# ─────────────────────────────────────────────────────────────────────────────
# Figura
# ─────────────────────────────────────────────────────────────────────────────

def make_figure(d):
    """Genera la figura de tres paneles (geometría | |Z| | fase) en español."""
    freqs        = d["freqs"]
    Z_fd_2el     = d["Z_fd_2el"]
    Z_gk_2el     = d["Z_gk_2el"]
    Z_cpe_2el    = d["Z_cpe_2el"]
    gk_available = d["gk_available"]
    exp_available = d["exp_available"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    ax_geo, ax_mag, ax_ph = axes

    # ── Panel geometría ───────────────────────────────────────────────────────
    geo_available = False
    try:
        from engine import build_geometry
        from SimulSheet_Funcs import Map_Electrode
        geo_available = True
    except ImportError as exc:
        print(f"[advertencia] No se pudo importar engine ({exc}); panel de geometría omitido.")

    if geo_available:
        try:
            geo  = build_geometry()
            Cult = geo["Culture_vector"]
            Sub  = geo["Substrate_vector"]
            p_g, d_g, side = geo["p"], geo["d"], geo["side"]

            def _combined(C, S):
                return 2 * C.astype(np.int32) + (S != 0).astype(np.int32)

            Map_Electrode(ax_geo, _combined(Cult, Sub), p_g["de"], d_g, side)
            leg = ax_geo.get_legend()
            if leg is not None:
                leg.remove()
            ax_geo.set_title("Dominio de simulación", fontsize=15)
            ax_geo.set_xlabel(r"Posición [$\mu$m]", fontsize=14)
            ax_geo.set_ylabel(r"Posición [$\mu$m]", fontsize=14)
            ax_geo.tick_params(labelsize=9)
        except Exception as exc:
            ax_geo.set_title(f"Geometría no disponible\n({exc})", fontsize=13)
            ax_geo.axis("off")
    else:
        ax_geo.set_title("Geometría no disponible\n(engine no importado)", fontsize=13)
        ax_geo.axis("off")

    # ── Ejes compartidos ──────────────────────────────────────────────────────
    for ax in (ax_mag, ax_ph):
        ax.set_xscale("log")
        ax.set_xlabel("Frecuencia [Hz]", fontsize=14)
        ax.grid(True, which="both", alpha=0.18)
        ax.tick_params(labelsize=10)

    # ── Panel |Z| ─────────────────────────────────────────────────────────────
    ax_mag.set_yscale("log")
    ax_mag.plot(freqs, np.abs(Z_fd_2el), color="tab:blue", lw=1.6,
                label=r"Dif. Finitas $\times2$ + $R_\mathrm{constr}$")
    if gk_available:
        ax_mag.plot(freqs, np.abs(Z_gk_2el), color="tab:orange", lw=1.4, ls="--",
                    label=r"GK analítico $\times2$ + $R_\mathrm{constr}$")
    ax_mag.plot(freqs, np.abs(Z_cpe_2el), color="tab:grey", lw=1.2, ls=":",
                label=r"CPE puro $\times2$ + $R_\mathrm{constr}$")
    if exp_available:
        ax_mag.errorbar(d["f_exp"], d["mag_mean"], yerr=d["mag_std"],
                        fmt="o", color="tab:red", ms=3, lw=1.2, capsize=3,
                        label="Experimento (cultivo confluente)")
    ax_mag.set_xlim(freqs.min(), freqs.max())
    mag_fd = np.abs(Z_fd_2el)
    ax_mag.set_ylim(mag_fd.min() * 0.5, mag_fd.max() * 2.0)
    ax_mag.set_ylabel(r"$|Z|$ [$\Omega$]", fontsize=14)
    ax_mag.set_title("Magnitud de impedancia", fontsize=15)

    # ── Cuadro de métricas (inferior izquierdo) ───────────────────────────────
    metrics = _compute_metrics(d)
    if metrics is not None:
        txt = (
            r"$\mathbf{DF\ vs.\ GK}$" + "\n"
            fr"RMSLE $|Z|$  = {metrics['rmsle_Z']:.4f}" + "\n"
            fr"RMSLE $R$    = {metrics['rmsle_R']:.4f}" + "\n"
            fr"RMSLE $C_s$  = {metrics['rmsle_C']:.4f}" + "\n"
            fr"RMSE $\varphi$    = {metrics['rmse_ph']:.2f}°"
        )
        ax_mag.text(
            0.03, 0.03, txt,
            transform=ax_mag.transAxes,
            fontsize=9, verticalalignment="bottom", horizontalalignment="left",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.82, edgecolor="#aaa"),
            family="monospace",
        )

    # ── Panel fase ────────────────────────────────────────────────────────────
    ax_ph.plot(freqs, np.angle(Z_fd_2el, deg=True), color="tab:blue", lw=1.6,
               label=r"Dif. Finitas + $R_\mathrm{constr}$")
    if gk_available:
        ax_ph.plot(freqs, np.angle(Z_gk_2el, deg=True), color="tab:orange", lw=1.4, ls="--",
                   label=r"GK analítico + $R_\mathrm{constr}$")
    ax_ph.plot(freqs, np.angle(Z_cpe_2el, deg=True), color="tab:grey", lw=1.2, ls=":",
               label=r"CPE puro + $R_\mathrm{constr}$")
    if exp_available:
        ax_ph.errorbar(d["f_exp"], d["ph_mean"], yerr=d["ph_std"],
                       fmt="o", color="tab:red", ms=3, lw=1.2, capsize=3,
                       label="Experimento (cultivo confluente)")
    ax_ph.set_xlim(freqs.min(), freqs.max())
    ph_fd = np.angle(Z_fd_2el, deg=True)
    ax_ph.set_ylim(ph_fd.min() - 5, ph_fd.max() + 5)
    ax_ph.set_ylabel(r"Fase [$^\circ$]", fontsize=14)
    ax_ph.set_title("Fase de la impedancia", fontsize=15)
    ax_ph.legend(fontsize=11, loc="best")

    fig.suptitle(
        "Espectro de referencia — Validación del motor FD frente al modelo GK analítico y experimento",
        fontsize=14,
    )
    out_png = os.path.join(OUT_DIR, "baseline_tabla.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  figura guardada -> {out_png}")


# ─────────────────────────────────────────────────────────────────────────────
# Tablas LaTeX
# ─────────────────────────────────────────────────────────────────────────────

def _pct(a, b):
    """Diferencia porcentual (a - b) / |b| × 100."""
    return (a - b) / abs(b) * 100.0


def make_table_ZPh(d, save=True):
    """Tabla 1: diferencias en |Z| y fase (DF vs. GK analítico) para todas las frecuencias.

    Columnas: Frecuencia | |Z|_DF [Ω] | |Z|_GK [Ω] | Δ|Z| [%] | φ_DF [°] | φ_GK [°] | Δφ [°]
    """
    if not d["gk_available"]:
        print("[tabla Z/fase] GK no disponible; tabla omitida.")
        return

    freqs    = d["freqs"]
    Z_fd_2el = d["Z_fd_2el"]
    Z_gk_2el = d["Z_gk_2el"]

    mag_fd = np.abs(Z_fd_2el)
    mag_gk = np.abs(Z_gk_2el)
    ph_fd  = np.angle(Z_fd_2el, deg=True)
    ph_gk  = np.angle(Z_gk_2el, deg=True)
    dpct   = _pct(mag_fd, mag_gk)
    dph    = ph_fd - ph_gk

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{Diferencias en magnitud e impedancia entre el motor DF y el modelo GK analítico para cada frecuencia.}")
    lines.append(r"\label{tab:diff_ZPh}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lrrrrrrr}")
    lines.append(r"\toprule")
    lines.append(
        r"Frecuencia & $|Z|_{\mathrm{DF}}$ [$\Omega$] & $|Z|_{\mathrm{GK}}$ [$\Omega$] "
        r"& $\Delta|Z|$ [\%] & $\varphi_{\mathrm{DF}}$ [$^\circ$] "
        r"& $\varphi_{\mathrm{GK}}$ [$^\circ$] & $\Delta\varphi$ [$^\circ$] \\"
    )
    lines.append(r"\midrule")

    for i, f in enumerate(freqs):
        row = (
            f"{_fmt_freq(f):<10s} & "
            f"{mag_fd[i]:.0f} & "
            f"{mag_gk[i]:.0f} & "
            f"{dpct[i]:+.2f} & "
            f"{ph_fd[i]:.2f} & "
            f"{ph_gk[i]:.2f} & "
            f"{dph[i]:+.2f} \\\\"
        )
        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    tex = "\n".join(lines)
    print("\n% ── Tabla 1: diferencias |Z| y fase ──────────────────────────────────────────")
    print(tex)

    if save:
        out = os.path.join(OUT_DIR, "baseline_tabla_ZPh.tex")
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(tex + "\n")
        print(f"  tabla guardada -> {out}")


def make_table_RC(d, save=True):
    """Tabla 2: diferencias en R (parte real) y C (capacitancia equivalente) DF vs. GK.

    C_eq = -1 / (omega × Im{Z})    [F]
    Columnas: Frecuencia | R_DF [Ω] | R_GK [Ω] | ΔR [%] | C_DF [nF] | C_GK [nF] | ΔC [%]
    """
    if not d["gk_available"]:
        print("[tabla R/C] GK no disponible; tabla omitida.")
        return

    freqs    = d["freqs"]
    omega    = d["omega"]
    Z_fd_2el = d["Z_fd_2el"]
    Z_gk_2el = d["Z_gk_2el"]

    R_fd = Z_fd_2el.real
    R_gk = Z_gk_2el.real
    C_fd = _capacitance_F(Z_fd_2el, omega)   # [F]
    C_gk = _capacitance_F(Z_gk_2el, omega)

    # Convertir a nF para la tabla
    C_fd_nF = C_fd * 1e9
    C_gk_nF = C_gk * 1e9

    dR_pct = _pct(R_fd, R_gk)
    dC_pct = _pct(C_fd, C_gk)

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{Diferencias en resistencia $R$ y capacitancia equivalente $C$ entre el motor DF y el modelo GK analítico para cada frecuencia.}")
    lines.append(r"\label{tab:diff_RC}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lrrrrrr}")
    lines.append(r"\toprule")
    lines.append(
        r"Frecuencia & $R_{\mathrm{DF}}$ [$\Omega$] & $R_{\mathrm{GK}}$ [$\Omega$] "
        r"& $\Delta R$ [\%] & $C_{\mathrm{DF}}$ [nF] & $C_{\mathrm{GK}}$ [nF] & $\Delta C$ [\%] \\"
    )
    lines.append(r"\midrule")

    for i, f in enumerate(freqs):
        row = (
            f"{_fmt_freq(f):<10s} & "
            f"{R_fd[i]:.1f} & "
            f"{R_gk[i]:.1f} & "
            f"{dR_pct[i]:+.2f} & "
            f"{C_fd_nF[i]:.4f} & "
            f"{C_gk_nF[i]:.4f} & "
            f"{dC_pct[i]:+.2f} \\\\"
        )
        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    tex = "\n".join(lines)
    print("\n% ── Tabla 2: diferencias R y C ───────────────────────────────────────────────")
    print(tex)

    if save:
        out = os.path.join(OUT_DIR, "baseline_tabla_RC.tex")
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(tex + "\n")
        print(f"  tabla guardada -> {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Punto de entrada
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Cargando datos ...")
    data = load_all()
    print("Generando figura ...")
    make_figure(data)
    print("Generando tablas LaTeX ...")
    make_table_ZPh(data, save=True)
    make_table_RC(data, save=True)
    print("Listo.")
