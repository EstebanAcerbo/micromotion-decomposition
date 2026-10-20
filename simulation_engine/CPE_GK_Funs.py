import numpy as np
from scipy.special import iv


def Z_CPE_B(f, A: float, B: float, n: float):
    """Calculate CPE (Constant Phase Element) impedance with constriction resistance.

    Inputs:
        f,        [Hz]      ; frequency Vector.
        A: float, [Ohm⁻¹·sⁿ]; A constant in CPE model.
        B: float, [Ohm]     ; Constriction resitance of the system.
        n: float, []        ; n CPE exponent (0 < n < 1, unitless).

    Returns:
        Z = 1/(A*(jw)^n)+B where w = 2πf, with f being frequency vector.
    """
    return 1/(A*(1j*f*2*np.pi)**n)+B

def GKSpectre(
    alpha: complex, # [(Ohm.cm)^.5]; Alpha = rc*sqrt{rho/h} with h the substrate-basal membrane height.
    Rb: complex,    # [Ohm.cm^2]  ; Intecellular union resistivity constant.
    Cm: complex,    # [cm2/F]   ; Cell membrane capacitance constant.
    ZN: complex,    # [Ohm.cm^2]  ; vector of specific electrode impedance.
    frec,           # [Hz]        ; frec vector.
    R_const: float, # [Ohm]       ; Constriction resistance.
    ae: float,      # [cm^2]      ; Electrode surface.
):
    """Calculate covered microelectrode impedance using GK model using three biophysical parameters.

    Inputs:
        alpha: complex, [(Ohm.cm)^.5]; Alpha = rc*sqrt{rho/h} with h the substrate-basal membrane height.
        Rb: complex, [Ohm.cm^2]; Intecellular union resistivity constant.
        Cm: complex, [cm2/F]; Cell membrane capacitance constant.
        naked_spec: complex, [Ohm.cm^2]; vector of specific electrode impedance.
        frec [Hz]; frecuency vector.
        R_const: float, [Ohm]; Constriction resistance.
        ae: float, [cm^2]; Electrode surface.

    Returns:
        (Z_cov [Ohm·cm²], R [Ohm], C [F]).
    """
    ZM = -1j/(2*np.pi*(Cm/2)*frec)  # Specific impedance of the naked microelectrode and cells
    gamma_rc = alpha*np.sqrt(1/ZN+1/ZM)  # Gamma.rc
    I_0 = iv(0, gamma_rc)
    I_1 = iv(1, gamma_rc)  # Modified Bessel function of first kind, order 0 and 1
    Z_cov = ((ZN*(ZN+ZM))/(ZN+ZM/(0.5*gamma_rc*I_0/I_1+Rb*(1/ZN+1/ZM))))/ae  # Specific impedance of the covered microelectrode
    R = np.real(Z_cov)+R_const
    C = -1/(2*np.pi*frec*np.imag(Z_cov))
    return Z_cov, R, C
