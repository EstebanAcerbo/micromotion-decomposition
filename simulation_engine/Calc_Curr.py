import numpy as np
def In_value(V, d: float, Zn: complex | np.ndarray, VN: complex, Substrate_vector):
    """
    Calculate total current through electrode substrate.
    Args:
    V: Voltage vector(s)
        - Single: shape (dim,) 
        - Batch: shape (num_freq, dim)
    d: Discretization length [cm]
    Zn: Electrode specific impedance [Ω·cm²]
        - Single: scalar complex
        - Batch: array of shape (num_freq,)
    VN: Electrode potential [V]
    Substrate_vector: Boolean array indicating electrode positions 
                        (1 = electrode, 0 = insulator)
    Now supports:
    - Single frequency: V is 1D array, Zn is scalar → returns scalar
    - Batch: V is 2D array (freq × space), Zn is 1D array → returns 1D array
    """
    electrode_idx = np.where(Substrate_vector == 1)[0]
    if len(electrode_idx) == 0:
        raise ValueError("No electrode points found in Substrate_vector")
    # Handle both single and batch cases
    if V.ndim == 1:
        # Single frequency
        return np.sum(VN - V[electrode_idx]) * d**2 / Zn
    else:
        # Batch (V is 2D: [num_freq, dim])
        return np.sum(VN - V[:, electrode_idx], axis=1) * d**2 / Zn
