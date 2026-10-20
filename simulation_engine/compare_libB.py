import numpy as np

PATH_WRONG = "C:/Users/esteb/Archivos/Workbench/Micromotion_Simul/results_wrong_A/library_B_rb.npz"
PATH_NEW   = "C:/Users/esteb/Archivos/Workbench/Micromotion_Simul/results/library_B_rb.npz"

def load_and_inspect(path):
    data = np.load(path, allow_pickle=True)
    print(f"\n--- Keys in {path.split('/')[-2]}/{path.split('/')[-1]} ---")
    for k in data.files:
        v = data[k]
        print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
    return data

dw = load_and_inspect(PATH_WRONG)
dn = load_and_inspect(PATH_NEW)

# Extract arrays — try common key names
def get_array(data, *candidates):
    for key in candidates:
        if key in data.files:
            return data[key]
    raise KeyError(f"None of {candidates} found in {data.files}")

freqs_w = get_array(dw, 'freqs', 'freq', 'frequencies', 'f')
freqs_n = get_array(dn, 'freqs', 'freq', 'frequencies', 'f')

Z_w = get_array(dw, 'Z_library', 'Z', 'Z_all', 'Zlib')
Z_n = get_array(dn, 'Z_library', 'Z', 'Z_all', 'Zlib')

Z0_w = get_array(dw, 'Z_baseline', 'Z0', 'Z_base', 'baseline')
Z0_n = get_array(dn, 'Z_baseline', 'Z0', 'Z_base', 'baseline')

print(f"\nFrequency array shapes: wrong={freqs_w.shape}, new={freqs_n.shape}")
print(f"Z_library shapes:       wrong={Z_w.shape}, new={Z_n.shape}")
print(f"Z_baseline shapes:      wrong={Z0_w.shape}, new={Z0_n.shape}")

# Find index closest to 1 kHz
TARGET_HZ = 1000.0
idx_w = int(np.argmin(np.abs(freqs_w - TARGET_HZ)))
idx_n = int(np.argmin(np.abs(freqs_n - TARGET_HZ)))
print(f"\nClosest freq index to 1 kHz:")
print(f"  wrong file: idx={idx_w}, f={freqs_w[idx_w]:.2f} Hz")
print(f"  new   file: idx={idx_n}, f={freqs_n[idx_n]:.2f} Hz")

# Z_library may be (n_modes, n_freqs) or (n_freqs, n_modes) — normalise to (n_modes, n_freqs)
# Z_baseline may be (n_freqs,) or (1, n_freqs)
def orient(Z, Z0):
    """Return Z shaped (n_modes, n_freqs) and Z0 shaped (n_freqs,)."""
    if Z.ndim == 1:
        Z = Z[np.newaxis, :]
    if Z.shape[0] == Z0.shape[-1] and Z.shape[1] != Z0.shape[-1]:
        Z = Z.T   # was (n_freqs, n_modes)
    Z0 = Z0.ravel()
    return Z, Z0

Z_w, Z0_w = orient(Z_w, Z0_w)
Z_n, Z0_n = orient(Z_n, Z0_n)

n_modes_w = Z_w.shape[0]
n_modes_n = Z_n.shape[0]
print(f"\nNumber of modes: wrong={n_modes_w}, new={n_modes_n}")

# dZ at 1 kHz for every mode
dZ_w = Z_w[:, idx_w] - Z0_w[idx_w]
dZ_n = Z_n[:, idx_n] - Z0_n[idx_n]

dR_w = dZ_w.real
dX_w = dZ_w.imag
dR_n = dZ_n.real
dX_n = dZ_n.imag

n_modes = min(n_modes_w, n_modes_n)

# Per-mode table
header = (f"{'mode':>5} | {'dR_wrong':>14} {'dR_correct':>14} {'|dR_diff|':>12} {'%dR_diff':>10} |"
          f" {'dX_wrong':>14} {'dX_correct':>14} {'|dX_diff|':>12} {'%dX_diff':>10}")
print(f"\n{header}")
print("-" * len(header))

abs_dR_diff  = np.abs(dR_w[:n_modes] - dR_n[:n_modes])
abs_dX_diff  = np.abs(dX_w[:n_modes] - dX_n[:n_modes])

# % diff relative to new (correct) value; guard against near-zero
eps = 1e-30
pct_dR = 100.0 * abs_dR_diff / (np.abs(dR_n[:n_modes]) + eps)
pct_dX = 100.0 * abs_dX_diff / (np.abs(dX_n[:n_modes]) + eps)

for i in range(n_modes):
    print(f"{i:>5} | {dR_w[i]:>14.6g} {dR_n[i]:>14.6g} {abs_dR_diff[i]:>12.4g} {pct_dR[i]:>10.2f} |"
          f" {dX_w[i]:>14.6g} {dX_n[i]:>14.6g} {abs_dX_diff[i]:>12.4g} {pct_dX[i]:>10.2f}")

# Summary stats
print("\n=== Summary statistics (over all modes, at 1 kHz) ===")
print(f"{'Metric':<30} {'dR':>14} {'dX':>14}")
print("-" * 60)
print(f"{'Mean |diff|':<30} {np.mean(abs_dR_diff):>14.6g} {np.mean(abs_dX_diff):>14.6g}")
print(f"{'Max  |diff|':<30} {np.max(abs_dR_diff):>14.6g} {np.max(abs_dX_diff):>14.6g}")
print(f"{'Mean % diff':<30} {np.mean(pct_dR):>14.2f} {np.mean(pct_dX):>14.2f}")
print(f"{'Max  % diff':<30} {np.max(pct_dR):>14.2f} {np.max(pct_dX):>14.2f}")
