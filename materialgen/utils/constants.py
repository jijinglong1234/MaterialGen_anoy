"""
Physical constants and unit conversions for crystal structure generation.

All constants in SI-compatible units with eV/Å as primary energy/length units
(standard in atomistic simulations).
"""

# === Fundamental Constants ===
KB = 8.617333262145e-5            # Boltzmann constant (eV/K)
KB_SI = 1.380649e-23              # Boltzmann constant (J/K)
EV_TO_J = 1.602176634e-19         # eV to Joules
J_TO_EV = 1.0 / EV_TO_J           # Joules to eV
ANGSTROM_TO_M = 1.0e-10           # Å to meters
M_TO_ANGSTROM = 1.0e10

# === Hartree units ===
HA_TO_EV = 27.211386245988        # Hartree to eV
EV_TO_HA = 1.0 / HA_TO_EV
RY_TO_EV = 13.605693122994        # Rydberg to eV

# === Inverse temperature at common temperatures ===
def beta_from_temperature(T: float) -> float:
    """β = 1/(kB·T) in eV⁻¹."""
    return 1.0 / (KB * T)

def temperature_from_beta(beta: float) -> float:
    """T = 1/(kB·β) in Kelvin."""
    return 1.0 / (KB * beta)

# Precomputed β values
BETA_300K = beta_from_temperature(300)    # ≈ 38.68 eV⁻¹
BETA_500K = beta_from_temperature(500)    # ≈ 23.21 eV⁻¹
BETA_1000K = beta_from_temperature(1000)  # ≈ 11.60 eV⁻¹
BETA_2000K = beta_from_temperature(2000)  # ≈ 5.80 eV⁻¹

# === Force unit conversions ===
# Forces in MLIPs are typically eV/Å
EV_PER_A_TO_N = EV_TO_J / ANGSTROM_TO_M  # 1 eV/Å ≈ 1.602e-9 N
N_TO_EV_PER_A = 1.0 / EV_PER_A_TO_N

# === Pressure / Stress unit conversions ===
# Stress in MLIPs is typically eV/Å³
EV_PER_A3_TO_GPA = EV_TO_J / (ANGSTROM_TO_M**3) * 1e-9  # 1 eV/Å³ ≈ 160.22 GPa
GPA_TO_EV_PER_A3 = 1.0 / EV_PER_A3_TO_GPA

# === Common covalent radii (Å) for overlap detection ===
# From Cordero et al., Dalton Trans. (2008)
COVALENT_RADII = {
    1: 0.31,  2: 0.28,  3: 1.28,  4: 0.96,  5: 0.84,
    6: 0.76,  7: 0.71,  8: 0.66,  9: 0.57, 10: 0.58,
    11: 1.66, 12: 1.41, 13: 1.21, 14: 1.11, 15: 1.07,
    16: 1.05, 17: 1.02, 18: 1.06, 19: 2.03, 20: 1.76,
    21: 1.70, 22: 1.60, 23: 1.53, 24: 1.39, 25: 1.39,
    26: 1.32, 27: 1.26, 28: 1.24, 29: 1.32, 30: 1.22,
    31: 1.22, 32: 1.20, 33: 1.19, 34: 1.20, 35: 1.20,
    36: 1.16, 37: 2.20, 38: 1.95, 39: 1.90, 40: 1.75,
    41: 1.64, 42: 1.54, 43: 1.47, 44: 1.46, 45: 1.42,
    46: 1.39, 47: 1.45, 48: 1.44, 49: 1.42, 50: 1.39,
    51: 1.39, 52: 1.38, 53: 1.39, 54: 1.40, 55: 2.44,
    56: 2.15, 57: 2.07, 72: 1.75, 73: 1.70, 74: 1.62,
    75: 1.51, 76: 1.44, 77: 1.41, 78: 1.36, 79: 1.36,
    80: 1.32, 81: 1.45, 82: 1.46, 83: 1.48,
}
