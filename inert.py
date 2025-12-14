#!/usr/bin/env python3
"""
High-resolution inert counterflow diffusion flame setup
covering full mixture fraction range (0 ≤ Z ≤ ~0.2).

Output file: counterflow_inert_solution.npz
"""

import cantera as ct
import numpy as np
import matplotlib.pyplot as plt

# --- High precision dtype ---
dtype = np.float128 if hasattr(np, "float128") else np.float64

# -----------------------------------------------------------------------------
# 1. Create gas and disable reactions (inert mode)
# -----------------------------------------------------------------------------
gas = ct.Solution('h2-burke.yaml')

# Disable chemistry (set all reaction multipliers to zero)
if hasattr(gas, "set_multiplier"):
    gas.set_multiplier(0.0)
else:
    gas.setMultiplier(0.0)

gas.transport_model = 'Multi'

# -----------------------------------------------------------------------------
# 2. Counterflow configuration
# -----------------------------------------------------------------------------
width = 0.06  # extended domain (m)
f = ct.CounterflowDiffusionFlame(gas, width=width)
f.transport_model = 'Multi'
f.soret_enabled = False

# Inlet conditions
f.fuel_inlet.X = 'H2:1'
f.fuel_inlet.T = 300.0
f.fuel_inlet.mdot = 0.15      # balanced but slightly weaker jet

f.oxidizer_inlet.X = "O2:0.082, AR:0.918"
f.oxidizer_inlet.T = 1300.0
f.oxidizer_inlet.mdot = 1.0   # stronger oxidizer → larger lean region

f.P = 1.0e5  # Pa

# -----------------------------------------------------------------------------
# 3. Solver setup and refinement
# -----------------------------------------------------------------------------
f.set_refine_criteria(ratio=2.0, slope=0.02, curve=0.02)
f.max_grid_points = 8000

# Initial solve
f.solve()

# Target strain rate loop
a_target = dtype(100.0)
tol = 1e-8
maxit = 30

for n in range(maxit):
    a_curr = dtype(f.strain_rate('max'))
    err = (a_curr - a_target) / a_target
    if abs(err) < tol:
        print(f"✅ Converged: a = {a_curr:.10f} 1/s")
        break
    factor = (a_target / max(a_curr, dtype(1e-30))) ** dtype(0.5)
    f.fuel_inlet.mdot *= float(factor)
    f.oxidizer_inlet.mdot *= float(factor)
    try:
        f.solve()
    except ct.CanteraError:
        factor = 1.0 + 0.5*(factor - 1.0)
        f.fuel_inlet.mdot *= float(factor)
        f.oxidizer_inlet.mdot *= float(factor)
        f.solve()

print("Chemistry multipliers set to zero (inert mixing).")
print(f"Transport model: {f.transport_model}")
print(f"Soret enabled?: {f.soret_enabled}")

# -----------------------------------------------------------------------------
# 4. Compute mixture fraction (H/O basis)
# -----------------------------------------------------------------------------
MH = dtype(gas.atomic_weight('H'))
MO = dtype(gas.atomic_weight('O'))

def elemental_mass_fractions(Y):
    gas.Y = Y
    return dtype(gas.elemental_mass_fraction('H')), dtype(gas.elemental_mass_fraction('O'))

# Reference states
gas.X = 'H2:1'
YH1, YO1 = elemental_mass_fractions(gas.Y)
gas.X = "O2:0.082, AR:0.918"
YH2, YO2 = elemental_mass_fractions(gas.Y)

def mixture_fraction(Y):
    YH, YO = elemental_mass_fractions(Y)
    num = 0.5*(YH - YH2)/MH - (YO - YO2)/MO
    den = 0.5*(YH1 - YH2)/MH - (YO1 - YO2)/MO
    return num / (den + dtype(1e-30))

# Z profile
Z = np.array([mixture_fraction(Y) for Y in f.Y.T], dtype=dtype)
T = np.array(f.T, dtype=dtype)

# Stoichiometric mixture fraction
num_st = 0.5*(0.0 - YH2)/MH - (0.0 - YO2)/MO
den_st = 0.5*(YH1 - YH2)/MH - (YO1 - YO2)/MO
Zst = num_st / (den_st + dtype(1e-30))
print(f"Z_st = {Zst:.12f}")

# -----------------------------------------------------------------------------
# 5. Diagnostics
# -----------------------------------------------------------------------------
print(f"Grid points: {len(Z)}")
print(f"Z range: {Z.min():.6f} → {Z.max():.6f}")

# -----------------------------------------------------------------------------
# 6. Save data
# -----------------------------------------------------------------------------
z_grid = np.array(f.grid, dtype=dtype)
P_grid = np.full_like(z_grid, f.P)
Y_H2_grid = np.array(f.Y[gas.species_index('H2'), :], dtype=dtype)
Y_O2_grid = np.array(f.Y[gas.species_index('O2'), :], dtype=dtype)
Y_AR_grid = np.array(f.Y[gas.species_index('AR'), :], dtype=dtype)

np.savez('counterflow_inert_solution.npz',
         z=z_grid,
         P=P_grid,
         T=T,
         Y_H2=Y_H2_grid,
         Y_O2=Y_O2_grid,
         Y_AR=Y_AR_grid,
         Z=Z)

print("✅ Inert counterflow data saved to 'counterflow_inert_solution.npz'.")

# -----------------------------------------------------------------------------
# 7. Quick check: Temperature and Species vs Z
# -----------------------------------------------------------------------------
plt.figure(figsize=(7,5))
plt.plot(Z, T, '-', lw=2, label="T(Z) inert")
plt.axvline(float(Zst), color='k', ls='--', label=f"Zst = {Zst:.5f}")
plt.xlabel("Mixture fraction Z")
plt.ylabel("Temperature [K]")
plt.title(f"Inert counterflow mixing (a ≈ {f.strain_rate('max'):.1f} 1/s)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Inert_T_vs_Z.png", dpi=300)
plt.show()

plt.figure(figsize=(8,6))
plt.plot(Z, Y_H2_grid, label="Y_H2")
plt.plot(Z, Y_O2_grid, label="Y_O2")
plt.plot(Z, Y_AR_grid, label="Y_AR")
plt.xlabel("Mixture fraction Z")
plt.ylabel("Mass fraction")
plt.legend()
plt.title("Species vs Mixture fraction (inert)")
plt.grid(True)
plt.tight_layout()
plt.savefig("Inert_species_vs_Z.png", dpi=300)
plt.show()
