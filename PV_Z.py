#!/usr/bin/env python3
import h5py
import numpy as np
import matplotlib.pyplot as plt
import glob
import cantera as ct
import os

# ----------------------------
# Cantera setup
# ----------------------------
fuel_yaml = "h2-burke.yaml"
fuel_comp = "H2:1.0"
ox_comp   = "O2:0.082, AR:0.918"

gas = ct.Solution(fuel_yaml)

fuel_gas = ct.Solution(fuel_yaml)
ox_gas   = ct.Solution(fuel_yaml)

fuel_gas.TPX = 300, ct.one_atm, fuel_comp
ox_gas.TPX   = 300, ct.one_atm, ox_comp

fuel_Y = fuel_gas.Y
ox_Y   = ox_gas.Y

# ----------------------------
# Species indices for PV
# ----------------------------
species_order = ["H","H2","O","OH","H2O","O2","HO2","H2O2","N2","AR","HE"]
idx_H    = species_order.index("H")
idx_H2O  = species_order.index("H2O")

# ----------------------------
# Load all prof*.h5 files
# ----------------------------
files = sorted(glob.glob("prof*.h5"))
os.makedirs("txt_PV_Z", exist_ok=True)

all_profiles_Z = []
all_profiles_PV = []
all_profiles_T  = []
labels = []

for fname in files:
    with h5py.File(fname,"r") as f:
        x = f["x"][()]
        Y = f["Y"][()]         # shape: (Ns, Nx)
        T = f["T"][()]         # temperature array Nx
        Ns, Nx = Y.shape

        Z = np.zeros(Nx)
        PV = np.zeros(Nx)

        for i in range(Nx):
            gas.Y = Y[:,i]
            Z[i] = gas.mixture_fraction(
                fuel=fuel_Y,
                oxidizer=ox_Y,
                basis='mass',
                element='Bilger'
            )
            PV[i] = Y[idx_H2O, i] - 10.0 * Y[idx_H, i]

        Z = np.clip(Z, 0.0, 1.0)

        # Save txt output
        outname = f"txt_PV_Z/{os.path.basename(fname).replace('.h5','')}_PV_Z.txt"
        np.savetxt(
            outname,
            np.column_stack([x, Z, PV]),
            header="x    Z    PV",
            fmt="%.8e"
        )
        print("Saved:", outname)

        all_profiles_Z.append((x, Z))
        all_profiles_PV.append((x, PV))
        all_profiles_T.append((x, T))
        labels.append(os.path.basename(fname))

# ----------------------------
# Plot PV vs Z
# ----------------------------
plt.figure(figsize=(10,6))
for (x,Z), (_,PV), label in zip(all_profiles_Z, all_profiles_PV, labels):
    plt.plot(Z, PV, label=label)

plt.xlabel("Z (mixture fraction)")
plt.ylabel("PV = Y_H2O - 10 Y_H")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("PV_vs_Z_profiles.png", dpi=200)
plt.show()
print("Saved plot: PV_vs_Z_profiles.png")

# ----------------------------
# NEW SECTION: Plot T vs Z
# ----------------------------
plt.figure(figsize=(10,6))

for (x,Z), (_,T), label in zip(all_profiles_Z, all_profiles_T, labels):
    plt.plot(Z, T, label=label)

plt.xlabel("Z (mixture fraction)")
plt.ylabel("Temperature [K]")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("T_vs_Z_profiles.png", dpi=200)
plt.show()

print("Saved plot: T_vs_Z_profiles.png")
