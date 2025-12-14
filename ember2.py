#!/usr/bin/python
"""
Compare reactive and non-reactive opposed-jet hydrogen flame.
Runs both cases, plots all species, temperature, axial velocity, and HRR for comparison.
"""

from ember import *
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import os
import numpy as np

output = 'run/ex_diffusion_compare_allspecies'
os.makedirs(output, exist_ok=True)

# Function to run a simulation given chemistry settings
def run_sim(case_name, chemistry_enabled=True):

    case_dir = os.path.join(output, case_name)
    os.makedirs(case_dir, exist_ok=True)

    # --------------------------------------------------
    # Path to Cantera-generated inert profile
    # --------------------------------------------------
    cantera_profile = 'counterflow_profile.h5'

    if not os.path.isfile(cantera_profile):
        raise FileNotFoundError(
            f"Cantera inert profile not found: {cantera_profile}"
        )

    # --------------------------------------------------
    # NON-REACTIVE CASE:
    # Just load Cantera profile, do NOT run Ember
    # --------------------------------------------------
    if not chemistry_enabled:
        struct = utils.load(cantera_profile)
        return struct

    # --------------------------------------------------
    # REACTIVE CASE:
    # Use Cantera profile as restartFile
    # --------------------------------------------------
    conf = Config(
        Paths(outputDir=case_dir),

        General(
            flameGeometry='planar',
            twinFlame=False
        ),

        Chemistry(
            mechanismFile='h2-burke.yaml',
            phaseID='h2-burke',
            rateMultiplierFunction=None
        ),

        InitialCondition(
            flameType='diffusion',
            restartFile=cantera_profile,   # <-- key change
            fuel={'H2': 1.0},
            oxidizer={'O2': 0.082, 'AR': 0.918},
            Tfuel=300,
            Toxidizer=1300,
            xLeft=-0.004,
            xRight=0.004,
            centerWidth=0.002,
            slopeWidth=0.001
        ),

        Times(
            tStart=0.0,
            globalTimestep=5e-6,        # was 2e-5
            outputTimeInterval=5e-6,    # optional, keep diagnostics consistent
            profileTimeInterval=2e-4   # was 1e-3
        ),

        TerminationCondition(tEnd=10.0)
    )

    conf.run()
    struct = utils.load(os.path.join(case_dir, 'profNow.h5'))
    return struct


# -------------------------
# Run non-reactive case
# -------------------------
struct_nonreactive = run_sim('non_reactive', chemistry_enabled=False)

# -------------------------
# Run reactive case
# -------------------------
struct_reactive = run_sim('reactive', chemistry_enabled=True)

# -------------------------
# Determine species count and names
# -------------------------
n_species = struct_reactive.Y.shape[0] if struct_reactive.Y.shape[0] != len(struct_reactive.x) else struct_reactive.Y.shape[1]
species_names = getattr(struct_reactive, 'speciesNames', [f'S{i}' for i in range(n_species)])

# Function to extract species data along grid robustly
def get_species_data(struct, species_index):
    Y = struct.Y
    if Y.shape[0] == len(struct.x):
        return Y[:, species_index]
    elif Y.shape[1] == len(struct.x):
        return Y[species_index, :]
    else:
        raise ValueError(f"Cannot match Y shape {Y.shape} with x length {len(struct.x)}")

# -------------------------
# Plot all species comparison
# -------------------------
for i, name in enumerate(species_names):
    data_nonreactive = get_species_data(struct_nonreactive, i)
    data_reactive = get_species_data(struct_reactive, i)
    f, ax = plt.subplots()
    ax.plot(struct_nonreactive.x, data_nonreactive, label='Non-reactive')
    ax.plot(struct_reactive.x, data_reactive, label='Reactive')
    ax.set_xlabel('Position [m]')
    ax.set_ylabel(f'{name} Mass Fraction')
    ax.set_title(f'{name} Comparison')
    ax.legend()
    f.savefig(os.path.join(output, f'{name}_Comparison.svg'))
    plt.close(f)

# -------------------------
# Plot Temperature comparison
# -------------------------
fT, axT = plt.subplots()
axT.plot(struct_nonreactive.x, struct_nonreactive.T, label='Non-reactive')
axT.plot(struct_reactive.x, struct_reactive.T, label='Reactive')
axT.set_xlabel('Position [m]')
axT.set_ylabel('Temperature [K]')
axT.set_title('Temperature Comparison')
axT.legend()
fT.savefig(os.path.join(output, 'Temperature_Comparison.svg'))
plt.close(fT)

# -------------------------
# Plot Axial Velocity comparison
# -------------------------
fV, axV = plt.subplots()
axV.plot(struct_nonreactive.x, struct_nonreactive.V / struct_nonreactive.rho, label='Non-reactive')
axV.plot(struct_reactive.x, struct_reactive.V / struct_reactive.rho, label='Reactive')
axV.set_xlabel('Position [m]')
axV.set_ylabel('Axial Velocity [m/s]')
axV.set_title('Axial Velocity Comparison')
axV.legend()
fV.savefig(os.path.join(output, 'AxialVelocity_Comparison.svg'))
plt.close(fV)

# -------------------------
# Plot Heat Release Rate (HRR) comparison
# -------------------------
def compute_HRR(struct):
    # Non-reactive or missing chemistry → zero HRR
    if not hasattr(struct, 'dYdtProd'):
        return np.zeros_like(struct.x)

    # Reactive case: qualitative HRR proxy
    hrr = -np.sum(struct.dYdtProd, axis=0)

    return hrr


hrr_nonreactive = compute_HRR(struct_nonreactive)
hrr_reactive = compute_HRR(struct_reactive)

fH, axH = plt.subplots()
axH.plot(struct_nonreactive.x, hrr_nonreactive, label='Non-reactive')
axH.plot(struct_reactive.x, hrr_reactive, label='Reactive')
axH.set_xlabel('Position [m]')
axH.set_ylabel('HRR [arbitrary units]')
axH.set_title('Heat Release Rate Comparison')
axH.legend()
fH.savefig(os.path.join(output, 'HRR_Comparison.svg'))
plt.close(fH)
