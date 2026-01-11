"""
SEDIMENT TRANSPORT MODEL
WITH 1-D LONGITUDINAL DIFFUSION SOLVER

Advection–diffusion–reaction formulation.
Explicit spatial marching with numerical stability diagnostics.
"""

import pandas as pd
import numpy as np
import os

# ======================================================================================
# 1. CONFIGURATION AND CONSTANTS
# ======================================================================================

KAPPA = 0.41
REF_HEIGHT_RATIO = 0.75
DRAG_COEFF_CD = 0.0025
DEFAULT_SETTLING_VEL = 0.0005
SMALL = 1e-9

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FILE = os.path.join(WORKING_DIR, "Input_Data.csv")
OUTPUT_FILE = os.path.join(WORKING_DIR, "Model_Output_With_Diffusion.csv")
REPORT_FILE = os.path.join(WORKING_DIR, "Model_Validation_and_Stability_Report.md")

# ======================================================================================
# 2. PHYSICAL FUNCTIONS
# ======================================================================================

def calc_shear_velocity(U):
    return abs(U) * np.sqrt(DRAG_COEFF_CD / 2.0)

def calc_rouse_number(w, u_star):
    if u_star <= SMALL:
        return 0.0
    return w / (KAPPA * u_star)

def calc_vertical_fluxes(w, sb, eps_z, P, h):
    a = REF_HEIGHT_RATIO * h
    if h - a <= 1e-4:
        return 0.0, 0.0, 0.0

    settling_flux = w * sb
    diffusive_flux = eps_z * (P * sb) / (h - a)
    Fs = settling_flux - diffusive_flux

    return settling_flux, diffusive_flux, Fs

def advection_diffusion_step(s_prev, Fs, dx, U, h, D):
    adv = -(Fs * dx) / (abs(U) * h)
    diff = D * (dx ** 2)
    s_next = s_prev + adv + diff
    return max(s_next, 0.0)

# ======================================================================================
# 3. MAIN EXECUTION
# ======================================================================================

def main():

    df = pd.read_csv(INPUT_FILE)

    if 'Settling_Velocity_w_m_s' not in df.columns:
        df['Settling_Velocity_w_m_s'] = DEFAULT_SETTLING_VEL

    df['Modelled_s0'] = 0.0
    df['Peclet_Number'] = 0.0

    results_map = {}

    df.sort_values(
        by=['Plot', 'Sampling_Day', 'Tide_Phase', 'Distance_to_Neighbor_m'],
        inplace=True
    )

    for idx, row in df.iterrows():

        U = row['Streamwise_Velocity_U_m_s']
        h = row['Water_Depth_m']
        w = row['Settling_Velocity_w_m_s']
        sb = row['Input_Near_Bed_Conc_sb_g_L']
        eps_z = row['Diffusivity_Vertical_m2_s']
        D = row['Diffusivity_Longitudinal_m2_s']
        dx = row['Distance_to_Neighbor_m']
        neighbor = row['Neighbor_Node']

        u_star = calc_shear_velocity(U)
        P = calc_rouse_number(w, u_star)

        _, _, Fs = calc_vertical_fluxes(w, sb, eps_z, P, h)

        Pe = abs(U) * dx / max(D, SMALL)
        df.at[idx, 'Peclet_Number'] = Pe

        key = f"{row['Plot']}_{row['Location']}_{row['Sampling_Day']}_{row['Tide_State']}"
        parent_key = f"{row['Plot']}_{neighbor}_{row['Sampling_Day']}_{row['Tide_State']}"

        if dx == 0 or neighbor == "Dutch_Canal":
            s0 = row['Target_Measured_Avg_s0_g_L']
        else:
            s_prev = results_map.get(parent_key, row['Target_Measured_Avg_s0_g_L'])
            s0 = advection_diffusion_step(s_prev, Fs, dx, U, h, D)

        results_map[key] = s0
        df.at[idx, 'Modelled_s0'] = s0

    # ==================================================================================
    # 4. VALIDATION AND STABILITY REPORT
    # ==================================================================================

    df['Residual'] = df['Target_Measured_Avg_s0_g_L'] - df['Modelled_s0']
    df['Abs_Error'] = df['Residual'].abs()
    df['Squared_Error'] = df['Residual'] ** 2

    rmse = np.sqrt(df['Squared_Error'].mean())
    bias = df['Residual'].mean()
    mae = df['Abs_Error'].mean()

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("# MODEL VALIDATION AND STABILITY REPORT\n\n")
        f.write("## Model With Longitudinal Diffusion Solver\n\n")
        f.write("### Validation Statistics\n")
        f.write(f"- Root Mean Squared Error : {rmse:.5f}\n")
        f.write(f"- Mean Bias : {bias:.5f}\n")
        f.write(f"- Mean Absolute Error  : {mae:.5f}\n\n")
        f.write("### Stability Diagnostics\n")
        f.write(f"- Max Peclet  : {df['Peclet_Number'].max():.2f}\n")
        f.write(f"- Mean Peclet : {df['Peclet_Number'].mean():.2f}\n")

    df.to_csv(OUTPUT_FILE, index=False)

if __name__ == "__main__":
    main()
