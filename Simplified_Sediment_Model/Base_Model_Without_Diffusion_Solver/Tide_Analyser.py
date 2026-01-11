import pandas as pd
import numpy as np
import os

# ========================================================================================
# CONFIGURATION
# ========================================================================================
# Uses current directory where the script is located
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(WORKING_DIR, "Model_Output_No_Diffusion.csv")

# Output Paths
OUTPUT_DIR = os.path.join(WORKING_DIR, "Analysis")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "Temporal_RMSE_Analysis.csv")
OUTPUT_MD = os.path.join(OUTPUT_DIR, "Temporal_RMSE_Analysis.md")

# Ensure Analysis folder exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========================================================================================
# HELPER FUNCTIONS
# ========================================================================================

def calculate_rmse(df, measured_col, modelled_col):
    """Calculates Root Mean Squared Error for a dataframe subset."""
    if len(df) == 0:
        return np.nan
    
    residuals = df[measured_col] - df[modelled_col]
    mse = (residuals ** 2).mean()
    return np.sqrt(mse)

# ========================================================================================
# MAIN EXECUTION
# ========================================================================================

def main():
    print(f"Reading data from: {INPUT_FILE}")
    
    if not os.path.exists(INPUT_FILE):
        print("Error: Input file not found. Please run the model first.")
        return

    df = pd.read_csv(INPUT_FILE)
    
    # Column names
    col_measured = 'Target_Measured_Avg_s0_g_L'
    col_modelled = 'Modelled_s0'
    col_tide_state = 'Tide_State'   # HT / LT
    col_day = 'Sampling_Day'        # Day_1, Day_2, Day_3

    # Check columns
    for col in [col_measured, col_modelled, col_tide_state, col_day]:
        if col not in df.columns:
            print(f"Error: Column '{col}' not found in CSV.")
            return

    # -------------------------------------------------------------------------
    # 1. TIDE STATE ANALYSIS (High vs Low)
    # -------------------------------------------------------------------------
    print("\n--- Analysis 1: Tide State (HT vs LT) ---")
    
    df_ht = df[df[col_tide_state] == 'HT']
    df_lt = df[df[col_tide_state] == 'LT']
    
    rmse_ht = calculate_rmse(df_ht, col_measured, col_modelled)
    rmse_lt = calculate_rmse(df_lt, col_measured, col_modelled)
    
    print(f"High Tide (HT) Samples (n={len(df_ht)}) : RMSE = {rmse_ht:.5f} g/L")
    print(f"Low Tide (LT) Samples  (n={len(df_lt)}) : RMSE = {rmse_lt:.5f} g/L")

    # -------------------------------------------------------------------------
    # 2. TIDAL PHASE ANALYSIS (Neap vs Spring)
    # -------------------------------------------------------------------------
    print("\n--- Analysis 2: Tidal Phase (Neap vs Spring) ---")
    
    # Definition: Neap = Day_1; Spring = Day_2 and Day_3
    df_neap = df[df[col_day] == 'Day_1']
    df_spring = df[df[col_day].isin(['Day_2', 'Day_3'])]
    
    rmse_neap = calculate_rmse(df_neap, col_measured, col_modelled)
    rmse_spring = calculate_rmse(df_spring, col_measured, col_modelled)
    
    print(f"Neap Tide (Day 1)      (n={len(df_neap)}) : RMSE = {rmse_neap:.5f} g/L")
    print(f"Spring Tide (Days 2&3) (n={len(df_spring)}) : RMSE = {rmse_spring:.5f} g/L")

    # -------------------------------------------------------------------------
    # SAVE RESULTS
    # -------------------------------------------------------------------------
    
    # Store data in a list of dictionaries
    results_data = [
        {'Category': 'High Tide (HT)', 'N': len(df_ht), 'RMSE': rmse_ht},
        {'Category': 'Low Tide (LT)', 'N': len(df_lt), 'RMSE': rmse_lt},
        {'Category': 'Neap Tide (Day 1)', 'N': len(df_neap), 'RMSE': rmse_neap},
        {'Category': 'Spring Tide (Days 2 & 3)', 'N': len(df_spring), 'RMSE': rmse_spring}
    ]
    
    # 1. Save CSV
    results_df = pd.DataFrame(results_data)
    # Rename column for CSV readability
    results_df.rename(columns={'RMSE': 'RMSE (g/L)'}, inplace=True)
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nCSV Results saved to: {OUTPUT_CSV}")

    # 2. Save Markdown
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write("# Temporal Model Performance Analysis\n\n")
        f.write("Analysis of Root Mean Square Error (RMSE) across different temporal conditions.\n\n")
        
        # Write Table Header
        f.write("| Category | N Samples | RMSE (g L⁻¹) |\n")
        f.write("| :--- | :---: | :---: |\n")
        
        # Write Rows
        for row in results_data:
            cat = row['Category']
            n = row['N']
            rmse_val = row['RMSE']
            f.write(f"| {cat} | {n} | {rmse_val:.5f} |\n")
            
    print(f"Markdown Results saved to: {OUTPUT_MD}")

if __name__ == "__main__":
    main()