import pandas as pd
import numpy as np
import scipy.stats as stats
import os
import sys

# ========================================================================================
# CONFIGURATION
# ========================================================================================
# Uses the current directory where the script is located
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(WORKING_DIR, "Model_Output_No_Diffusion.csv")
OUTPUT_DIR = os.path.join(WORKING_DIR, "Analysis")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========================================================================================
# HELPER FUNCTIONS
# ========================================================================================

def calculate_r2(y_true, y_pred):
    """Calculates Coefficient of Determination (R^2)"""
    if len(y_true) < 2: return np.nan
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0: return np.nan
    return 1 - (ss_res / ss_tot)

def to_markdown_manual(df):
    """
    Converts a pandas DataFrame to a Markdown table string 
    without requiring the 'tabulate' library.
    """
    if df.empty:
        return ""
    
    # Create header
    columns = df.columns.tolist()
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    
    # Create rows
    rows = []
    for _, row in df.iterrows():
        row_str = "| " + " | ".join(str(val) for val in row) + " |"
        rows.append(row_str)
    
    return f"{header}\n{separator}\n" + "\n".join(rows)

def fmt_p(p):
    """Formats p-value for display"""
    if pd.isna(p): return "N/A"
    return "< 0.001" if p < 0.001 else f"{p:.4f}"

# ========================================================================================
# DYNAMIC INTERPRETATION LOGIC
# ========================================================================================

def get_interpretations(stats_dict):
    """
    Generates dynamic interpretation strings based on calculated statistics.
    Returns a dictionary mapping statistic names to interpretation strings.
    """
    interp = {}
    
    # 1. Measured Mean
    interp['meas'] = "Baseline condition"
    
    # 2. Modelled Mean
    if stats_dict['mean_mod'] < stats_dict['mean_meas']:
        interp['mod'] = "Systematic underestimation"
    elif stats_dict['mean_mod'] > stats_dict['mean_meas']:
        interp['mod'] = "Systematic overestimation"
    else:
        interp['mod'] = "Close agreement with baseline"
        
    # 3. Mean Difference (Measured - Modelled)
    # Positive Diff = Underestimation (Measured > Modelled)
    if stats_dict['mean_diff'] > 0:
        interp['diff'] = "Model underestimates measured SSC"
    elif stats_dict['mean_diff'] < 0:
        interp['diff'] = "Model overestimates measured SSC"
    else:
        interp['diff'] = "No mean bias observed"
        
    # 4. Median Difference
    interp['median'] = "Central tendency of differences"
    
    # 5. SD of Differences
    # Heuristic: Compare SD to Mean Difference to judge scatter
    if abs(stats_dict['sd_diff']) > abs(stats_dict['mean_diff']):
        interp['sd'] = "High scatter around the mean bias"
    else:
        interp['sd'] = "Moderate scatter around the mean bias"
        
    # 6. Max Positive Error
    interp['max_err'] = "Worst-case underprediction"
    
    # 7. 95% CI
    lower, upper = stats_dict['ci_95']
    if (lower < 0 < upper):
        interp['ci'] = "Bias is not significantly different from zero"
    else:
        interp['ci'] = "Bias is significantly different from zero"
        
    # 8. RMSE / MAE
    interp['rmse'] = "Overall prediction accuracy"
    interp['mae'] = "Relative error magnitude"
    
    # 9. Significance (t-test / Wilcoxon)
    if stats_dict['p_ttest'] < 0.05:
        t_int = "Mean bias is statistically significant"
    else:
        t_int = "Mean bias is not statistically significant"
        
    if stats_dict['p_wilcox'] < 0.05:
        w_int = "Median bias is statistically significant"
    else:
        w_int = "Median bias is not statistically significant"
        
    interp['sig'] = f"{t_int}; {w_int}"
    
    # 10. Correlations
    r = stats_dict['pearson_r']
    if pd.isna(r):
        r_str = "Insufficient data"
    elif r > 0.9:
        r_str = "Very strong linear relationship"
    elif r > 0.7:
        r_str = "Strong linear relationship"
    elif r > 0.5:
        r_str = "Moderate linear relationship"
    else:
        r_str = "Weak linear relationship"
        
    r2 = stats_dict['r2']
    if pd.isna(r2):
        r2_str = ""
    else:
        r2_str = f"model explains {r2*100:.1f}% of variance"
        
    interp['corr'] = f"{r_str}; {r2_str}"
    
    # 11. Bland-Altman
    pct_out = stats_dict['pct_outliers']
    if pct_out < 5.0:
        interp['ba'] = "Good agreement; few outliers"
    elif pct_out < 10.0:
        interp['ba'] = "Most points within limits"
    else:
        interp['ba'] = "Significant number of outliers detected"
        
    return interp

# ========================================================================================
# MAIN ANALYSIS
# ========================================================================================

def main():
    print(f"Reading output file: {INPUT_FILE}")
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Run the model first.")
        return

    df = pd.read_csv(INPUT_FILE)
    
    # Ensure columns exist
    req_cols = ['Canal_Type', 'Location', 'Target_Measured_Avg_s0_g_L', 'Modelled_s0']
    for c in req_cols:
        if c not in df.columns:
            print(f"Error: Column '{c}' missing from CSV.")
            return

    # ------------------------------------------------------------------------------------
    # PART 1: CANAL TYPE ANALYSIS (Table 1)
    # ------------------------------------------------------------------------------------
    print("Performing Canal Type Analysis (Table 1)...")
    
    # Define Subsets
    stem_locs = ['C', 'D', 'F', 'F_across', 'H']
    frond_locs = ['E', 'G']
    
    subsets = [
        ('LC (Ponds 14, 15)', df[df['Canal_Type'] == 'LC']),
        ('CPC Stem (Ponds 6, 8)', df[(df['Canal_Type'] == 'CPC') & (df['Location'].str.strip().isin(stem_locs))]),
        ('CPC Frond (Ponds 6, 8)', df[(df['Canal_Type'] == 'CPC') & (df['Location'].str.strip().isin(frond_locs))]),
        ('All CPC Instances', df[df['Canal_Type'] == 'CPC'])
    ]
    
    results_t1 = []
    
    for label, sub_df in subsets:
        if len(sub_df) == 0:
            results_t1.append({'Canal Type': label, 'Mean Error': np.nan})
            continue
            
        meas = sub_df['Target_Measured_Avg_s0_g_L']
        mod = sub_df['Modelled_s0']
        # Error = Measured - Modelled (Positive = Underestimation)
        error = meas - mod
        
        results_t1.append({
            'Canal Type': label,
            'Mean Error (g L⁻¹)': round(error.mean(), 5),
            'SD Error (g L⁻¹)': round(error.std(ddof=1), 5),
            'RMSE (g L⁻¹)': round(np.sqrt((error ** 2).mean()), 5),
            'R²': round(calculate_r2(meas, mod), 4)
        })
    
    df_t1 = pd.DataFrame(results_t1)
    
    # Save Table 1
    df_t1.to_csv(os.path.join(OUTPUT_DIR, "Table1_Canal_Analysis.csv"), index=False)
    with open(os.path.join(OUTPUT_DIR, "Table1_Canal_Analysis.md"), 'w', encoding='utf-8') as f:
        f.write(to_markdown_manual(df_t1))
        
    print("Table 1 generated.")

    # ------------------------------------------------------------------------------------
    # PART 2: DETAILED STATISTICS (Table 2)
    # ------------------------------------------------------------------------------------
    print("Calculating Detailed Statistics (Table 2)...")
    
    # Use full dataset
    meas = df['Target_Measured_Avg_s0_g_L']
    mod = df['Modelled_s0']
    diff = meas - mod
    
    # Calculate Stats
    stats_dict = {
        'mean_meas': meas.mean(),
        'mean_mod': mod.mean(),
        'mean_diff': diff.mean(),
        'median_diff': diff.median(),
        'sd_diff': diff.std(ddof=1),
        'max_pos_err': diff.max(),
        'ci_95': stats.t.interval(0.95, len(diff)-1, loc=diff.mean(), scale=stats.sem(diff)),
        'rmse': np.sqrt((diff ** 2).mean()),
        'mae': diff.abs().mean(),
        'p_ttest': stats.ttest_rel(meas, mod)[1],
        'p_wilcox': stats.wilcoxon(meas, mod)[1],
        'pearson_r': stats.pearsonr(meas, mod)[0],
        'p_pearson': stats.pearsonr(meas, mod)[1],
        'spearman_rho': stats.spearmanr(meas, mod)[0],
        'p_spearman': stats.spearmanr(meas, mod)[1],
        'r2': calculate_r2(meas, mod)
    }
    
    # Bland-Altman Outliers
    ba_upper = stats_dict['mean_diff'] + 1.96 * stats_dict['sd_diff']
    ba_lower = stats_dict['mean_diff'] - 1.96 * stats_dict['sd_diff']
    outliers = diff[(diff > ba_upper) | (diff < ba_lower)]
    stats_dict['pct_outliers'] = (len(outliers) / len(diff)) * 100
    stats_dict['n_outliers'] = len(outliers)
    stats_dict['n_total'] = len(diff)
    
    # Percent Bias
    pct_bias = (stats_dict['mean_diff'] / stats_dict['mean_meas']) * 100

    # Get Dynamic Interpretations
    interp = get_interpretations(stats_dict)

    # ------------------------------------------------------------------------------------
    # GENERATE MARKDOWN TABLE 2
    # ------------------------------------------------------------------------------------
    
    # Define T-test/Wilcoxon string for the value column
    t_val_str = f"t = {stats.ttest_rel(meas, mod)[0]:.4f}, df = {len(diff)-1}, p = {fmt_p(stats_dict['p_ttest'])}"
    w_val_str = f"W = {stats.wilcoxon(meas, mod)[0]:.1f}, p = {fmt_p(stats_dict['p_wilcox'])}"

    table_md = f"""
| Statistic | Value | Interpretation |
| :--- | :--- | :--- |
| Mean Measured SSC of Samples | {stats_dict['mean_meas']:.4f} g L⁻¹ | {interp['meas']} |
| Mean Modelled SSC | {stats_dict['mean_mod']:.4f} g L⁻¹ | {interp['mod']} |
| Mean Difference (Measured − Modelled) | {stats_dict['mean_diff']:+.4f} g L⁻¹; {pct_bias:.2f}% bias | {interp['diff']} |
| Median Difference | {stats_dict['median_diff']:.4f} g L⁻¹ | {interp['median']} |
| Standard Deviation of Differences | {stats_dict['sd_diff']:.4f} g L⁻¹ | {interp['sd']} |
| Maximum Positive Error (underestimation) | {stats_dict['max_pos_err']:.4f} g L⁻¹ | {interp['max_err']} |
| 95% Confidence Interval of Bias | {stats_dict['ci_95'][0]:.4f} – {stats_dict['ci_95'][1]:.4f} g L⁻¹ | {interp['ci']} |
| Root Mean Square Error (RMSE) | {stats_dict['rmse']:.4f} g L⁻¹ | {interp['rmse']} |
| Mean Absolute Error (MAE) | {stats_dict['mae']:.4f} g L⁻¹ | {interp['mae']} |
| **Statistical Significance of Bias** | | {interp['sig']} |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Paired t-test | {t_val_str} | |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Wilcoxon signed rank | {w_val_str} | |
| **Correlation Metrics** | | {interp['corr']} |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Pearson r | {stats_dict['pearson_r']:.4f} (p {fmt_p(stats_dict['p_pearson'])}) | |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Spearman ρ | {stats_dict['spearman_rho']:.4f} (p {fmt_p(stats_dict['p_spearman'])}) | |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; R² | {stats_dict['r2']:.4f} | |
| **Bland-Altman Analysis** | | {interp['ba']} |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Mean diff | {stats_dict['mean_diff']:.4f} g L⁻¹ | |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 95% limits | {ba_lower:.4f} to {ba_upper:.4f} g L⁻¹ | |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Outliers | {stats_dict['n_outliers']}/{stats_dict['n_total']} ({stats_dict['pct_outliers']:.1f}%) outside limits | |
"""
    
    # Save Markdown
    t2_md_path = os.path.join(OUTPUT_DIR, "Table2_Detailed_Stats.md")
    with open(t2_md_path, 'w', encoding='utf-8') as f:
        f.write(table_md)

    # Save CSV Version of Table 2 (Simplified structure for CSV)
    t2_csv_rows = [
        ['Mean Measured SSC', stats_dict['mean_meas'], interp['meas']],
        ['Mean Modelled SSC', stats_dict['mean_mod'], interp['mod']],
        ['Mean Difference', stats_dict['mean_diff'], interp['diff']],
        ['RMSE', stats_dict['rmse'], interp['rmse']],
        ['R2', stats_dict['r2'], interp['corr']],
    ]
    df_t2 = pd.DataFrame(t2_csv_rows, columns=['Statistic', 'Value', 'Interpretation'])
    df_t2.to_csv(os.path.join(OUTPUT_DIR, "Table2_Detailed_Stats.csv"), index=False)

    print(f"Table 2 saved to {OUTPUT_DIR}")
    print("Analysis Complete.")

if __name__ == "__main__":
    main()