import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error
import os

# Set professional thesis-ready formatting
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.5
})

work_dir = os.path.dirname(os.path.abspath(__file__))

# Create Plots subfolder
plots_dir = os.path.join(work_dir, "Plots")
os.makedirs(plots_dir, exist_ok=True)

try:
    # Load model output data (contains both measured and modelled values)
    model_df = pd.read_csv("Model_Output_No_Diffusion.csv")
    print(f"Successfully loaded {len(model_df)} model observations")
except FileNotFoundError:
    print("Error: Model_Output_No_Diffusion.csv not found")
    exit()

# Extract measured and modelled SSC values
measured_ssc = model_df['Target_Measured_Avg_s0_g_L'].values
modelled_ssc = model_df['Modelled_s0'].values

# Calculate performance metrics
r2 = r2_score(measured_ssc, modelled_ssc)
rmse = np.sqrt(mean_squared_error(measured_ssc, modelled_ssc))
nse = 1 - np.sum((modelled_ssc - measured_ssc)**2) / np.sum((measured_ssc - np.mean(measured_ssc))**2)
pbias = 100 * np.sum(modelled_ssc - measured_ssc) / np.sum(measured_ssc)
mae = np.mean(np.abs(modelled_ssc - measured_ssc))

# --- Plot 1: Scatter Plot with 1:1 Line and Regression ---
fig1, ax1 = plt.subplots(figsize=(8, 6), dpi=100)
min_val = min(measured_ssc.min(), modelled_ssc.min())
max_val = max(measured_ssc.max(), modelled_ssc.max())

# 1:1 line
ax1.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5, 
         alpha=0.8, label='1:1 line', zorder=2)

# Scatter points colored by canal type
canal_types = model_df['Canal_Type'].unique()
colors = plt.cm.Set2(np.linspace(0, 1, len(canal_types)))
for i, ctype in enumerate(canal_types):
    mask = model_df['Canal_Type'] == ctype
    ax1.scatter(measured_ssc[mask], modelled_ssc[mask], 
               c=[colors[i]], s=80, alpha=0.8, edgecolors='k', 
               linewidth=0.5, label=f'{ctype}', zorder=3)

# Regression line
slope, intercept, r, p, se = stats.linregress(measured_ssc, modelled_ssc)
reg_line = slope * measured_ssc + intercept
ax1.plot(measured_ssc, reg_line, 'r-', linewidth=2, 
         label=f'Fit: y = {slope:.2f}x + {intercept:.3f}', zorder=4)

ax1.set_xlabel('Measured SSC (g L$^{-1}$)')
ax1.set_ylabel('Modelled SSC (g L$^{-1}$)')
ax1.set_title('Modelled vs Measured Suspended Sediment Concentration', fontweight='bold')
ax1.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
ax1.grid(True, alpha=0.3)

# Metrics textbox
metrics_text = f'R$^2$ = {r2:.3f}\nRMSE = {rmse:.4f} g L$^{-1}$\nNSE = {nse:.3f}\nPBIAS = {pbias:.2f}%'
ax1.text(0.05, 0.95, metrics_text, transform=ax1.transAxes, 
         verticalalignment='top', fontsize=10,
         bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat', alpha=0.7))

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'SSC_Scatter_Plot.png'), dpi=300, bbox_inches='tight')
plt.show()

# --- Plot 2: Bland-Altman Plot ---
fig2, ax2 = plt.subplots(figsize=(8, 6), dpi=100)
mean_vals = (measured_ssc + modelled_ssc) / 2
diff_vals = modelled_ssc - measured_ssc
mean_diff = np.mean(diff_vals)
std_diff = np.std(diff_vals)
ci_upper = mean_diff + 1.96 * std_diff
ci_lower = mean_diff - 1.96 * std_diff

ax2.scatter(mean_vals, diff_vals, c='steelblue', s=80, alpha=0.8, 
           edgecolors='k', linewidth=0.5, zorder=3)
ax2.axhline(mean_diff, color='k', linestyle='-', linewidth=2, 
           label=f'Mean: {mean_diff:.4f}', zorder=2)
ax2.axhline(ci_upper, color='r', linestyle='--', linewidth=1.5, 
           label=f'+1.96 SD: {ci_upper:.4f}', zorder=2)
ax2.axhline(ci_lower, color='r', linestyle='--', linewidth=1.5, 
           label=f'-1.96 SD: {ci_lower:.4f}', zorder=2)

ax2.set_xlabel('Mean SSC (g L$^{-1}$)')
ax2.set_ylabel('Difference (Modelled - Measured) (g L$^{-1}$)')
ax2.set_title('Bland-Altman Analysis of Model Performance', fontweight='bold')
ax2.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'SSC_Bland_Altman.png'), dpi=300, bbox_inches='tight')
plt.show()

# --- Plot 3: Residuals vs Fitted Values ---
fig3, ax3 = plt.subplots(figsize=(8, 6), dpi=100)
residuals = measured_ssc - modelled_ssc  # Measured - Modelled
ax3.scatter(modelled_ssc, residuals, c='seagreen', s=80, alpha=0.8, 
           edgecolors='k', linewidth=0.5)
ax3.axhline(y=0, color='k', linestyle='-', linewidth=2, label='Zero line')

ax3.set_xlabel('Modelled SSC (g L$^{-1}$)')
ax3.set_ylabel('Residuals (Measured - Modelled) (g L$^{-1}$)')
ax3.set_title('Residuals vs Fitted Values', fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Add lowess trend line if seaborn is available
try:
    sns.regplot(x=modelled_ssc, y=residuals, ax=ax3, 
                scatter=False, lowess=True, color='r', line_kws={'linewidth': 1.5})
except:
    pass

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'SSC_Residuals_vs_Fitted.png'), dpi=300, bbox_inches='tight')
plt.show()

# --- Plot 4: Histogram of Residuals ---
fig4, ax4 = plt.subplots(figsize=(8, 6), dpi=100)
ax4.hist(residuals, bins=int(np.sqrt(len(residuals)))+1, color='darkorange', 
         alpha=0.7, edgecolor='k', linewidth=1)
ax4.axvline(x=0, color='k', linestyle='-', linewidth=2, label='Zero')
ax4.axvline(x=np.mean(residuals), color='r', linestyle='--', linewidth=2, 
           label=f'Mean: {np.mean(residuals):.4f}')

ax4.set_xlabel('Residuals (g L$^{-1}$)')
ax4.set_ylabel('Frequency')
ax4.set_title('Distribution of Residuals', fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'SSC_Residual_Histogram.png'), dpi=300, bbox_inches='tight')
plt.show()

# --- Plot 5: Performance by Tide State ---
fig5, ax5 = plt.subplots(figsize=(8, 6), dpi=100)
for tide in ['HT', 'LT']:
    mask = model_df['Tide_State'] == tide
    ax5.scatter(measured_ssc[mask], modelled_ssc[mask], 
               s=90, alpha=0.8, edgecolors='k', linewidth=0.5,
               label=f'{tide} (n={mask.sum()})')

ax5.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5)
ax5.set_xlabel('Measured SSC (g L$^{-1}$)')
ax5.set_ylabel('Modelled SSC (g L$^{-1}$)')
ax5.set_title('Model Performance by Tide State', fontweight='bold')
ax5.legend(title='Tide State', title_fontsize=11, frameon=True, fancybox=True, shadow=True)
ax5.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'SSC_Performance_by_Tide.png'), dpi=300, bbox_inches='tight')
plt.show()

# --- Plot 6: Performance by Sampling Day ---
fig6, ax6 = plt.subplots(figsize=(8, 6), dpi=100)
days = sorted(model_df['Sampling_Day'].unique())
for i, day in enumerate(days):
    mask = model_df['Sampling_Day'] == day
    ax6.scatter(measured_ssc[mask], modelled_ssc[mask], 
               s=90, alpha=0.8, edgecolors='k', linewidth=0.5,
               label=f'{day} (n={mask.sum()})')

ax6.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5)
ax6.set_xlabel('Measured SSC (g L$^{-1}$)')
ax6.set_ylabel('Modelled SSC (g L$^{-1}$)')
ax6.set_title('Model Performance by Sampling Day', fontweight='bold')
ax6.legend(title='Sampling Day', title_fontsize=11, frameon=True, fancybox=True, shadow=True)
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'SSC_Performance_by_Day.png'), dpi=300, bbox_inches='tight')
plt.show()

# --- Plot 7: Normal Q-Q Plot of Residuals ---
fig7, ax7 = plt.subplots(figsize=(8, 6), dpi=100)
stats.probplot(residuals, dist="norm", plot=ax7)
ax7.get_lines()[0].set_marker('o')
ax7.get_lines()[0].set_markerfacecolor('steelblue')
ax7.get_lines()[0].set_markeredgecolor('k')
ax7.get_lines()[0].set_markersize(8)
ax7.get_lines()[1].set_color('r')
ax7.get_lines()[1].set_linewidth(2)
ax7.set_title('Normal Q-Q Plot of Residuals', fontweight='bold')
ax7.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'SSC_QQ_Plot.png'), dpi=300, bbox_inches='tight')
plt.show()

# --- Summary Statistics Output ---
# Create markdown content
markdown_content = f"""# Model Performance Summary - Suspended Sediment Concentration

| Metric | Value | Units |
|--------|-------|-------|
| Observations | {len(measured_ssc)} | - |
| R-squared | {r2:.4f} | - |
| RMSE | {rmse:.4f} | g L⁻¹ |
| Nash-Sutcliffe Efficiency | {nse:.4f} | - |
| Percent Bias (PBIAS) | {pbias:.2f} | % |
| Mean Absolute Error | {mae:.4f} | g L⁻¹ |
| Mean Residual | {np.mean(residuals):.4f} | g L⁻¹ |
"""

# Save markdown file
markdown_path = os.path.join(plots_dir, 'Model_Performance_Summary.md')
with open(markdown_path, 'w', encoding='utf-8') as f:
    f.write(markdown_content)

print(f"\nMarkdown summary saved to: {markdown_path}")

# Print to console
print("\n" + "="*55)
print("MODEL PERFORMANCE SUMMARY - SUSPENDED SEDIMENT CONCENTRATION")
print("="*55)
print(f"{'Metric':<25} {'Value':<15} {'Units':<15}")
print("-"*55)
print(f"{'Observations':<25} {len(measured_ssc):<15} {'-':<15}")
print(f"{'R-squared':<25} {r2:<15.4f} {'-':<15}")
print(f"{'RMSE':<25} {rmse:<15.4f} {'g L⁻¹':<15}")
print(f"{'Nash-Sutcliffe Efficiency':<25} {nse:<15.4f} {'-':<15}")
print(f"{'Percent Bias (PBIAS)':<25} {pbias:<15.2f} {'%':<15}")
print(f"{'Mean Absolute Error':<25} {mae:<15.4f} {'g L⁻¹':<15}")
print(f"{'Mean Residual':<25} {np.mean(residuals):<15.4f} {'g L⁻¹':<15}")
print("="*55 + "\n")

# --- Save results to CSV ---
results_df = pd.DataFrame({
    'Code': model_df['Code'],
    'Measured_SSC_g_L': measured_ssc,
    'Modelled_SSC_g_L': modelled_ssc,
    'Residual_g_L': residuals,
    'Canal_Type': model_df['Canal_Type'],
    'Tide_State': model_df['Tide_State'],
    'Sampling_Day': model_df['Sampling_Day']
})

results_df.to_csv('Model_Performance_Results.csv', index=False)
print(f"Detailed results saved to: {os.path.join(work_dir, 'Model_Performance_Results.csv')}")
print(f"All plots saved to: {plots_dir}")
print("\nIndividual plot files:")
for filename in os.listdir(plots_dir):
    if filename.endswith('.png'):
        print(f"  - {filename}")