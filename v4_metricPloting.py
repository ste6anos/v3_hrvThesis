import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import base64
from io import BytesIO
from datetime import datetime

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string."""
    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode()
    plt.close(fig)
    return img_str


def create_scatter_with_stats(ax, x_data, y_data, x_label, y_label, title):
    """Create scatter plot with regression line and statistics."""
    # Remove NaN values
    mask = ~(np.isnan(x_data) | np.isnan(y_data))
    x_clean = x_data[mask]
    y_clean = y_data[mask]

    if len(x_clean) < 3:
        ax.text(0.5, 0.5, 'Insufficient data',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_xlabel(x_label, fontsize=9)
        ax.set_ylabel(y_label, fontsize=9)
        ax.set_title(title, fontsize=10)
        return None, None

    # Scatter plot
    ax.scatter(x_clean, y_clean, alpha=0.6, s=40, edgecolors='k', linewidth=0.5)

    # Calculate correlation
    pearson_r, pearson_p = stats.pearsonr(x_clean, y_clean)
    spearman_r, spearman_p = stats.spearmanr(x_clean, y_clean)

    # Regression line
    z = np.polyfit(x_clean, y_clean, 1)
    p = np.poly1d(z)
    x_line = np.linspace(x_clean.min(), x_clean.max(), 100)
    ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)

    # Add statistics text box
    stats_text = f'Pearson r = {pearson_r:.3f}'
    if pearson_p < 0.001:
        stats_text += '\np < 0.001***'
    elif pearson_p < 0.01:
        stats_text += f'\np = {pearson_p:.3f}**'
    elif pearson_p < 0.05:
        stats_text += f'\np = {pearson_p:.3f}*'
    else:
        stats_text += f'\np = {pearson_p:.3f}'

    stats_text += f'\nSpearman ρ = {spearman_r:.3f}'
    stats_text += f'\nn = {len(x_clean)}'

    # Color code the box based on significance and strength
    if pearson_p < 0.05 and abs(pearson_r) > 0.3:
        bbox_color = 'lightgreen'
    elif pearson_p < 0.05:
        bbox_color = 'lightyellow'
    else:
        bbox_color = 'lightgray'

    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor=bbox_color, alpha=0.8))

    # Labels and title
    ax.set_xlabel(x_label, fontsize=9)
    ax.set_ylabel(y_label, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.grid(alpha=0.3)

    return pearson_r, pearson_p


def create_target_vs_all_metrics(df_data, clinical_data_df, target_name, all_features):
    """Create a figure with subplots: one subplot per HRV metric vs the target."""
    # Merge with target
    temp_df = clinical_data_df[['patientid', target_name]]
    df_merged = pd.merge(df_data, temp_df, on="patientid")
    df_clean = df_merged.dropna(subset=[target_name]).dropna()

    if len(df_clean) < 3:
        print(f"⚠️ Insufficient data for {target_name}: {len(df_clean)} samples")
        return None, []

    # Calculate grid size
    n_features = len(all_features)
    n_cols = 4
    n_rows = int(np.ceil(n_features / n_cols))

    # Create figure
    fig = plt.figure(figsize=(18, 4 * n_rows))

    correlations = []

    # Create subplots
    for idx, feature in enumerate(all_features):
        ax = plt.subplot(n_rows, n_cols, idx + 1)

        x_data = df_clean[feature].values
        y_data = df_clean[target_name].values

        feature_short = feature.replace('HRV_', '')

        r, p = create_scatter_with_stats(
            ax, x_data, y_data,
            feature_short, target_name,
            f'{feature_short} vs {target_name}'
        )

        if r is not None:
            correlations.append({'feature': feature_short, 'r': r, 'p': p})

    # Remove empty subplots
    for idx in range(n_features, n_rows * n_cols):
        fig.delaxes(plt.subplot(n_rows, n_cols, idx + 1))

    # Main title
    fig.suptitle(f'{target_name} vs All HRV Metrics (n={len(df_clean)})',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    return fig, correlations


def create_hrv_collinearity_heatmap(df_data, all_features, state_name):
    """Create collinearity heatmap for HRV metrics (feature-to-feature correlations)."""
    df_clean = df_data[all_features].dropna()

    if len(df_clean) < 3:
        return None

    # Calculate correlation matrix
    corr_matrix = df_clean.corr()

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Create heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                cmap='coolwarm', center=0, square=True,
                linewidths=0.5, cbar_kws={"shrink": 0.8},
                ax=ax, vmin=-1, vmax=1, annot_kws={'fontsize': 8})

    # Labels
    labels = [f.replace('HRV_', '') for f in all_features]
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(labels, rotation=0, fontsize=10)

    plt.title(f'HRV Metrics Collinearity Analysis - {state_name} (n={len(df_clean)})',
              fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()

    return fig


def create_correlation_heatmap(df_data, clinical_data_df, target_name, all_features):
    """Create correlation heatmap between target and all features."""
    # Merge with target
    temp_df = clinical_data_df[['patientid', target_name]]
    df_merged = pd.merge(df_data, temp_df, on="patientid")
    df_clean = df_merged.dropna(subset=[target_name]).dropna()

    if len(df_clean) < 3:
        return None

    # Calculate correlations
    correlations = []
    p_values = []

    for feature in all_features:
        r, p = stats.pearsonr(df_clean[feature], df_clean[target_name])
        correlations.append(r)
        p_values.append(p)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Correlation coefficients
    feature_names_short = [f.replace('HRV_', '') for f in all_features]
    colors = ['red' if r < 0 else 'green' for r in correlations]

    ax1.barh(feature_names_short, correlations, color=colors, alpha=0.7, edgecolor='k')
    ax1.axvline(x=0, color='k', linestyle='-', linewidth=1)
    ax1.set_xlabel('Pearson Correlation Coefficient', fontsize=12)
    ax1.set_ylabel('HRV Metric', fontsize=12)
    ax1.set_title(f'Correlations with {target_name}', fontsize=13, fontweight='bold')
    ax1.grid(alpha=0.3, axis='x')

    # Add significance stars
    for i, (r, p) in enumerate(zip(correlations, p_values)):
        if p < 0.001:
            sig = '***'
        elif p < 0.01:
            sig = '**'
        elif p < 0.05:
            sig = '*'
        else:
            sig = ''

        x_pos = r + (0.02 if r > 0 else -0.02)
        ax1.text(x_pos, i, sig, va='center', fontsize=10, fontweight='bold')

    # Plot 2: -log10(p-value) for significance
    log_p_values = [-np.log10(p) if p > 0 else 10 for p in p_values]

    colors_sig = ['darkred' if lp > -np.log10(0.05) else 'lightgray' for lp in log_p_values]
    ax2.barh(feature_names_short, log_p_values, color=colors_sig, alpha=0.7, edgecolor='k')
    ax2.axvline(x=-np.log10(0.05), color='red', linestyle='--', linewidth=2, label='p=0.05')
    ax2.axvline(x=-np.log10(0.01), color='darkred', linestyle='--', linewidth=2, label='p=0.01')
    ax2.set_xlabel('-log10(p-value)', fontsize=12)
    ax2.set_ylabel('HRV Metric', fontsize=12)
    ax2.set_title('Statistical Significance', fontsize=13, fontweight='bold')
    ax2.grid(alpha=0.3, axis='x')
    ax2.legend()

    fig.suptitle(f'{target_name} - Correlation Analysis (n={len(df_clean)})',
                 fontsize=14, fontweight='bold', y=1.00)

    plt.tight_layout()

    return fig


def create_html_report(results, collinearity_low_hr, collinearity_high_hr, output_file='hrv_correlation_report.html'):
    """Generate comprehensive HTML report with all correlation analyses."""

    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HRV Correlation Analysis Report</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', sans-serif;
            line-height: 1.6;
            color: #2c3e50;
            background: #f8f9fa;
            padding: 0;
        }}

        .container {{
            max-width: 1600px;
            margin: 0 auto;
            background: white;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}

        .header {{
            background: white;
            border-bottom: 2px solid #e9ecef;
            padding: 40px;
            text-align: center;
        }}

        .header h1 {{
            font-size: 2em;
            margin-bottom: 8px;
            font-weight: 600;
            color: #2c3e50;
            letter-spacing: -0.5px;
        }}

        .header p {{
            font-size: 1em;
            color: #6c757d;
            font-weight: 400;
        }}

        .content {{
            padding: 40px;
            max-width: 100%;
        }}

        .navigation {{
            background: white;
            padding: 24px;
            border: 1px solid #e9ecef;
            border-radius: 4px;
            margin-bottom: 40px;
        }}

        .navigation h3 {{
            margin-bottom: 16px;
            color: #2c3e50;
            font-size: 0.875em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-weight: 600;
        }}

        .navigation ul {{
            list-style: none;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 8px;
        }}

        .navigation li a {{
            display: block;
            padding: 10px 16px;
            background: #f8f9fa;
            border-left: 3px solid #dee2e6;
            text-decoration: none;
            color: #495057;
            font-size: 0.95em;
            transition: all 0.2s;
        }}

        .navigation li a:hover {{
            background: #e9ecef;
            border-left-color: #2c3e50;
            color: #2c3e50;
        }}

        .target-section {{
            margin-bottom: 80px;
            padding: 0;
            background: white;
        }}

        .target-section h2 {{
            color: #2c3e50;
            margin-bottom: 24px;
            font-size: 1.75em;
            font-weight: 600;
            padding-bottom: 12px;
            border-bottom: 1px solid #dee2e6;
        }}

        .state-comparison {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
            margin-bottom: 40px;
        }}

        .state-panel {{
            background: #fafbfc;
            padding: 20px;
            border: 1px solid #e1e4e8;
            border-radius: 4px;
        }}

        .state-panel h3 {{
            color: #2c3e50;
            margin-bottom: 16px;
            font-size: 1.125em;
            font-weight: 600;
            text-align: center;
            padding: 12px;
            background: white;
            border: 1px solid #e1e4e8;
            border-radius: 4px;
        }}

        .plot-container {{
            margin: 16px 0;
            text-align: center;
        }}

        .plot-container img {{
            max-width: 100%;
            border: 1px solid #e1e4e8;
            border-radius: 4px;
        }}

        .plot-label {{
            font-weight: 600;
            color: #495057;
            margin: 16px 0 10px 0;
            font-size: 1em;
        }}

        .summary-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 12px;
            margin: 24px 0;
        }}

        .stat-card {{
            background: white;
            padding: 16px;
            border: 1px solid #e1e4e8;
            border-radius: 4px;
            text-align: center;
        }}

        .stat-card h4 {{
            font-size: 0.75em;
            color: #6c757d;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-weight: 600;
        }}

        .stat-card p {{
            font-size: 1.75em;
            font-weight: 600;
            color: #2c3e50;
        }}

        .correlation-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 24px 0;
            background: white;
            border: 1px solid #e1e4e8;
            font-size: 0.9em;
        }}

        .correlation-table th {{
            background: #f6f8fa;
            color: #24292e;
            padding: 12px 16px;
            text-align: left;
            font-weight: 600;
            border-bottom: 2px solid #e1e4e8;
            font-size: 0.875em;
        }}

        .correlation-table td {{
            padding: 10px 16px;
            border-bottom: 1px solid #e1e4e8;
        }}

        .correlation-table tr:hover {{
            background: #f6f8fa;
        }}

        .badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 3px;
            font-size: 0.75em;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.3px;
        }}

        .badge-significant {{ background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }}
        .badge-not-significant {{ background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }}

        .subsection-title {{
            margin: 40px 0 20px 0;
            color: #24292e;
            font-size: 1.25em;
            font-weight: 600;
            padding-bottom: 8px;
            border-bottom: 1px solid #e1e4e8;
        }}

        .footer {{
            background: #f6f8fa;
            color: #586069;
            padding: 24px;
            text-align: center;
            border-top: 1px solid #e1e4e8;
            font-size: 0.875em;
        }}

        @media (max-width: 1200px) {{
            .state-comparison {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>HRV Correlation Analysis</h1>
            <p>Clinical Targets vs HRV Metrics - Low & High HR Activity States</p>
            <p style="font-size: 0.9em; margin-top: 10px;">Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>

        <div class="content">
            <div class="navigation">
                <h3>Quick Navigation</h3>
                <ul>
"""

    # Add navigation links
    html += '                    <li><a href="#collinearity">HRV Metrics Collinearity</a></li>\n'
    for target in results.keys():
        html += f'                    <li><a href="#{target}">{target}</a></li>\n'

    html += """
                </ul>
            </div>

            <!-- COLLINEARITY SECTION -->
            <div class="target-section" id="collinearity">
                <h2>HRV Metrics Collinearity Analysis</h2>
                <p style="margin-bottom: 24px; font-size: 0.95em; color: #586069; line-height: 1.6;">
                    Correlation matrix showing relationships between HRV features. High correlations (|r| > 0.8)
                    indicate potential multicollinearity issues for predictive modeling.
                </p>

                <div class="state-comparison">
                    <div class="state-panel">
                        <h3>Low HR Activity </h3>
"""

    if collinearity_low_hr:
        html += f"""
                        <div class="plot-container">
                            <img src="data:image/png;base64,{collinearity_low_hr}" alt="Low HR Activity Collinearity">
                        </div>
"""
    else:
        html += """
                        <p style="text-align: center; color: #999; padding: 40px;">No data available</p>
"""

    html += """
                    </div>
                    <div class="state-panel">
                        <h3>High HR Activity </h3>
"""

    if collinearity_high_hr:
        html += f"""
                        <div class="plot-container">
                            <img src="data:image/png;base64,{collinearity_high_hr}" alt="High HR Activity Collinearity">
                        </div>
"""
    else:
        html += """
                        <p style="text-align: center; color: #999; padding: 40px;">No data available</p>
"""

    html += """
                    </div>
                </div>

                <div style="background: #f6f8fa; padding: 20px; border: 1px solid #e1e4e8; border-radius: 4px; margin-top: 32px;">
                    <h4 style="color: #24292e; margin-bottom: 12px; font-size: 0.95em; font-weight: 600;">Interpretation Guide</h4>
                    <ul style="margin-left: 20px; color: #586069; line-height: 1.8; font-size: 0.9em;">
                        <li><strong>|r| > 0.9:</strong> Very high collinearity - consider removing one feature</li>
                        <li><strong>0.7 < |r| < 0.9:</strong> High collinearity - may cause issues in regression</li>
                        <li><strong>0.5 < |r| < 0.7:</strong> Moderate correlation - acceptable for most analyses</li>
                        <li><strong>|r| < 0.5:</strong> Low correlation - independent features</li>
                    </ul>
                </div>
            </div>
"""

    # Add sections for each target
    for target, target_data in results.items():
        low_hr = target_data['low_hr']
        high_hr = target_data['high_hr']

        html += f"""
            <div class="target-section" id="{target}">
                <h2>{target}</h2>

                <div class="summary-stats">
                    <div class="stat-card">
                        <h4>Low HR Activity - Samples</h4>
                        <p>{low_hr.get('n_samples', 'N/A')}</p>
                    </div>
                    <div class="stat-card">
                        <h4>High HR Activity - Samples</h4>
                        <p>{high_hr.get('n_samples', 'N/A')}</p>
                    </div>
                    <div class="stat-card">
                        <h4>Low HR - Significant</h4>
                        <p>{low_hr.get('n_significant', 0)}</p>
                    </div>
                    <div class="stat-card">
                        <h4>High HR - Significant</h4>
                        <p>{high_hr.get('n_significant', 0)}</p>
                    </div>
                </div>

                <h3 class="subsection-title">Correlation Heatmaps</h3>
                <div class="state-comparison">
                    <div class="state-panel">
                        <h3>Low HR Activity </h3>
"""

        if low_hr.get('heatmap'):
            html += f"""
                        <div class="plot-container">
                            <img src="data:image/png;base64,{low_hr['heatmap']}" alt="Low HR Activity Heatmap">
                        </div>
"""

        html += """
                    </div>
                    <div class="state-panel">
                        <h3>High HR Activity </h3>
"""

        if high_hr.get('heatmap'):
            html += f"""
                        <div class="plot-container">
                            <img src="data:image/png;base64,{high_hr['heatmap']}" alt="High HR Activity Heatmap">
                        </div>
"""

        html += """
                    </div>
                </div>

                <h3 class="subsection-title">Individual Metric Correlations</h3>
                <div class="state-comparison">
                    <div class="state-panel">
                        <h3>Low HR Activity </h3>
"""

        if low_hr.get('scatter_grid'):
            html += f"""
                        <div class="plot-container">
                            <img src="data:image/png;base64,{low_hr['scatter_grid']}" alt="Low HR Activity Scatter Plots">
                        </div>
"""

        html += """
                    </div>
                    <div class="state-panel">
                        <h3>High HR Activity </h3>
"""

        if high_hr.get('scatter_grid'):
            html += f"""
                        <div class="plot-container">
                            <img src="data:image/png;base64,{high_hr['scatter_grid']}" alt="High HR Activity Scatter Plots">
                        </div>
"""

        html += """
                    </div>
                </div>
"""

        # Add correlation comparison table
        if low_hr.get('correlations') and high_hr.get('correlations'):
            html += """
                <h3 style="margin: 30px 0 20px 0; color: #764ba2;">Correlation Comparison Table</h3>
                <table class="correlation-table">
                    <thead>
                        <tr>
                            <th>HRV Metric</th>
                            <th>Low HR - r</th>
                            <th>Low HR - p</th>
                            <th>High HR - r</th>
                            <th>High HR - p</th>
                        </tr>
                    </thead>
                    <tbody>
"""

            # Combine correlations
            for low_corr in low_hr['correlations']:
                high_corr = next((h for h in high_hr['correlations'] if h['feature'] == low_corr['feature']), None)

                low_r = low_corr['r']
                low_p = low_corr['p']
                low_sig_badge = '<span class="badge badge-significant">Sig</span>' if low_p < 0.05 else '<span class="badge badge-not-significant">NS</span>'

                if high_corr:
                    high_r = high_corr['r']
                    high_p = high_corr['p']
                    high_sig_badge = '<span class="badge badge-significant">Sig</span>' if high_p < 0.05 else '<span class="badge badge-not-significant">NS</span>'
                else:
                    high_r = 0
                    high_p = 1
                    high_sig_badge = '<span class="badge badge-not-significant">N/A</span>'

                html += f"""
                        <tr>
                            <td><strong>{low_corr['feature']}</strong></td>
                            <td>{low_r:.3f} {low_sig_badge}</td>
                            <td>{low_p:.4f}</td>
                            <td>{high_r:.3f} {high_sig_badge}</td>
                            <td>{high_p:.4f}</td>
                        </tr>
"""

            html += """
                    </tbody>
                </table>
"""

        html += """
            </div>
"""

    # Close HTML
    html += """
        </div>

        <div class="footer">
            <p>HRV Correlation Analysis | Low & High HR Activity States</p>
        </div>
    </div>
</body>
</html>
"""

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"\n✓ HTML report saved to '{output_file}'")


# ===========================================================================
# MAIN EXECUTION
# ===========================================================================

if __name__ == "__main__":
    # Load data
    clinical_data_df = pd.read_csv(r"C:\Users\spbtu\Documents\dataset_psath\bbi_metadata\clinical_d32_T0_v2.csv")
    df_sl = pd.read_csv(r"C:\Users\spbtu\Documents\hrvThesis\window90m_hrv_mean_measurements_sl.csv", index_col=0)
    df_aw = pd.read_csv(r"C:\Users\spbtu\Documents\hrvThesis\window90m_hrv_mean_measurements_aw.csv", index_col=0)

    # Define features and targets
    all_features = [
        "HRV_RMSSD", "HRV_SDNN", "HRV_HTI",
        "HRV_ULF", "HRV_VLF", "HRV_LF", "HRV_HF", "HRV_VHF",
        "HRV_TP", "HRV_LFHF", "HRV_LFn", "HRV_HFn", "HRV_LnHF"
    ]

    target_list = ["DOC_PAT_BMI", "DAPSA_Score", "PASDAS", "PSAID_Final_Score", "CRP_mg_dL", "DEMOGR_AGE", "Overall HAQ Score", "DOC_VAS_H"]

    print("="*80)
    print("CREATING HRV CORRELATION ANALYSIS REPORT")
    print("="*80)

    # Create collinearity heatmaps first
    print("\n" + "="*80)
    print("Creating HRV Collinearity Heatmaps...")
    print("="*80)

    print("  - Low HR Activity ...")
    fig_collinearity_low = create_hrv_collinearity_heatmap(df_sl, all_features, "Low HR Activity ")
    collinearity_low_hr = fig_to_base64(fig_collinearity_low) if fig_collinearity_low else None

    print("  - High HR Activity ...")
    fig_collinearity_high = create_hrv_collinearity_heatmap(df_aw, all_features, "High HR Activity ")
    collinearity_high_hr = fig_to_base64(fig_collinearity_high) if fig_collinearity_high else None

    print("  ✓ Collinearity analysis complete")

    results = {}

    # Process each target
    for target in target_list:
        print(f"\nProcessing {target}...")

        results[target] = {
            'low_hr': {},
            'high_hr': {}
        }

        # Low HR Activity 
        print(f"  - Low HR Activity ...")
        fig_scatter, correlations = create_target_vs_all_metrics(df_sl, clinical_data_df, target, all_features)
        if fig_scatter:
            results[target]['low_hr']['scatter_grid'] = fig_to_base64(fig_scatter)
            results[target]['low_hr']['correlations'] = correlations
            results[target]['low_hr']['n_samples'] = len(correlations) if correlations else 0
            results[target]['low_hr']['n_significant'] = sum(1 for c in correlations if c['p'] < 0.05)

        fig_heatmap = create_correlation_heatmap(df_sl, clinical_data_df, target, all_features)
        if fig_heatmap:
            results[target]['low_hr']['heatmap'] = fig_to_base64(fig_heatmap)

        # High HR Activity 
        print(f"  - High HR Activity ...")
        fig_scatter, correlations = create_target_vs_all_metrics(df_aw, clinical_data_df, target, all_features)
        if fig_scatter:
            results[target]['high_hr']['scatter_grid'] = fig_to_base64(fig_scatter)
            results[target]['high_hr']['correlations'] = correlations
            results[target]['high_hr']['n_samples'] = len(correlations) if correlations else 0
            results[target]['high_hr']['n_significant'] = sum(1 for c in correlations if c['p'] < 0.05)

        fig_heatmap = create_correlation_heatmap(df_aw, clinical_data_df, target, all_features)
        if fig_heatmap:
            results[target]['high_hr']['heatmap'] = fig_to_base64(fig_heatmap)

    # Generate HTML report
    print("\n" + "="*80)
    print("Generating HTML report...")
    print("="*80)
    create_html_report(results, collinearity_low_hr, collinearity_high_hr,
                      output_file='hrv_correlation_report_w90m.html')

    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE")
    print("="*80)
    print("\nGenerated:")
    print("  - hrv_correlation_report_w90m.html")
    print("\nThe report includes:")
    print("  - HRV metrics collinearity heatmaps (Low & High HR Activity)")
    print("  - Correlation heatmaps for each clinical target")
    print("  - Individual scatter plots for all HRV metrics")
    print("  - Comparison tables with significance tests")
    print(f"  - Coverage: {len(target_list)} clinical targets")
