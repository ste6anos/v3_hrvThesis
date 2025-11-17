import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from pysr import PySRRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import base64
from io import BytesIO
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def calculate_vif(df):
    """Calculate VIF for each feature to detect multicollinearity."""
    vif_data = pd.DataFrame()
    vif_data["Feature"] = df.columns
    vif_data["VIF"] = [variance_inflation_factor(df.values, i) for i in range(len(df.columns))]
    return vif_data.sort_values('VIF', ascending=False)


def remove_collinear_features(df, threshold=5.0):
    """
    Remove features with high VIF (multicollinearity).
    VIF > 5: moderate collinearity
    VIF > 10: high collinearity
    """
    print(f"\n{'='*60}")
    print("COLLINEARITY ANALYSIS (VIF)")
    print(f"{'='*60}")

    features = df.copy()
    removed_features = []

    while True:
        vif_data = calculate_vif(features)
        print(f"\n{vif_data.to_string(index=False)}")

        max_vif = vif_data['VIF'].max()

        if max_vif > threshold:
            feature_to_remove = vif_data.loc[vif_data['VIF'].idxmax(), 'Feature']
            print(f"\nRemoving '{feature_to_remove}' (VIF={max_vif:.2f})")
            features = features.drop(columns=[feature_to_remove])
            removed_features.append(feature_to_remove)
        else:
            break

    print(f"\nFinal features: {list(features.columns)}")
    if removed_features:
        print(f"Removed features: {removed_features}")

    return features, removed_features


def cap_outliers(df, z_threshold=3):
    """Cap outliers using z-score method."""
    df_capped = df.copy()

    for col in df_capped.columns:
        mean = df_capped[col].mean()
        std = df_capped[col].std()

        if std > 0:
            upper_limit = mean + z_threshold * std
            lower_limit = mean - z_threshold * std

            n_capped = ((df_capped[col] > upper_limit) | (df_capped[col] < lower_limit)).sum()

            if n_capped > 0:
                df_capped[col] = df_capped[col].clip(lower_limit, upper_limit)
                print(f"  {col}: Capped {n_capped} outliers (±{z_threshold}σ)")

    return df_capped


def preprocess_data(X, y, apply_vif=True, apply_scaling=False, apply_capping=True, vif_threshold=5.0):
    """
    Preprocess features: VIF removal, capping, optional scaling.

    Parameters:
    - X: Features DataFrame
    - y: Target series
    - apply_vif: Whether to apply VIF-based collinearity filtering
    - apply_scaling: Whether to apply StandardScaler
    - apply_capping: Whether to cap outliers with z-score
    - vif_threshold: VIF threshold for collinearity removal
    """
    print(f"\n{'='*60}")
    print("PREPROCESSING PIPELINE")
    print(f"{'='*60}")
    print(f"Initial features: {X.shape[1]}")
    print(f"Samples: {X.shape[0]}")

    # Step 1: Remove collinear features (optional)
    X_reduced = X.copy()
    removed_features = []

    if apply_vif:
        X_reduced, removed_features = remove_collinear_features(X, threshold=vif_threshold)
    else:
        print(f"\n{'='*60}")
        print("VIF FILTERING: DISABLED")
        print(f"{'='*60}")
        print("Keeping all features without collinearity check")

    # Step 2: Cap outliers
    if apply_capping:
        print(f"\n{'='*60}")
        print("CAPPING OUTLIERS (Z-score = 3)")
        print(f"{'='*60}")
        X_reduced = cap_outliers(X_reduced, z_threshold=3)

    # Step 3: Optional scaling
    scaler = None
    if apply_scaling:
        print(f"\n{'='*60}")
        print("APPLYING STANDARDSCALER")
        print(f"{'='*60}")
        scaler = StandardScaler()
        X_reduced = pd.DataFrame(
            scaler.fit_transform(X_reduced),
            columns=X_reduced.columns,
            index=X_reduced.index
        )
        print("Features scaled to mean=0, std=1")

    print(f"\n{'='*60}")
    print(f"Preprocessing complete")
    print(f"  Final features: {X_reduced.shape[1]}")
    print(f"  Removed: {len(removed_features)} features")
    print(f"{'='*60}\n")

    return X_reduced, removed_features, scaler


def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string."""
    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode()
    plt.close(fig)
    return img_str


def create_html_report(results, config, output_file='symbolic_regression_report_dual_state_90m.html'):
    """Generate professional HTML report for symbolic regression results with side-by-side comparison."""

    # Calculate summary statistics for both states
    all_low_r2 = [state_res['low_hr']['test_r2'] for state_res in results.values() if 'low_hr' in state_res]
    all_high_r2 = [state_res['high_hr']['test_r2'] for state_res in results.values() if 'high_hr' in state_res]

    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Symbolic Regression - Dual State Analysis</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', sans-serif;
            line-height: 1.6;
            color: #24292e;
            background: #f8f9fa;
            padding: 20px;
        }}

        .container {{
            max-width: 1600px;
            margin: 0 auto;
            background: white;
            border-radius: 4px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            overflow: hidden;
        }}

        .header {{
            background: white;
            color: #2c3e50;
            padding: 40px;
            text-align: center;
            border-bottom: 1px solid #e1e4e8;
        }}

        .header h1 {{
            font-size: 2em;
            margin-bottom: 10px;
            font-weight: 600;
        }}

        .header p {{
            font-size: 1em;
            color: #586069;
        }}

        .content {{
            padding: 40px;
        }}

        .config-section {{
            background: #fafbfc;
            border: 1px solid #e1e4e8;
            border-radius: 4px;
            padding: 20px;
            margin-bottom: 40px;
        }}

        .config-section h3 {{
            font-size: 1.125em;
            font-weight: 600;
            margin-bottom: 16px;
            color: #24292e;
        }}

        .config-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 12px;
        }}

        .summary-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 16px;
            margin-bottom: 40px;
        }}

        .stat-card {{
            background: white;
            padding: 20px;
            border: 1px solid #e1e4e8;
            border-radius: 4px;
            text-align: center;
        }}

        .stat-card h3 {{
            font-size: 0.875em;
            color: #586069;
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

        .target-section {{
            margin-bottom: 60px;
            padding: 30px;
            background: #fafbfc;
            border: 1px solid #e1e4e8;
            border-radius: 4px;
        }}

        .target-section h2 {{
            color: #24292e;
            margin-bottom: 24px;
            font-size: 1.75em;
            font-weight: 600;
            padding-bottom: 8px;
            border-bottom: 1px solid #e1e4e8;
        }}

        .state-comparison {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
            margin-top: 20px;
        }}

        .state-panel {{
            background: white;
            border: 1px solid #e1e4e8;
            border-radius: 4px;
            padding: 20px;
        }}

        .state-panel h3 {{
            font-size: 1.25em;
            font-weight: 600;
            margin-bottom: 16px;
            color: #24292e;
            padding-bottom: 8px;
            border-bottom: 1px solid #e1e4e8;
        }}

        .subsection-title {{
            margin: 24px 0 12px 0;
            color: #24292e;
            font-size: 1.125em;
            font-weight: 600;
            padding-bottom: 8px;
            border-bottom: 1px solid #e1e4e8;
        }}

        .equation-box {{
            background: #f6f8fa;
            padding: 16px;
            border-radius: 4px;
            margin: 12px 0;
            font-family: 'Courier New', monospace;
            font-size: 0.95em;
            border: 1px solid #e1e4e8;
            overflow-x: auto;
            color: #24292e;
        }}

        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 12px;
            margin: 16px 0;
        }}

        .metric-item {{
            background: #f6f8fa;
            padding: 12px;
            border-radius: 4px;
            border: 1px solid #e1e4e8;
        }}

        .metric-item label {{
            display: block;
            font-size: 0.875em;
            color: #586069;
            margin-bottom: 4px;
            font-weight: 600;
        }}

        .metric-item value {{
            display: block;
            font-size: 1.25em;
            font-weight: 600;
            color: #24292e;
        }}

        .plot-container {{
            margin: 16px 0;
            text-align: center;
        }}

        .plot-container img {{
            max-width: 100%;
            border-radius: 4px;
            border: 1px solid #e1e4e8;
        }}

        .pareto-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 16px 0;
            background: white;
            border: 1px solid #e1e4e8;
            border-radius: 4px;
            overflow: hidden;
        }}

        .pareto-table th {{
            background: #f6f8fa;
            color: #24292e;
            padding: 10px;
            text-align: left;
            font-weight: 600;
            font-size: 0.875em;
            border-bottom: 1px solid #e1e4e8;
        }}

        .pareto-table td {{
            padding: 10px;
            border-bottom: 1px solid #e1e4e8;
            font-size: 0.875em;
        }}

        .pareto-table tr:hover {{
            background: #f6f8fa;
        }}

        .footer {{
            background: #f6f8fa;
            color: #586069;
            padding: 20px;
            text-align: center;
            border-top: 1px solid #e1e4e8;
        }}

        .badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 3px;
            font-size: 0.75em;
            font-weight: 600;
            margin-left: 8px;
        }}

        .badge-excellent {{ background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }}
        .badge-good {{ background: #d1ecf1; color: #0c5460; border: 1px solid #bee5eb; }}
        .badge-moderate {{ background: #fff3cd; color: #856404; border: 1px solid #ffeeba; }}
        .badge-poor {{ background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }}

        .preprocessing-info {{
            background: #f6f8fa;
            border: 1px solid #e1e4e8;
            border-radius: 4px;
            padding: 12px;
            margin: 12px 0;
            font-size: 0.875em;
        }}

        .preprocessing-info p {{
            margin: 4px 0;
            color: #24292e;
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
            <h1>Symbolic Regression Analysis - Dual State Comparison</h1>
            <p>HRV Metrics Predictive Modeling: Low vs High HR Activity</p>
            <p style="font-size: 0.9em; margin-top: 10px;">Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>

        <div class="content">
            <div class="config-section">
                <h3>Preprocessing Configuration</h3>
                <div class="config-grid">
                    <p><strong>VIF Filtering:</strong> {'Enabled' if config['apply_vif'] else 'Disabled'}</p>
                    <p><strong>Outlier Capping:</strong> {'Enabled (Z=3)' if config['apply_capping'] else 'Disabled'}</p>
                    <p><strong>Scaling:</strong> {'StandardScaler' if config['apply_scaling'] else 'Disabled'}</p>
                    <p><strong>VIF Threshold:</strong> {config['vif_threshold']}</p>
                </div>
            </div>

            <h2 style="margin-bottom: 20px; font-size: 1.5em; font-weight: 600;">Summary Statistics</h2>
            <div class="summary-stats">
                <div class="stat-card">
                    <h3>Total Targets</h3>
                    <p>{len(results)}</p>
                </div>
                <div class="stat-card">
                    <h3>Low HR Avg R²</h3>
                    <p>{np.mean(all_low_r2):.3f}</p>
                </div>
                <div class="stat-card">
                    <h3>High HR Avg R²</h3>
                    <p>{np.mean(all_high_r2):.3f}</p>
                </div>
                <div class="stat-card">
                    <h3>Best Low HR R²</h3>
                    <p>{max(all_low_r2) if all_low_r2 else 0:.3f}</p>
                </div>
                <div class="stat-card">
                    <h3>Best High HR R²</h3>
                    <p>{max(all_high_r2) if all_high_r2 else 0:.3f}</p>
                </div>
            </div>
"""

    # Helper function for badge determination
    def get_badge(r2):
        if r2 >= 0.7:
            return '<span class="badge badge-excellent">Excellent</span>'
        elif r2 >= 0.5:
            return '<span class="badge badge-good">Good</span>'
        elif r2 >= 0.3:
            return '<span class="badge badge-moderate">Moderate</span>'
        else:
            return '<span class="badge badge-poor">Poor</span>'

    # Add individual target sections with side-by-side comparison
    for target, state_results in results.items():
        low_hr_res = state_results.get('low_hr', {})
        high_hr_res = state_results.get('high_hr', {})

        low_r2 = low_hr_res.get('test_r2', 0)
        high_r2 = high_hr_res.get('test_r2', 0)

        low_badge = get_badge(low_r2)
        high_badge = get_badge(high_r2)

        html += f"""
            <div class="target-section">
                <h2>{target}</h2>

                <div class="state-comparison">
                    <!-- Low HR Activity Panel -->
                    <div class="state-panel">
                        <h3>Low HR Activity (Sleep) {low_badge}</h3>

                        <div class="preprocessing-info">
                            <p><strong>Samples:</strong> {low_hr_res.get('n_samples', 'N/A')}</p>
                            <p><strong>Features Used:</strong> {', '.join(low_hr_res.get('features_used', []))}</p>
"""

        # Add VIF info for low HR
        if config['apply_vif']:
            features_removed_low = ', '.join(low_hr_res.get('features_removed', [])) if low_hr_res.get('features_removed') else 'None'
            html += f"""                            <p><strong>Features Removed (VIF):</strong> {features_removed_low}</p>
"""
        else:
            html += """                            <p><strong>VIF Filtering:</strong> Disabled</p>
"""

        html += f"""                        </div>

                        <h4 class="subsection-title">Best Equation</h4>
                        <div class="equation-box">
                            {low_hr_res.get('equation', 'N/A')}
                        </div>

                        <div class="metrics-grid">
                            <div class="metric-item">
                                <label>Train R²</label>
                                <value>{low_hr_res.get('train_r2', 0):.4f}</value>
                            </div>
                            <div class="metric-item">
                                <label>Test R²</label>
                                <value>{low_hr_res.get('test_r2', 0):.4f}</value>
                            </div>
                            <div class="metric-item">
                                <label>Train RMSE</label>
                                <value>{low_hr_res.get('train_rmse', 0):.4f}</value>
                            </div>
                            <div class="metric-item">
                                <label>Test RMSE</label>
                                <value>{low_hr_res.get('test_rmse', 0):.4f}</value>
                            </div>
                        </div>

                        <div class="plot-container">
                            <img src="data:image/png;base64,{low_hr_res.get('plot', '')}" alt="Low HR Predictions">
                        </div>

                        <h4 class="subsection-title">Top 5 Equations (Pareto Front)</h4>
                        <table class="pareto-table">
                            <thead>
                                <tr>
                                    <th>Rank</th>
                                    <th>Complexity</th>
                                    <th>Loss</th>
                                    <th>Equation</th>
                                </tr>
                            </thead>
                            <tbody>
"""

        # Add Pareto front for low HR
        if 'pareto_front' in low_hr_res:
            for idx, row in low_hr_res['pareto_front'].head(5).iterrows():
                html += f"""
                                <tr>
                                    <td>{idx + 1}</td>
                                    <td>{row['complexity']}</td>
                                    <td>{row['loss']:.6f}</td>
                                    <td style="font-family: monospace; font-size: 0.8em;">{row['equation']}</td>
                                </tr>
"""

        html += """
                            </tbody>
                        </table>
                    </div>

                    <!-- High HR Activity Panel -->
                    <div class="state-panel">
                        <h3>High HR Activity (Awake) """ + high_badge + """</h3>

                        <div class="preprocessing-info">
                            <p><strong>Samples:</strong> """ + str(high_hr_res.get('n_samples', 'N/A')) + """</p>
                            <p><strong>Features Used:</strong> """ + ', '.join(high_hr_res.get('features_used', [])) + """</p>
"""

        # Add VIF info for high HR
        if config['apply_vif']:
            features_removed_high = ', '.join(high_hr_res.get('features_removed', [])) if high_hr_res.get('features_removed') else 'None'
            html += f"""                            <p><strong>Features Removed (VIF):</strong> {features_removed_high}</p>
"""
        else:
            html += """                            <p><strong>VIF Filtering:</strong> Disabled</p>
"""

        html += """                        </div>

                        <h4 class="subsection-title">Best Equation</h4>
                        <div class="equation-box">
                            """ + high_hr_res.get('equation', 'N/A') + """
                        </div>

                        <div class="metrics-grid">
                            <div class="metric-item">
                                <label>Train R²</label>
                                <value>""" + f"{high_hr_res.get('train_r2', 0):.4f}" + """</value>
                            </div>
                            <div class="metric-item">
                                <label>Test R²</label>
                                <value>""" + f"{high_hr_res.get('test_r2', 0):.4f}" + """</value>
                            </div>
                            <div class="metric-item">
                                <label>Train RMSE</label>
                                <value>""" + f"{high_hr_res.get('train_rmse', 0):.4f}" + """</value>
                            </div>
                            <div class="metric-item">
                                <label>Test RMSE</label>
                                <value>""" + f"{high_hr_res.get('test_rmse', 0):.4f}" + """</value>
                            </div>
                        </div>

                        <div class="plot-container">
                            <img src="data:image/png;base64,""" + high_hr_res.get('plot', '') + """" alt="High HR Predictions">
                        </div>

                        <h4 class="subsection-title">Top 5 Equations (Pareto Front)</h4>
                        <table class="pareto-table">
                            <thead>
                                <tr>
                                    <th>Rank</th>
                                    <th>Complexity</th>
                                    <th>Loss</th>
                                    <th>Equation</th>
                                </tr>
                            </thead>
                            <tbody>
"""

        # Add Pareto front for high HR
        if 'pareto_front' in high_hr_res:
            for idx, row in high_hr_res['pareto_front'].head(5).iterrows():
                html += f"""
                                <tr>
                                    <td>{idx + 1}</td>
                                    <td>{row['complexity']}</td>
                                    <td>{row['loss']:.6f}</td>
                                    <td style="font-family: monospace; font-size: 0.8em;">{row['equation']}</td>
                                </tr>
"""

        html += """
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
"""

    # Close HTML
    html += """
        </div>

        <div class="footer">
            <p>Symbolic Regression with PySR | Dual State HRV Metrics Analysis</p>
        </div>
    </div>
</body>
</html>
"""

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"\nHTML report saved to '{output_file}'")


# Main execution
if __name__ == "__main__":
    # ========== CONFIGURATION ==========
    APPLY_VIF = False           # Set to True to enable VIF-based collinearity filtering
    APPLY_SCALING = False      # Set to True to enable StandardScaler
    APPLY_CAPPING = False       # Set to True to enable outlier capping (Z=3)
    VIF_THRESHOLD = 5.0        # Collinearity threshold (5.0 = moderate, 10.0 = high)

    print("="*60)
    print("SYMBOLIC REGRESSION - DUAL STATE ANALYSIS")
    print("="*60)
    print(f"VIF Filtering: {APPLY_VIF}")
    print(f"Scaling: {APPLY_SCALING}")
    print(f"Capping: {APPLY_CAPPING}")
    print(f"VIF Threshold: {VIF_THRESHOLD}")
    print("="*60)

    # Load data
    clinical_data_df = pd.read_csv(r"C:\Users\spbtu\Documents\dataset_psath\bbi_metadata\clinical_d32_T0_v2.csv")

    # Load both states
    X_df_aw = pd.read_csv("window90m_hrv_mean_measurements_aw.csv", index_col=0)
    X_df_aw = X_df_aw.iloc[:, :-1]

    X_df_sl = pd.read_csv("window90m_hrv_mean_measurements_sl.csv", index_col=0)
    X_df_sl = X_df_sl.iloc[:, :-1]



    y_df = clinical_data_df[["patientid", "DOC_PAT_BMI", "DAPSA_Score",
                             "PASDAS", "PSAID_Final_Score", "CRP_mg_dL", "Overall HAQ Score", "DOC_VAS_H"]]

    results = {}

    # Train models for each target - DUAL STATE PROCESSING
    for target in y_df.columns[1:]:
        print(f"\n{'='*60}")
        print(f"TARGET: {target}")
        print(f"{'='*60}")

        # Initialize nested results structure
        results[target] = {
            'low_hr': {},
            'high_hr': {}
        }

        # Process both states
        for state_name, X_df in [('low_hr', X_df_sl), ('high_hr', X_df_aw)]:
            state_label = 'Low HR Activity' if state_name == 'low_hr' else 'High HR Activity'

            print(f"\n--- Processing {state_label} ---")

            # Merge and clean data
            df = pd.merge(X_df, y_df[["patientid", target]], on="patientid")
            df = df.dropna(subset=[target])
            df = df.drop(["patientid"], axis=1)

            # Separate features and target
            X_raw = df.iloc[:, :-1]
            y_raw = df.iloc[:, -1]

            # Preprocess features (VIF + capping + optional scaling)
            X_processed, removed_features, scaler = preprocess_data(
                X_raw, y_raw,
                apply_vif=APPLY_VIF,
                apply_scaling=APPLY_SCALING,
                apply_capping=APPLY_CAPPING,
                vif_threshold=VIF_THRESHOLD
            )

            # Get final variable names after preprocessing
            final_variable_names = X_processed.columns.tolist()

            # Convert to numpy arrays
            X = X_processed.values
            y = y_raw.values

            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Configure PySR with optimized hyperparameters for small datasets
            model = PySRRegressor(
                niterations=40,              # Reduced for faster convergence
                binary_operators=["+", "-", "*", "/"],
                unary_operators=["square", "cube", "sqrt", "log", "exp"],  # Simplified operators
                populations=15,              # Reduced populations
                population_size=33,          # Smaller population
                maxsize=10,                  # Allow slightly more complex equations
                parsimony=0.01,             # Lower parsimony for better fit
                model_selection="best",
                verbosity=1,
                random_state=42,
                deterministic=True,
                parallelism='serial',
                temp_equation_file=False,
                delete_tempfiles=True,
                # Additional parameters for better performance
                ncyclesperiteration=550,    # More cycles per iteration
                weight_optimize=0.001,       # Focus on optimization
                timeout_in_seconds=300       # 5 min timeout per target
            )

            print(f"\n{'='*60}")
            print(f"TRAINING MODEL FOR {target} - {state_label}")
            print(f"Features: {final_variable_names}")
            print(f"Samples: {len(y_train)} train, {len(y_test)} test")
            print(f"{'='*60}")

            model.fit(X_train, y_train, variable_names=final_variable_names)

            # Predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            # Metrics
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

            # Create prediction plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # Train plot
            ax1.scatter(y_train, y_pred_train, alpha=0.6, edgecolors='k', linewidth=0.5)
            ax1.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()],
                     'r--', lw=2, label='Perfect prediction')
            ax1.set_xlabel('Actual Values', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Predicted Values', fontsize=12, fontweight='bold')
            ax1.set_title(f'Training Set (R² = {train_r2:.3f})', fontsize=14, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Test plot
            ax2.scatter(y_test, y_pred_test, alpha=0.6, edgecolors='k', linewidth=0.5, color='orange')
            ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                     'r--', lw=2, label='Perfect prediction')
            ax2.set_xlabel('Actual Values', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Predicted Values', fontsize=12, fontweight='bold')
            ax2.set_title(f'Test Set (R² = {test_r2:.3f})', fontsize=14, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            fig.suptitle(f'{target} - {state_label} - Predictions vs Actual', fontsize=16, fontweight='bold')
            plt.tight_layout()

            plot_base64 = fig_to_base64(fig)

            # Get best equation
            best_equation = model.sympy()

            # Get Pareto front
            pareto_df = model.equations_[['complexity', 'loss', 'equation']].copy()

            # Store results for this state
            results[target][state_name] = {
                'model': model,
                'equation': str(best_equation),
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'complexity': pareto_df.iloc[0]['complexity'] if len(pareto_df) > 0 else 0,
                'plot': plot_base64,
                'pareto_front': pareto_df,
                'features_used': final_variable_names,
                'features_removed': removed_features,
                'n_samples': len(y)
            }

            print(f"\n{'='*60}")
            print(f"COMPLETED {target} - {state_label}")
            print(f"{'='*60}")
            print(f"  Test R²: {test_r2:.4f}")
            print(f"  Test RMSE: {test_rmse:.4f}")
            print(f"  Features used: {len(final_variable_names)}")
            print(f"  Best equation: {str(best_equation)}")
            print(f"{'='*60}")

    # Generate HTML report
    print(f"\n{'='*60}")
    print("Generating HTML report...")
    print(f"{'='*60}")

    # Prepare configuration dictionary for HTML report
    config = {
        'apply_vif': APPLY_VIF,
        'apply_scaling': APPLY_SCALING,
        'apply_capping': APPLY_CAPPING,
        'vif_threshold': VIF_THRESHOLD
    }

    create_html_report(results, config, output_file='symbolic_regression_report_dual_state_90m.html')

    # Save summary CSV
    summary_data = []
    for target, state_results in results.items():
        # Low HR Activity
        if 'low_hr' in state_results:
            summary_data.append({
                'Target': target,
                'HR_Activity': 'Low',
                'Test_R2': state_results['low_hr']['test_r2'],
                'Test_RMSE': state_results['low_hr']['test_rmse'],
                'Complexity': state_results['low_hr']['complexity'],
                'Equation': state_results['low_hr']['equation']
            })

        # High HR Activity
        if 'high_hr' in state_results:
            summary_data.append({
                'Target': target,
                'HR_Activity': 'High',
                'Test_R2': state_results['high_hr']['test_r2'],
                'Test_RMSE': state_results['high_hr']['test_rmse'],
                'Complexity': state_results['high_hr']['complexity'],
                'Equation': state_results['high_hr']['equation']
            })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('symbolic_regression_summary_dual_state.csv', index=False)
    print("\nSummary CSV saved to 'symbolic_regression_summary_dual_state.csv'")
    print("\nDual-state analysis complete!")
