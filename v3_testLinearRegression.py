from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np
from v3_utils import StatsUtils
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
from scipy import stats

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load data
clinical_data_df = pd.read_csv(r"C:\Users\spbtu\Documents\dataset_psath\bbi_metadata\clinical_d32_T0_v2.csv")
df_sl = pd.read_csv(r"C:\Users\spbtu\Documents\hrvThesis\window90m_hrv_mean_measurements_aw.csv", index_col=0)

# HRV features
hrv_features = [
    "HRV_RMSSD", "HRV_SDNN", "HRV_HTI",
    "HRV_ULF", "HRV_VLF", "HRV_LF", "HRV_HF", "HRV_VHF",
    "HRV_TP", "HRV_LFHF", "HRV_LFn", "HRV_HFn", "HRV_LnHF"
]

# Clinical covariate: CRP
clinical_covariate = "CRP_mg_dL"

# All features = HRV + CRP
all_features = hrv_features + [clinical_covariate]

# Targets to predict
target_list = ["DOC_PAT_BMI", "DAPSA_Score", "PASDAS", "PSAID_Final_Score"]

# Parameters
kf = KFold(n_splits=5, shuffle=True, random_state=42)
alphas = np.logspace(-3, 3, 20)

# Store results
all_results = {}

print("="*80)
print("RIDGE REGRESSION WITH CRP AS COVARIATE")
print("="*80)
print(f"\nHRV Features: {len(hrv_features)}")
print(f"Clinical Covariate: {clinical_covariate}")
print(f"Total Features: {len(all_features)} (HRV + CRP)")
print(f"Targets: {len(target_list)}")
print("="*80)

# ============================================================================
# PREPROCESSING FUNCTIONS
# ============================================================================

def z_score_capping(data, z_threshold=3):
    """
    Cap outliers using Z-score method
    Values beyond ±z_threshold standard deviations are capped
    
    Parameters:
    -----------
    data : array-like
        Input data
    z_threshold : float
        Z-score threshold (default: 3)
    
    Returns:
    --------
    capped_data : array
        Data with outliers capped
    n_outliers : int
        Number of outliers detected
    """
    data = np.array(data)
    mean = np.mean(data)
    std = np.std(data)
    
    # Calculate Z-scores
    z_scores = np.abs((data - mean) / std)
    
    # Find outliers
    outliers_mask = z_scores > z_threshold
    n_outliers = np.sum(outliers_mask)
    
    # Cap outliers
    lower_bound = mean - z_threshold * std
    upper_bound = mean + z_threshold * std
    
    capped_data = np.copy(data)
    capped_data[data < lower_bound] = lower_bound
    capped_data[data > upper_bound] = upper_bound
    
    return capped_data, n_outliers

def normalize_features_0_1(df, feature_cols):
    """
    Normalize features to [0, 1] range using MinMaxScaler
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe
    feature_cols : list
        List of feature column names to normalize
    
    Returns:
    --------
    df_normalized : DataFrame
        Dataframe with normalized features
    scaler : MinMaxScaler
        Fitted scaler object
    """
    df_normalized = df.copy()
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    df_normalized[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    return df_normalized, scaler

print("\n" + "="*80)
print("PREPROCESSING PIPELINE")
print("="*80)
print("1. Z-score capping (Z=3) for targets and CRP")
print("2. Min-Max normalization [0,1] for all features (HRV + CRP)")
print("="*80)

def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string for HTML embedding"""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_str

def find_best_alpha(X_scaled, y, alphas, kf):
    """Find best alpha via cross-validation"""
    best_alpha_score = -np.inf
    best_alpha = alphas[0]
    
    for alpha in alphas:
        model = Ridge(alpha=alpha, random_state=42)
        cv_scores = []
        
        for train_idx, val_idx in kf.split(X_scaled):
            X_train_cv, X_val_cv = X_scaled[train_idx], X_scaled[val_idx]
            y_train_cv, y_val_cv = y[train_idx], y[val_idx]
            
            model.fit(X_train_cv, y_train_cv)
            y_pred_cv = model.predict(X_val_cv)
            cv_scores.append(r2_score(y_val_cv, y_pred_cv))
        
        mean_cv_score = np.mean(cv_scores)
        
        if mean_cv_score > best_alpha_score:
            best_alpha_score = mean_cv_score
            best_alpha = alpha
    
    return best_alpha, best_alpha_score

def exhaustive_forward_selection(df_clean, target_name, all_features, clinical_covariate, kf, alphas):
    """Exhaustive forward selection"""
    y = df_clean[target_name].values
    results_by_size = {}
    current_best_features = []
    remaining_features = all_features.copy()
    
    print(f"\nEXHAUSTIVE FORWARD SELECTION: {target_name}")
    print("="*80)
    
    for size in range(1, len(all_features) + 1):
        print(f"Testing size {size}/{len(all_features)}...", end=' ')
        
        size_results = []
        
        for feat in remaining_features:
            candidate_features = current_best_features + [feat]
            
            X = df_clean[candidate_features]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            best_alpha, cv_score = find_best_alpha(X_scaled, y, alphas, kf)
            
            size_results.append({
                'features': candidate_features.copy(),
                'new_feature': feat,
                'cv_score': cv_score,
                'best_alpha': best_alpha,
                'size': size
            })
        
        best_for_size = max(size_results, key=lambda x: x['cv_score'])
        current_best_features = best_for_size['features'].copy()
        remaining_features.remove(best_for_size['new_feature'])
        results_by_size[size] = best_for_size
        
        # Mark if CRP was selected
        feat_name = best_for_size['new_feature']
        if feat_name == clinical_covariate:
            marker = " 🩺 (CRP!)"
        else:
            marker = ""
            feat_name = feat_name.replace('HRV_', '')
        
        print(f"Best: {feat_name}{marker}, CV R²: {best_for_size['cv_score']:.4f}")
    
    best_overall = max(results_by_size.values(), key=lambda x: x['cv_score'])
    
    print(f"\nBest: {best_overall['size']} features, CV R²: {best_overall['cv_score']:.4f}")
    
    # Check if CRP is in the best model
    if clinical_covariate in best_overall['features']:
        crp_position = best_overall['features'].index(clinical_covariate) + 1
        print(f"✓ CRP selected at position {crp_position}")
    else:
        print(f"✗ CRP NOT selected in best model")
    
    return results_by_size, best_overall

def create_target_metric_scatter(df_clean, target_name, all_features, clinical_covariate):
    """Create scatter plots: each HRV metric + CRP vs target"""
    n_features = len(all_features)
    n_cols = 4
    n_rows = int(np.ceil(n_features / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
    axes = axes.flatten()
    
    for idx, feature in enumerate(all_features):
        ax = axes[idx]
        
        x = df_clean[feature].values
        y = df_clean[target_name].values
        
        # Color: red for CRP, blue for HRV
        color = 'red' if feature == clinical_covariate else 'steelblue'
        
        ax.scatter(x, y, alpha=0.6, s=40, edgecolors='k', linewidth=0.5, color=color)
        
        # Trend line
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x.min(), x.max(), 100)
        line_color = 'darkred' if feature == clinical_covariate else 'blue'
        ax.plot(x_line, p(x_line), linestyle='--', alpha=0.8, linewidth=2, color=line_color)
        
        # Correlation
        from scipy import stats
        r, p_val = stats.pearsonr(x, y)
        
        # Color code by significance
        if p_val < 0.001:
            sig_text = '***'
            box_color = 'lightgreen'
        elif p_val < 0.01:
            sig_text = '**'
            box_color = 'lightgreen'
        elif p_val < 0.05:
            sig_text = '*'
            box_color = 'lightyellow'
        else:
            sig_text = 'ns'
            box_color = 'lightgray'
        
        ax.text(0.05, 0.95, f'r={r:.3f}{sig_text}\np={p_val:.4f}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor=box_color, alpha=0.8),
                fontsize=9)
        
        feat_label = '🩺 CRP' if feature == clinical_covariate else feature.replace('HRV_', '')
        ax.set_xlabel(feat_label, fontsize=10)
        ax.set_ylabel(target_name if idx % n_cols == 0 else '', fontsize=10)
        ax.set_title(f'{feat_label}', fontsize=11, fontweight='bold')
        ax.grid(alpha=0.3)
    
    # Remove empty subplots
    for idx in range(n_features, len(axes)):
        fig.delaxes(axes[idx])
    
    plt.suptitle(f'{target_name} vs HRV Metrics + CRP', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    return fig

# Loop through each target
for trg in target_list:
    print("\n" + "="*80)
    print(f"ANALYSIS: {trg}")
    print("="*80)
    
    # Merge HRV data with target and CRP
    temp_df = clinical_data_df[['patientid', trg, clinical_covariate]]
    df_merged = pd.merge(df_sl, temp_df, on="patientid")
    
    # Clean data (remove NaN)
    df_clean = df_merged.dropna(subset=[trg, clinical_covariate])
    df_clean = df_clean.dropna()
    
    if len(df_clean) < 10:
        print(f"⚠️ Skipping (only {len(df_clean)} samples)")
        continue
    
    print(f"\nInitial dataset: {len(df_clean)} samples")
    
    # ========================================================================
    # STEP 1: Z-SCORE CAPPING FOR TARGET
    # ========================================================================
    print("\n1️⃣ Z-score capping for TARGET ({})".format(trg))
    print("-" * 80)
    
    target_original = df_clean[trg].values
    target_capped, n_outliers_target = z_score_capping(target_original, z_threshold=3)
    
    print(f"   Original range: [{target_original.min():.2f}, {target_original.max():.2f}]")
    print(f"   Outliers detected: {n_outliers_target} ({n_outliers_target/len(target_original)*100:.1f}%)")
    print(f"   Capped range: [{target_capped.min():.2f}, {target_capped.max():.2f}]")
    
    df_clean[trg] = target_capped
    
    # ========================================================================
    # STEP 2: Z-SCORE CAPPING FOR CRP
    # ========================================================================
    print("\n2️⃣ Z-score capping for CRP")
    print("-" * 80)
    
    crp_original = df_clean[clinical_covariate].values
    crp_capped, n_outliers_crp = z_score_capping(crp_original, z_threshold=3)
    
    print(f"   Original range: [{crp_original.min():.2f}, {crp_original.max():.2f}]")
    print(f"   Outliers detected: {n_outliers_crp} ({n_outliers_crp/len(crp_original)*100:.1f}%)")
    print(f"   Capped range: [{crp_capped.min():.2f}, {crp_capped.max():.2f}]")
    
    df_clean[clinical_covariate] = crp_capped
    
    # ========================================================================
    # STEP 3: MIN-MAX NORMALIZATION [0,1] FOR ALL FEATURES
    # ========================================================================
    print("\n3️⃣ Min-Max normalization [0,1] for all features")
    print("-" * 80)
    
    # Show before normalization
    print("   Before normalization (sample ranges):")
    for feat in all_features[:3]:  # Show first 3 as example
        print(f"      {feat}: [{df_clean[feat].min():.2f}, {df_clean[feat].max():.2f}]")
    print(f"      {clinical_covariate}: [{df_clean[clinical_covariate].min():.2f}, {df_clean[clinical_covariate].max():.2f}]")
    
    # Normalize all features to [0, 1]
    df_clean_norm, feature_scaler = normalize_features_0_1(df_clean, all_features)
    
    # Show after normalization
    print("\n   After normalization [0,1]:")
    for feat in all_features[:3]:
        print(f"      {feat}: [{df_clean_norm[feat].min():.2f}, {df_clean_norm[feat].max():.2f}]")
    print(f"      {clinical_covariate}: [{df_clean_norm[clinical_covariate].min():.2f}, {df_clean_norm[clinical_covariate].max():.2f}]")
    
    # Use normalized dataframe for modeling
    df_clean = df_clean_norm
    
    print(f"\n✓ Preprocessing complete: {len(df_clean)} samples ready")
    print(f"   Target: {trg} (capped)")
    print(f"   Features: {len(all_features)} (normalized to [0,1])")
    
    # ========================================================================
    # FORWARD SELECTION (using preprocessed data)
    # ========================================================================
    
    y = df_clean[trg].values
    
    # Run exhaustive forward selection
    results_by_size, best_overall = exhaustive_forward_selection(
        df_clean, trg, all_features, clinical_covariate, kf, alphas
    )
    
    # Train final model
    X_final = df_clean[best_overall['features']]
    
    # Use StandardScaler for Ridge (even though features are already 0-1)
    # This centers them around 0 which Ridge prefers
    scaler_final = StandardScaler()
    X_final_scaled = scaler_final.fit_transform(X_final)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_final_scaled, y, test_size=0.2, random_state=42
    )
    
    final_model = Ridge(alpha=best_overall['best_alpha'], random_state=42)
    final_model.fit(X_train, y_train)
    y_pred = final_model.predict(X_test)
    
    # Evaluation
    test_r2 = r2_score(y_test, y_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_mae = mean_absolute_error(y_test, y_pred)
    
    # Create scatter plot: target vs all metrics (using original scale for visualization)
    fig_scatter = create_target_metric_scatter(df_clean, trg, all_features, clinical_covariate)
    scatter_img = fig_to_base64(fig_scatter)
    
    # Store results
    all_results[trg] = {
        'results_by_size': results_by_size,
        'best_overall': best_overall,
        'test_r2': test_r2,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'n_samples': len(df_clean),
        'scatter_img': scatter_img,
        'final_model': final_model,
        'scaler': scaler_final,
        'df_clean': df_clean,
        'crp_selected': clinical_covariate in best_overall['features'],
        'n_outliers_target': n_outliers_target,
        'n_outliers_crp': n_outliers_crp
    }

# ===========================================================================
# CREATE HTML REPORT
# ===========================================================================

print("\n" + "="*80)
print("CREATING HTML REPORT")
print("="*80)

def get_r2_color(r2):
    """Color code based on R² value"""
    if r2 >= 0.7:
        return '#90ee90'
    elif r2 >= 0.5:
        return '#add8e6'
    elif r2 >= 0.3:
        return '#fff8dc'
    elif r2 >= 0:
        return '#f0f0f0'
    else:
        return '#ffcccb'

# Build HTML
html = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Ridge Regression with CRP Covariate - Analysis Report</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            border-radius: 8px;
        }
        h1 {
            color: #2c3e50;
            border-bottom: 4px solid #e74c3c;
            padding-bottom: 15px;
            margin-bottom: 30px;
        }
        h2 {
            color: #34495e;
            border-bottom: 2px solid #95a5a6;
            padding-bottom: 10px;
            margin-top: 40px;
            margin-bottom: 20px;
        }
        h3 {
            color: #7f8c8d;
            margin-top: 30px;
            margin-bottom: 15px;
        }
        .summary-box {
            background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
        }
        .summary-box p {
            margin: 8px 0;
            font-size: 14px;
        }
        .highlight-box {
            background: #fff3cd;
            border-left: 5px solid #ffc107;
            padding: 15px;
            margin: 20px 0;
            border-radius: 4px;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        th, td {
            border: 1px solid #ddd;
            padding: 12px;
            text-align: center;
        }
        th {
            background-color: #e74c3c;
            color: white;
            font-weight: bold;
        }
        td:first-child {
            text-align: left;
            font-weight: 600;
        }
        .metric-value {
            font-family: 'Courier New', monospace;
            font-size: 13px;
        }
        .target-section {
            background: #f8f9fa;
            padding: 25px;
            margin: 30px 0;
            border-radius: 8px;
            border-left: 5px solid #e74c3c;
        }
        .plot-container {
            margin: 25px 0;
            text-align: center;
        }
        .plot-container img {
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .insight-box {
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 20px 0;
            border-radius: 4px;
        }
        .good { color: #27ae60; font-weight: bold; }
        .warning { color: #f39c12; font-weight: bold; }
        .bad { color: #e74c3c; font-weight: bold; }
        .crp-badge {
            background: #e74c3c;
            color: white;
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: bold;
            margin-left: 10px;
        }
        .legend {
            margin: 20px 0;
            padding: 15px;
            background: #ecf0f1;
            border-radius: 8px;
        }
        .legend-item {
            display: inline-block;
            margin-right: 20px;
            padding: 8px 15px;
            border-radius: 4px;
            font-size: 13px;
        }
        .footer {
            margin-top: 50px;
            padding-top: 20px;
            border-top: 2px solid #ecf0f1;
            text-align: center;
            color: #95a5a6;
            font-size: 12px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🩺 Ridge Regression with CRP as Covariate</h1>
        
        <div class="summary-box">
            <p><strong>Analysis Type:</strong> Exhaustive Forward Selection with Ridge Regression</p>
            <p><strong>HRV Features:</strong> """ + str(len(hrv_features)) + """ metrics</p>
            <p><strong>Clinical Covariate:</strong> CRP (C-Reactive Protein)</p>
            <p><strong>Total Features:</strong> """ + str(len(all_features)) + """ (HRV + CRP)</p>
            <p><strong>Targets:</strong> """ + str(len(all_results)) + """ clinical outcomes</p>
            <p><strong>Method:</strong> 5-Fold Cross-Validation with Hyperparameter Tuning (α)</p>
            <p><strong>Preprocessing:</strong> Z-score capping (Z=3) + Min-Max normalization [0,1]</p>
            <p><strong>Date:</strong> """ + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
        </div>
        
        <div class="highlight-box">
            <h3>🔧 Preprocessing Pipeline</h3>
            <p><strong>1. Z-score Capping (Z=3):</strong> Outliers in targets and CRP are capped at ±3 standard deviations 
            to reduce the influence of extreme values while preserving data points.</p>
            <p><strong>2. Min-Max Normalization [0,1]:</strong> All features (HRV + CRP) are scaled to [0,1] range 
            for fair comparison and improved model stability.</p>
        </div>
        
        <div class="highlight-box">
            <h3>💡 Key Question: Does CRP improve predictions?</h3>
            <p>This analysis tests whether adding <strong>CRP (C-Reactive Protein)</strong> as a covariate alongside 
            HRV metrics improves the prediction of clinical outcomes. CRP is a marker of systemic inflammation and may 
            provide complementary information to cardiac autonomic function (HRV).</p>
        </div>
        
        <div class="legend">
            <strong>R² Performance Color Coding:</strong><br><br>
            <span class="legend-item" style="background: #90ee90;">R² ≥ 0.7 (Excellent)</span>
            <span class="legend-item" style="background: #add8e6;">0.5 ≤ R² < 0.7 (Good)</span>
            <span class="legend-item" style="background: #fff8dc;">0.3 ≤ R² < 0.5 (Moderate)</span>
            <span class="legend-item" style="background: #f0f0f0;">0 ≤ R² < 0.3 (Weak)</span>
            <span class="legend-item" style="background: #ffcccb;">R² < 0 (Poor)</span>
        </div>
        
        <h2>📋 Summary Table: All Targets</h2>
        <table>
            <thead>
                <tr>
                    <th>Target</th>
                    <th>N Samples</th>
                    <th>CRP Selected?</th>
                    <th>Best N Features</th>
                    <th>CV R²</th>
                    <th>Test R²</th>
                    <th>RMSE</th>
                    <th>MAE</th>
                    <th>|CV-Test|</th>
                </tr>
            </thead>
            <tbody>
"""

# Add summary table rows
for trg, results in all_results.items():
    best = results['best_overall']
    cv_r2 = best['cv_score']
    test_r2 = results['test_r2']
    cv_test_diff = abs(cv_r2 - test_r2)
    
    cv_color = get_r2_color(cv_r2)
    test_color = get_r2_color(test_r2)
    
    consistency_class = 'good' if cv_test_diff < 0.1 else 'warning' if cv_test_diff < 0.2 else 'bad'
    
    crp_status = '✅ YES' if results['crp_selected'] else '❌ NO'
    crp_color = '#d5f4e6' if results['crp_selected'] else '#f8d7da'
    
    html += f"""
                <tr>
                    <td>{trg}</td>
                    <td>{results['n_samples']}</td>
                    <td style="background-color: {crp_color}; font-weight: bold;">{crp_status}</td>
                    <td>{best['size']}</td>
                    <td style="background-color: {cv_color};" class="metric-value">{cv_r2:.4f}</td>
                    <td style="background-color: {test_color};" class="metric-value">{test_r2:.4f}</td>
                    <td class="metric-value">{results['test_rmse']:.4f}</td>
                    <td class="metric-value">{results['test_mae']:.4f}</td>
                    <td class="{consistency_class} metric-value">{cv_test_diff:.4f}</td>
                </tr>
"""

html += """
            </tbody>
        </table>
"""

# Add detailed sections for each target
for trg, results in all_results.items():
    best = results['best_overall']
    
    # Check if CRP was selected
    if results['crp_selected']:
        crp_position = best['features'].index(clinical_covariate) + 1
        crp_info = f"✅ Selected at position {crp_position}"
        crp_class = 'good'
    else:
        crp_info = "❌ Not selected"
        crp_class = 'bad'
    
    # Create CV curve plot
    fig_curve, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: CV R² vs Feature Count
    ax1 = axes[0]
    sizes = sorted(results['results_by_size'].keys())
    cv_scores = [results['results_by_size'][s]['cv_score'] for s in sizes]
    
    ax1.plot(sizes, cv_scores, 'o-', linewidth=2, markersize=8, color='steelblue')
    best_size = best['size']
    ax1.plot(best_size, best['cv_score'], 'r*', markersize=20, label=f'Best: {best_size} features')
    ax1.axhline(y=results['test_r2'], color='orange', linestyle='--', linewidth=2, label=f'Test R²: {results["test_r2"]:.3f}')
    ax1.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_xlabel('Number of Features', fontsize=12)
    ax1.set_ylabel('R²', fontsize=12)
    ax1.set_title('CV R² vs Feature Count', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    ax1.set_xticks(range(1, len(sizes)+1, max(1, len(sizes)//10)))
    
    # Plot 2: Feature Addition Order (with CRP highlighted)
    ax2 = axes[1]
    feature_names = []
    feature_types = []
    
    for s in sizes:
        feat = results['results_by_size'][s]['new_feature']
        if feat == clinical_covariate:
            feature_names.append('🩺 CRP')
            feature_types.append('CRP')
        else:
            feature_names.append(feat.replace('HRV_', ''))
            feature_types.append('HRV')
    
    # Color: red for CRP, blue for HRV
    colors = ['red' if ft == 'CRP' else 'lightblue' for ft in feature_types]
    # Highlight best size
    colors = ['darkred' if (s == best_size and feature_types[s-1] == 'CRP') 
              else 'darkblue' if (s == best_size and feature_types[s-1] == 'HRV')
              else c for s, c in zip(sizes, colors)]
    
    ax2.barh(range(len(sizes)), cv_scores, color=colors, edgecolor='k', alpha=0.7)
    ax2.set_yticks(range(len(sizes)))
    ax2.set_yticklabels([f"{i+1}. {name}" for i, name in enumerate(feature_names)], fontsize=9)
    ax2.set_xlabel('CV R²', fontsize=12)
    ax2.set_title('Feature Selection Order', fontsize=13, fontweight='bold')
    ax2.axvline(x=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax2.grid(alpha=0.3, axis='x')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightblue', edgecolor='k', label='HRV Feature'),
        Patch(facecolor='red', edgecolor='k', label='CRP (Clinical)')
    ]
    ax2.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    curve_img = fig_to_base64(fig_curve)
    
    # Create coefficient plot
    fig_coef, ax = plt.subplots(figsize=(10, 6))
    
    feature_labels = []
    feature_colors = []
    
    for f in best['features']:
        if f == clinical_covariate:
            feature_labels.append('🩺 CRP')
            feature_colors.append('clinical')
        else:
            feature_labels.append(f.replace('HRV_', ''))
            feature_colors.append('hrv')
    
    coef_df = pd.DataFrame({
        'Feature': feature_labels,
        'Coefficient': results['final_model'].coef_,
        'Type': feature_colors
    }).sort_values('Coefficient')
    
    # Color by sign and type
    colors_coef = []
    for _, row in coef_df.iterrows():
        if row['Type'] == 'clinical':
            colors_coef.append('darkred' if row['Coefficient'] < 0 else 'darkgreen')
        else:
            colors_coef.append('red' if row['Coefficient'] < 0 else 'green')
    
    ax.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors_coef, alpha=0.7, edgecolor='k')
    ax.axvline(x=0, color='k', linestyle='-', linewidth=1)
    ax.set_xlabel('Coefficient Value (Standardized)', fontsize=12)
    ax.set_title(f'{trg}: Feature Coefficients', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3, axis='x')
    
    # Add legend
    legend_elements = [
        Patch(facecolor='green', edgecolor='k', label='HRV Positive (↑)'),
        Patch(facecolor='red', edgecolor='k', label='HRV Negative (↓)'),
        Patch(facecolor='darkgreen', edgecolor='k', label='CRP Positive (↑)'),
        Patch(facecolor='darkred', edgecolor='k', label='CRP Negative (↓)')
    ]
    ax.legend(handles=legend_elements, loc='best', fontsize=9)
    
    plt.tight_layout()
    coef_img = fig_to_base64(fig_coef)
    
    # Assessment
    test_r2 = results['test_r2']
    if test_r2 < 0:
        assessment = '❌ Poor (worse than baseline)'
        assessment_class = 'bad'
    elif test_r2 < 0.1:
        assessment = '⚠️ Very Weak'
        assessment_class = 'bad'
    elif test_r2 < 0.3:
        assessment = '⚠️ Weak'
        assessment_class = 'warning'
    elif test_r2 < 0.5:
        assessment = '✓ Moderate'
        assessment_class = 'warning'
    else:
        assessment = '✓✓ Good'
        assessment_class = 'good'
    
    cv_test_diff = abs(best['cv_score'] - test_r2)
    if cv_test_diff < 0.1:
        consistency = '✅ Stable'
    elif cv_test_diff < 0.2:
        consistency = '⚠️ Moderate variability'
    else:
        consistency = '❌ High variability (overfitting concern)'
    
    html += f"""
        <div class="target-section">
            <h2>🎯 {trg} <span class="crp-badge">CRP: {crp_info}</span></h2>
            
            <h3>📊 Dataset Information</h3>
            <p><strong>Samples:</strong> {results['n_samples']}</p>
            <p><strong>Target Range:</strong> [{results['df_clean'][trg].min():.2f}, {results['df_clean'][trg].max():.2f}] (after capping)</p>
            <p><strong>Target Mean ± SD:</strong> {results['df_clean'][trg].mean():.2f} ± {results['df_clean'][trg].std():.2f}</p>
            <p><strong>Target Outliers Capped:</strong> {results['n_outliers_target']} ({results['n_outliers_target']/results['n_samples']*100:.1f}%)</p>
            <p><strong>CRP Range:</strong> [{results['df_clean'][clinical_covariate].min():.2f}, {results['df_clean'][clinical_covariate].max():.2f}] (after capping, normalized)</p>
            <p><strong>CRP Outliers Capped:</strong> {results['n_outliers_crp']} ({results['n_outliers_crp']/results['n_samples']*100:.1f}%)</p>
            <p><strong>Features Normalized:</strong> All {len(all_features)} features scaled to [0, 1]</p>
            
            <h3>🏆 Best Model</h3>
            <p><strong>Number of Features:</strong> {best['size']} / {len(all_features)}</p>
            <p><strong>Selected Features:</strong> {', '.join([('🩺 CRP' if f == clinical_covariate else f.replace('HRV_', '')) for f in best['features']])}</p>
            <p><strong>Best Alpha:</strong> {best['best_alpha']:.4f}</p>
            <p><strong>CRP Status:</strong> <span class="{crp_class}">{crp_info}</span></p>
            
            <h3>📈 Performance Metrics</h3>
            <table style="width: 60%;">
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>CV R²</td><td class="metric-value">{best['cv_score']:.4f}</td></tr>
                <tr><td>Test R²</td><td class="metric-value">{test_r2:.4f}</td></tr>
                <tr><td>Test RMSE</td><td class="metric-value">{results['test_rmse']:.4f}</td></tr>
                <tr><td>Test MAE</td><td class="metric-value">{results['test_mae']:.4f}</td></tr>
            </table>
            
            <div class="insight-box">
                <p><strong>💡 Assessment:</strong> <span class="{assessment_class}">{assessment}</span></p>
                <p><strong>🔄 Consistency:</strong> {consistency}</p>
"""

    # Add interpretation for CRP
    if results['crp_selected']:
        crp_pos = best['features'].index(clinical_covariate) + 1
        if crp_pos == 1:
            interpretation = "🔥 <strong>Inflammation (CRP) is the PRIMARY predictor!</strong> Systemic inflammation is more important than cardiac autonomic function for this outcome."
        elif crp_pos <= 3:
            interpretation = "⚠️ <strong>Inflammation matters.</strong> CRP selected early suggests systemic inflammation plays a significant role alongside autonomic dysfunction."
        else:
            interpretation = "ℹ️ <strong>Marginal contribution.</strong> CRP provides some improvement but HRV features are more important."
    else:
        interpretation = "✓ <strong>HRV alone is sufficient.</strong> Systemic inflammation (CRP) does not improve predictions - autonomic function is the key."
    
    html += f"""
                <p><strong>🔍 Clinical Interpretation:</strong> {interpretation}</p>
            </div>
            
            <h3>📊 Feature Selection Curve</h3>
            <div class="plot-container">
                <img src="data:image/png;base64,{curve_img}" alt="CV Curve">
            </div>
            
            <h3>🔧 Feature Coefficients</h3>
            <div class="plot-container">
                <img src="data:image/png;base64,{coef_img}" alt="Coefficients">
            </div>
            
            <h3>🔍 Target vs HRV Metrics + CRP (Scatter Plots)</h3>
            <div class="plot-container">
                <img src="data:image/png;base64,{results['scatter_img']}" alt="Scatter plots">
            </div>
        </div>
"""

# CRP Selection Summary
crp_selected_count = sum([1 for r in all_results.values() if r['crp_selected']])

html += f"""
        <h2>📊 CRP Selection Summary</h2>
        <div class="highlight-box">
            <p><strong>CRP was selected in {crp_selected_count} out of {len(all_results)} targets ({crp_selected_count/len(all_results)*100:.0f}%)</strong></p>
"""

if crp_selected_count > 0:
    html += "<ul>"
    for trg, results in all_results.items():
        if results['crp_selected']:
            pos = results['best_overall']['features'].index(clinical_covariate) + 1
            html += f"<li><strong>{trg}:</strong> CRP selected at position {pos}/{len(all_features)}</li>"
    html += "</ul>"
    html += "<p><strong>Interpretation:</strong> Systemic inflammation (CRP) provides valuable predictive information "
    html += "for these outcomes, suggesting inflammatory processes contribute to disease burden.</p>"
else:
    html += "<p><strong>Interpretation:</strong> CRP was not selected in any model. This suggests that cardiac autonomic "
    html += "dysfunction (measured by HRV) is a more relevant predictor than systemic inflammation for these outcomes.</p>"

html += """
        </div>
        
        <div class="footer">
            <h3>📖 Methodology</h3>
            
            <p><strong>Preprocessing Steps:</strong></p>
            <p><strong>1. Z-score Capping (Z=3):</strong> Outliers in target variables and CRP are identified using Z-scores. 
            Values beyond ±3 standard deviations from the mean are capped (not removed) to these bounds. This preserves 
            sample size while reducing the influence of extreme values that could skew the model.</p>
            
            <p><strong>2. Min-Max Normalization [0,1]:</strong> All features (13 HRV metrics + CRP) are scaled to the 
            [0, 1] range using MinMaxScaler. This ensures fair comparison between features with different units and scales, 
            and improves numerical stability during optimization.</p>
            
            <p><strong>CRP as Covariate:</strong> C-Reactive Protein (CRP) is included as an additional feature 
            alongside 13 HRV metrics. The forward selection process treats all 14 features equally, allowing us to 
            determine whether inflammatory markers (CRP) or autonomic measures (HRV) are more predictive.</p>
            
            <p><strong>Exhaustive Forward Selection:</strong> Tests all possible feature combinations at each size level.
            At each step, the algorithm keeps the best features from the previous step and tries adding each remaining 
            feature, selecting the combination with the highest cross-validation R².</p>
            
            <p><strong>Clinical Relevance:</strong> If CRP is selected early (positions 1-3), it suggests that systemic 
            inflammation is a primary driver of the outcome. If CRP is not selected, autonomic dysfunction (HRV) is 
            the key predictor. This distinction helps guide treatment strategies.</p>
            
            <p><strong>Ridge Regression:</strong> Linear regression with L2 regularization prevents overfitting, 
            especially important when working with 14 potentially correlated features. After normalization, features 
            are further standardized (mean=0, std=1) before Ridge fitting.</p>
            
            <hr style="margin: 30px 0;">
            
            <p><em>Report generated automatically | """ + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S') + """</em></p>
            <p><em>HRV: Heart Rate Variability (cardiac autonomic function)</em></p>
            <p><em>CRP: C-Reactive Protein (systemic inflammation marker)</em></p>
            <p><em>Preprocessing: Z-score capping (Z=3) + Min-Max scaling [0,1]</em></p>
        </div>
    </div>
</body>
</html>
"""

# Save HTML
html_filename = 'ridge_regression_with_crp_comprehensive_report.html'
with open(html_filename, 'w', encoding='utf-8') as f:
    f.write(html)

print(f"\n✅ HTML report created: {html_filename}")
print(f"   Total size: {len(html) / 1024:.1f} KB")
print(f"   Targets analyzed: {len(all_results)}")
print(f"   Total plots embedded: {len(all_results) * 3}")

print("\n" + "="*80)
print("KEY FINDINGS")
print("="*80)
print(f"\nCRP Selection Rate: {crp_selected_count}/{len(all_results)} targets ({crp_selected_count/len(all_results)*100:.0f}%)")

for trg, results in all_results.items():
    status = "✅ CRP" if results['crp_selected'] else "❌ HRV only"
    if results['crp_selected']:
        pos = results['best_overall']['features'].index(clinical_covariate) + 1
        print(f"  {status} | {trg}: Test R² = {results['test_r2']:.4f} (CRP at position {pos})")
    else:
        print(f"  {status} | {trg}: Test R² = {results['test_r2']:.4f}")

print("\n" + "="*80)
print("REPORT COMPLETE!")
print("="*80)
print(f"\n📄 Open '{html_filename}' in your browser to view the full report")