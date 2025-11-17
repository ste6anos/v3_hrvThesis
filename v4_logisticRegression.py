import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve, precision_recall_curve,
                             average_precision_score)
from sklearn.feature_selection import SequentialFeatureSelector
from imblearn.over_sampling import SMOTE
from statsmodels.stats.outliers_influence import variance_inflation_factor
import base64
from io import BytesIO
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300


# ===========================================================================
# CONFIGURATION
# ===========================================================================

# Data paths
DATA_PATHS = {
    'clinical_data': r"C:\Users\spbtu\Documents\dataset_psath\bbi_metadata\clinical_d32_T0_v2.csv",
    'low_hr_data': r"C:\Users\spbtu\Documents\hrvThesis\window90m_hrv_mean_measurements_sl.csv",
    'high_hr_data': r"C:\Users\spbtu\Documents\hrvThesis\window90m_hrv_mean_measurements_aw.csv",
}

# Target variables
TARGETS = ['DOC_FLARE', 'PAT_FLARE']

# Covariates from clinical data
COVARIATES = ['DEMOGR_AGE', 'DEMOGR_SEX', 'PAT_SMOKE_PAST_H', 'CRP_mg_dL', 'DOC_PAT_BMI']

# Preprocessing flags
APPLY_FORWARD_SELECTION = True
APPLY_CAPPING = False
APPLY_SMOTE = False
APPLY_SCALING = True
APPLY_THRESHOLDOPT = True

# Other parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
Z_THRESHOLD = 3
MAX_CRP = 100
SMOTE_K_NEIGHBORS = 3

# Logistic Regression hyperparameter grid
PARAM_GRID = {
    'C': np.arange(0.1, 10.1, 0.5),
    'l1_ratio': np.arange(0.1, 1.1, 0.2),
    'class_weight': ['balanced', None]
}


# ===========================================================================
# HELPER FUNCTIONS
# ===========================================================================

def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string."""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64


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


def check_class_imbalance(y, target_name):
    """Check and report class imbalance."""
    class_counts = y.value_counts()
    class_ratio = class_counts.min() / class_counts.max()

    print(f"\nClass distribution for {target_name}:")
    print(f"  - Class 0 (No Flare): {class_counts.get(0, 0)} ({class_counts.get(0, 0)/len(y)*100:.1f}%)")
    print(f"  - Class 1 (Flare):    {class_counts.get(1, 0)} ({class_counts.get(1, 0)/len(y)*100:.1f}%)")
    print(f"  - Imbalance ratio: {class_ratio:.3f}")

    if class_ratio < 0.2:
        print(f"  ⚠️  SEVERE CLASS IMBALANCE DETECTED!")

    return class_ratio


def optimize_threshold(y_true, y_pred_proba):
    """Find optimal classification threshold using F1-score."""
    thresholds = np.arange(0.1, 0.9, 0.05)
    f1_scores = []

    for thresh in thresholds:
        y_pred = (y_pred_proba >= thresh).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        f1_scores.append(f1)

    best_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]

    return optimal_threshold, best_f1, thresholds, f1_scores


# ===========================================================================
# DATA LOADING AND PREPROCESSING
# ===========================================================================

def load_and_prepare_data(hrv_metrics):
    """Load clinical and HRV data."""
    print("="*70)
    print("DATA LOADING")
    print("="*70)

    # Load clinical data
    clinical_df = pd.read_csv(DATA_PATHS['clinical_data'])
    print(f"✓ Clinical data loaded: {clinical_df.shape}")

    # Load HRV data
    df_low_hr = pd.read_csv(DATA_PATHS['low_hr_data'], index_col=0)
    df_high_hr = pd.read_csv(DATA_PATHS['high_hr_data'], index_col=0)

    print(f"✓ Low HR Activity data loaded: {df_low_hr.shape}")
    print(f"✓ High HR Activity data loaded: {df_high_hr.shape}")
    print(f"✓ HRV features to use: {len(hrv_metrics)}")

    return clinical_df, df_low_hr, df_high_hr


def prepare_target_data(clinical_df, hrv_df, target_name, hrv_features, state_name):
    """Prepare data for a specific target and HR activity state."""
    print(f"\n{'='*70}")
    print(f"PREPARING DATA: {target_name} - {state_name}")
    print(f"{'='*70}")

    # Select relevant clinical columns
    clinical_subset = clinical_df[['patientid', target_name] + COVARIATES].copy()

    # Select HRV features
    hrv_subset = hrv_df[['patientid'] + hrv_features].copy()

    # Merge HRV with clinical data
    df_merged = pd.merge(hrv_subset, clinical_subset, on='patientid', how='inner')
    print(f"✓ Data merged: {df_merged.shape}")

    # Clean CRP
    if 'CRP_mg_dL' in df_merged.columns:
        df_merged = df_merged.dropna(subset=['CRP_mg_dL'])
        df_merged = df_merged[df_merged['CRP_mg_dL'] <= MAX_CRP]
        print(f"✓ CRP filtered: {df_merged.shape}")

    # Drop rows with missing target
    df_merged = df_merged.dropna(subset=[target_name])
    print(f"✓ Target filtered: {df_merged.shape}")

    if len(df_merged) == 0:
        print(f"✗ ERROR: No data available for {target_name} - {state_name}")
        return None

    # Convert target to binary
    df_merged[f'{target_name}_binary'] = (df_merged[target_name].str.lower() == 'yes').astype(int)
    y = df_merged[f'{target_name}_binary']

    # Check class imbalance
    class_ratio = check_class_imbalance(y, target_name)

    # Prepare features (HRV + covariates)
    feature_cols = hrv_features + COVARIATES
    X = df_merged[feature_cols].copy()

    # Handle categorical variables - convert all object/string columns to numeric
    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype.name == 'category':
            print(f"  Converting categorical column '{col}' to numeric...")
            # Handle Yes/No values
            if X[col].str.lower().isin(['yes', 'no']).any():
                X[col] = (X[col].str.lower() == 'yes').astype(int)
            else:
                # For other categorical values (e.g., Male/Female)
                X[col] = pd.Categorical(X[col]).codes
            print(f"    Unique values in {col}: {X[col].nunique()}")

    # Drop rows with missing features
    X = X.dropna()
    y = y.loc[X.index]

    print(f"✓ Final dataset: {X.shape[0]} samples, {X.shape[1]} features")

    if len(X) < 20:
        print(f"✗ ERROR: Insufficient samples for {target_name} - {state_name}")
        return None

    if y.value_counts().min() < 5:
        print(f"⚠️  WARNING: Minority class has only {y.value_counts().min()} samples")

    return {
        'X': X,
        'y': y,
        'feature_names': list(X.columns),
        'class_ratio': class_ratio,
        'df_merged': df_merged
    }


# ===========================================================================
# FEATURE SELECTION AND MODEL TRAINING
# ===========================================================================

def perform_forward_selection(X, y, config):
    """Perform forward feature selection with F1-score optimization."""
    if not APPLY_FORWARD_SELECTION:
        print("\nForward selection DISABLED - using all features")
        return list(X.columns), {}

    print(f"\n{'='*70}")
    print("FORWARD FEATURE SELECTION (F1-Score Optimization)")
    print(f"{'='*70}")

    selected_features = []
    remaining_features = list(X.columns)
    best_score = 0
    selection_history = []

    kf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # Check if we can use SMOTE
    minority_class_count = y.value_counts().min()
    use_smote = APPLY_SMOTE and minority_class_count >= SMOTE_K_NEIGHBORS + 1

    if use_smote:
        print("✓ SMOTE will be applied during CV")
    else:
        print("⚠️  SMOTE disabled (minority class too small)")

    print("\nFeature selection progress:")
    print("-" * 70)

    while remaining_features:
        scores = []

        for feat in remaining_features:
            current_features = selected_features + [feat]
            X_current = X[current_features]

            # Standardize
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_current)

            # Cross-validation with F1-score
            cv_f1_scores = []

            for train_idx, val_idx in kf.split(X_scaled, y):
                X_train_cv, X_val_cv = X_scaled[train_idx], X_scaled[val_idx]
                y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]

                # Apply SMOTE to training data only
                if use_smote and y_train_cv.value_counts().min() >= SMOTE_K_NEIGHBORS:
                    smote = SMOTE(k_neighbors=SMOTE_K_NEIGHBORS, random_state=RANDOM_STATE)
                    try:
                        X_train_cv, y_train_cv = smote.fit_resample(X_train_cv, y_train_cv)
                    except:
                        pass

                # Train model
                model = LogisticRegression(
                    penalty='elasticnet',
                    solver='saga',
                    max_iter=10000,
                    C=1.0,
                    l1_ratio=0.5,
                    class_weight='balanced',
                    random_state=RANDOM_STATE
                )

                try:
                    model.fit(X_train_cv, y_train_cv)
                    y_pred_cv = model.predict(X_val_cv)
                    f1_cv = f1_score(y_val_cv, y_pred_cv, zero_division=0)
                    cv_f1_scores.append(f1_cv)
                except:
                    cv_f1_scores.append(0)

            mean_f1 = np.mean(cv_f1_scores)
            scores.append({'feature': feat, 'cv_score': mean_f1})

        # Pick best feature
        best_candidate = max(scores, key=lambda x: x['cv_score'])

        # Check if improvement
        if best_candidate['cv_score'] > best_score or len(selected_features) == 0:
            selected_features.append(best_candidate['feature'])
            remaining_features.remove(best_candidate['feature'])
            best_score = best_candidate['cv_score']
            selection_history.append(best_candidate)

            print(f"✓ Feature {len(selected_features):2d}: {best_candidate['feature']:20s} | CV F1: {best_score:.3f}")
        else:
            print(f"✗ No improvement. Stopping selection.")
            break

    print(f"\n✓ Selected {len(selected_features)} features")
    return selected_features, selection_history


def train_final_model(X, y, selected_features, state_name, target_name):
    """Train final logistic regression model."""
    print(f"\n{'='*70}")
    print(f"TRAINING FINAL MODEL: {target_name} - {state_name}")
    print(f"{'='*70}")

    X_final = X[selected_features]

    # Apply capping
    if APPLY_CAPPING:
        print("\nApplying outlier capping (Z-score method):")
        X_final = cap_outliers(X_final, z_threshold=Z_THRESHOLD)

    # Apply scaling
    scaler = None
    if APPLY_SCALING:
        print("\n✓ Applying StandardScaler")
        scaler = StandardScaler()
        X_final_scaled = scaler.fit_transform(X_final)
    else:
        X_final_scaled = X_final.values

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_final_scaled, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # Apply SMOTE to training data
    smote_applied = False
    X_train_resampled = X_train.copy()
    y_train_resampled = y_train.copy()

    if APPLY_SMOTE and y_train.value_counts().min() >= SMOTE_K_NEIGHBORS:
        smote = SMOTE(k_neighbors=SMOTE_K_NEIGHBORS, random_state=RANDOM_STATE)
        try:
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            print(f"\n✓ SMOTE applied: {len(y_train)} → {len(y_train_resampled)} samples")
            print(f"  Original: {y_train.value_counts().to_dict()}")
            print(f"  After SMOTE: {pd.Series(y_train_resampled).value_counts().to_dict()}")
            smote_applied = True
        except Exception as e:
            print(f"⚠️  SMOTE failed: {e}")

    # Hyperparameter optimization with GridSearchCV
    print("\n✓ Optimizing hyperparameters with GridSearchCV...")

    model = LogisticRegression(
        penalty='elasticnet',
        solver='saga',
        max_iter=10000,
        random_state=RANDOM_STATE
    )

    kf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    grid_search = GridSearchCV(
        model, PARAM_GRID, cv=kf,
        scoring='f1', n_jobs=-1, verbose=0
    )
    grid_search.fit(X_train_resampled, y_train_resampled)

    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    print(f"  Best params: C={best_params['C']:.1f}, l1_ratio={best_params['l1_ratio']:.2f}, "
          f"class_weight={best_params['class_weight']}")

    # Predictions
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)
    y_pred_proba_test = best_model.predict_proba(X_test)[:, 1]

    # Optimize threshold
    optimal_threshold = 0.5
    threshold_f1 = f1_score(y_test, y_pred_test, zero_division=0)
    thresholds_list = [0.5]
    f1_scores_list = [threshold_f1]

    if APPLY_THRESHOLDOPT and len(np.unique(y_test)) > 1:
        optimal_threshold, threshold_f1, thresholds_list, f1_scores_list = optimize_threshold(
            y_test, y_pred_proba_test
        )
        y_pred_test_optimized = (y_pred_proba_test >= optimal_threshold).astype(int)

        print(f"\n✓ Optimal threshold: {optimal_threshold:.2f} (F1: {threshold_f1:.3f})")
        print(f"  Default threshold (0.5) F1: {f1_score(y_test, y_pred_test, zero_division=0):.3f}")
    else:
        y_pred_test_optimized = y_pred_test

    # Metrics
    metrics = {
        'train_accuracy': accuracy_score(y_train, y_pred_train),
        'test_accuracy': accuracy_score(y_test, y_pred_test_optimized),
        'precision': precision_score(y_test, y_pred_test_optimized, zero_division=0),
        'recall': recall_score(y_test, y_pred_test_optimized, zero_division=0),
        'f1': f1_score(y_test, y_pred_test_optimized, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_proba_test) if len(np.unique(y_test)) > 1 else 0,
        'avg_precision': average_precision_score(y_test, y_pred_proba_test) if len(np.unique(y_test)) > 1 else 0,
        'confusion_matrix': confusion_matrix(y_test, y_pred_test_optimized),
        'classification_report': classification_report(y_test, y_pred_test_optimized,
                                                      output_dict=True, zero_division=0),
        'optimal_threshold': optimal_threshold,
        'threshold_f1': threshold_f1
    }

    # Print final performance
    print(f"\n{'='*70}")
    print("FINAL MODEL PERFORMANCE")
    print(f"{'='*70}")
    print(f"Train Accuracy: {metrics['train_accuracy']:.3f}")
    print(f"Test Accuracy:  {metrics['test_accuracy']:.3f}")
    print(f"Precision:      {metrics['precision']:.3f}")
    print(f"Recall:         {metrics['recall']:.3f}")
    print(f"F1-Score:       {metrics['f1']:.3f}")
    print(f"ROC-AUC:        {metrics['roc_auc']:.3f}")
    print(f"Avg Precision:  {metrics['avg_precision']:.3f}")

    return {
        'model': best_model,
        'scaler': scaler,
        'best_params': best_params,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred_test': y_pred_test_optimized,
        'y_pred_proba_test': y_pred_proba_test,
        'metrics': metrics,
        'coefficients': dict(zip(selected_features, best_model.coef_[0])),
        'thresholds_list': thresholds_list,
        'f1_scores_list': f1_scores_list,
        'smote_applied': smote_applied,
        'selected_features': selected_features
    }


# ===========================================================================
# PLOTTING FUNCTIONS
# ===========================================================================

def create_feature_importance_plot(coefficients, target_name, state_name):
    """Create feature importance plot."""
    coef_df = pd.DataFrame({
        'Feature': list(coefficients.keys()),
        'Coefficient': list(coefficients.values())
    })
    coef_df = coef_df.sort_values('Coefficient', key=abs, ascending=False)

    fig, ax = plt.subplots(figsize=(10, max(6, len(coef_df) * 0.4)))

    colors = ['green' if x > 0 else 'red' for x in coef_df['Coefficient']]
    ax.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors, alpha=0.7)

    ax.set_xlabel('Coefficient Value', fontsize=12)
    ax.set_title(f'Feature Importance - {target_name} ({state_name})', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    return fig


def create_confusion_matrix_plot(cm, target_name, state_name):
    """Create confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No Flare', 'Flare'],
                yticklabels=['No Flare', 'Flare'])

    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_title(f'Confusion Matrix - {target_name} ({state_name})', fontsize=14, fontweight='bold')

    plt.tight_layout()
    return fig


def create_roc_curve_plot(y_test, y_pred_proba, roc_auc, target_name, state_name):
    """Create ROC curve plot."""
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot(fpr, tpr, color='darkorange', lw=2,
            label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'ROC Curve - {target_name} ({state_name})', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def create_precision_recall_curve_plot(y_test, y_pred_proba, avg_precision, target_name, state_name):
    """Create Precision-Recall curve plot."""
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot(recall, precision, color='blue', lw=2,
            label=f'PR curve (AP = {avg_precision:.3f})')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(f'Precision-Recall Curve - {target_name} ({state_name})', fontsize=14, fontweight='bold')
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def create_threshold_optimization_plot(thresholds_list, f1_scores_list, optimal_threshold, target_name, state_name):
    """Create threshold optimization plot."""
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(thresholds_list, f1_scores_list, 'b-', linewidth=2, label='F1-Score')
    ax.axvline(x=optimal_threshold, color='red', linestyle='--', linewidth=2,
               label=f'Optimal Threshold = {optimal_threshold:.2f}')
    ax.axvline(x=0.5, color='gray', linestyle=':', linewidth=2,
               label='Default Threshold = 0.5')

    ax.set_xlabel('Classification Threshold', fontsize=12)
    ax.set_ylabel('F1-Score', fontsize=12)
    ax.set_title(f'Threshold Optimization - {target_name} ({state_name})', fontsize=14, fontweight='bold')
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ===========================================================================
# HTML REPORT GENERATION
# ===========================================================================

def create_html_report(results, output_file='logistic_regression_dual_state_90m.html'):
    """Generate comprehensive HTML report with dual-state logistic regression results."""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Logistic Regression - Dual State Flare Prediction</title>
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
        }}

        .container {{
            max-width: 1600px;
            margin: 0 auto;
            background: white;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}

        .header {{
            background: white;
            border-bottom: 2px solid #e1e4e8;
            padding: 40px;
            text-align: center;
        }}

        .header h1 {{
            font-size: 2em;
            margin-bottom: 8px;
            font-weight: 600;
            color: #24292e;
        }}

        .header p {{
            font-size: 1em;
            color: #586069;
        }}

        .timestamp {{
            font-size: 0.85em;
            color: #6a737d;
            margin-top: 8px;
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

        .config-item {{
            padding: 8px 12px;
            background: white;
            border: 1px solid #e1e4e8;
            border-radius: 3px;
            font-size: 0.9em;
        }}

        .config-item strong {{
            color: #24292e;
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

        .badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 3px;
            font-size: 0.75em;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-left: 8px;
        }}

        .badge-excellent {{ background: #d4edda; color: #155724; }}
        .badge-good {{ background: #d1ecf1; color: #0c5460; }}
        .badge-moderate {{ background: #fff3cd; color: #856404; }}
        .badge-poor {{ background: #f8d7da; color: #721c24; }}

        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 12px;
            margin: 20px 0;
        }}

        .metric-box {{
            padding: 12px;
            background: #f6f8fa;
            border: 1px solid #e1e4e8;
            border-radius: 3px;
        }}

        .metric-box h4 {{
            font-size: 0.75em;
            color: #586069;
            margin-bottom: 4px;
            text-transform: uppercase;
            letter-spacing: 0.3px;
        }}

        .metric-box p {{
            font-size: 1.5em;
            font-weight: 600;
            color: #24292e;
        }}

        .plot-container {{
            margin: 20px 0;
            text-align: center;
        }}

        .plot-container img {{
            max-width: 100%;
            border: 1px solid #e1e4e8;
            border-radius: 4px;
        }}

        .features-list {{
            margin: 16px 0;
            padding: 12px;
            background: #f6f8fa;
            border-left: 3px solid #28a745;
            font-size: 0.9em;
        }}

        .features-list strong {{
            display: block;
            margin-bottom: 8px;
            color: #24292e;
        }}

        .features-list ul {{
            margin-left: 20px;
            color: #586069;
        }}

        .info-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 12px;
            margin-bottom: 20px;
        }}

        .info-item {{
            padding: 10px;
            background: #f6f8fa;
            border-left: 3px solid #0366d6;
            font-size: 0.9em;
        }}

        .info-item strong {{
            color: #24292e;
            display: block;
            margin-bottom: 4px;
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
            <h1>Logistic Regression - Dual State Flare Prediction</h1>
            <p>HRV Metrics + Clinical Covariates: Low vs High HR Activity</p>
            <p class="timestamp">Generated: {timestamp}</p>
        </div>

        <div class="content">
            <!-- CONFIGURATION -->
            <div class="config-section">
                <h3>Configuration</h3>
                <div class="config-grid">
                    <div class="config-item">
                        <strong>Forward Selection:</strong> {'Enabled' if APPLY_FORWARD_SELECTION else 'Disabled'}
                    </div>
                    <div class="config-item">
                        <strong>Outlier Capping:</strong> {'Enabled (Z=3)' if APPLY_CAPPING else 'Disabled'}
                    </div>
                    <div class="config-item">
                        <strong>SMOTE:</strong> {'Enabled' if APPLY_SMOTE else 'Disabled'}
                    </div>
                    <div class="config-item">
                        <strong>Scaling:</strong> {'StandardScaler' if APPLY_SCALING else 'Disabled'}
                    </div>
                    <div class="config-item">
                        <strong>Threshold Opt:</strong> {'Enabled' if APPLY_THRESHOLDOPT else 'Disabled'}
                    </div>
                    <div class="config-item">
                        <strong>CV Folds:</strong> {CV_FOLDS}
                    </div>
                </div>
                <div style="margin-top: 16px; padding: 12px; background: white; border: 1px solid #e1e4e8; border-radius: 3px;">
                    <strong style="display: block; margin-bottom: 8px; color: #24292e;">Covariates Included:</strong>
                    <p style="color: #586069; font-size: 0.9em;">{', '.join(COVARIATES)}</p>
                </div>
            </div>
"""

    # Add sections for each target
    for target in TARGETS:
        if target not in results:
            continue

        target_results = results[target]

        html += f"""
            <div class="target-section">
                <h2>{target}</h2>

                <div class="state-comparison">
                    <!-- LOW HR ACTIVITY -->
                    <div class="state-panel">
                        <h3>Low HR Activity"""

        # Add badge for Low HR
        if 'low_hr' in target_results and target_results['low_hr'] is not None:
            f1 = target_results['low_hr']['metrics']['f1']
            if f1 >= 0.7:
                html += '<span class="badge badge-excellent">Excellent</span>'
            elif f1 >= 0.5:
                html += '<span class="badge badge-good">Good</span>'
            elif f1 >= 0.3:
                html += '<span class="badge badge-moderate">Moderate</span>'
            else:
                html += '<span class="badge badge-poor">Poor</span>'

        html += """
                        </h3>
"""

        if 'low_hr' in target_results and target_results['low_hr'] is not None:
            low_hr = target_results['low_hr']

            html += f"""
                        <div class="info-grid">
                            <div class="info-item">
                                <strong>Samples</strong>
                                {len(low_hr['y_train']) + len(low_hr['y_test'])}
                            </div>
                            <div class="info-item">
                                <strong>Features Used</strong>
                                {len(low_hr['selected_features'])}
                            </div>
                            <div class="info-item">
                                <strong>SMOTE</strong>
                                {'Applied' if low_hr['smote_applied'] else 'Not Applied'}
                            </div>
                            <div class="info-item">
                                <strong>Optimal Threshold</strong>
                                {low_hr['metrics']['optimal_threshold']:.2f}
                            </div>
                        </div>

                        <div class="features-list">
                            <strong>Selected Features:</strong>
                            <ul>
                                {''.join([f'<li>{f} (β={low_hr["coefficients"][f]:.4f})</li>' for f in low_hr['selected_features']])}
                            </ul>
                        </div>

                        <div class="metrics-grid">
                            <div class="metric-box">
                                <h4>Train Accuracy</h4>
                                <p>{low_hr['metrics']['train_accuracy']:.3f}</p>
                            </div>
                            <div class="metric-box">
                                <h4>Test Accuracy</h4>
                                <p>{low_hr['metrics']['test_accuracy']:.3f}</p>
                            </div>
                            <div class="metric-box">
                                <h4>Precision</h4>
                                <p>{low_hr['metrics']['precision']:.3f}</p>
                            </div>
                            <div class="metric-box">
                                <h4>Recall</h4>
                                <p>{low_hr['metrics']['recall']:.3f}</p>
                            </div>
                            <div class="metric-box">
                                <h4>F1-Score</h4>
                                <p>{low_hr['metrics']['f1']:.3f}</p>
                            </div>
                            <div class="metric-box">
                                <h4>ROC-AUC</h4>
                                <p>{low_hr['metrics']['roc_auc']:.3f}</p>
                            </div>
                        </div>

                        <div class="plot-container">
                            <h4>Feature Importance</h4>
                            <img src="data:image/png;base64,{low_hr['plots']['feature_importance']}" alt="Feature Importance">
                        </div>

                        <div class="plot-container">
                            <h4>Confusion Matrix</h4>
                            <img src="data:image/png;base64,{low_hr['plots']['confusion_matrix']}" alt="Confusion Matrix">
                        </div>

                        <div class="plot-container">
                            <h4>ROC Curve</h4>
                            <img src="data:image/png;base64,{low_hr['plots']['roc_curve']}" alt="ROC Curve">
                        </div>

                        <div class="plot-container">
                            <h4>Precision-Recall Curve</h4>
                            <img src="data:image/png;base64,{low_hr['plots']['pr_curve']}" alt="Precision-Recall Curve">
                        </div>

                        <div class="plot-container">
                            <h4>Threshold Optimization</h4>
                            <img src="data:image/png;base64,{low_hr['plots']['threshold_opt']}" alt="Threshold Optimization">
                        </div>
"""
        else:
            html += """
                        <p style="color: #721c24; padding: 20px; background: #f8d7da; border-radius: 4px;">
                            No data available for Low HR Activity
                        </p>
"""

        html += """
                    </div>

                    <!-- HIGH HR ACTIVITY -->
                    <div class="state-panel">
                        <h3>High HR Activity"""

        # Add badge for High HR
        if 'high_hr' in target_results and target_results['high_hr'] is not None:
            f1 = target_results['high_hr']['metrics']['f1']
            if f1 >= 0.7:
                html += '<span class="badge badge-excellent">Excellent</span>'
            elif f1 >= 0.5:
                html += '<span class="badge badge-good">Good</span>'
            elif f1 >= 0.3:
                html += '<span class="badge badge-moderate">Moderate</span>'
            else:
                html += '<span class="badge badge-poor">Poor</span>'

        html += """
                        </h3>
"""

        if 'high_hr' in target_results and target_results['high_hr'] is not None:
            high_hr = target_results['high_hr']

            html += f"""
                        <div class="info-grid">
                            <div class="info-item">
                                <strong>Samples</strong>
                                {len(high_hr['y_train']) + len(high_hr['y_test'])}
                            </div>
                            <div class="info-item">
                                <strong>Features Used</strong>
                                {len(high_hr['selected_features'])}
                            </div>
                            <div class="info-item">
                                <strong>SMOTE</strong>
                                {'Applied' if high_hr['smote_applied'] else 'Not Applied'}
                            </div>
                            <div class="info-item">
                                <strong>Optimal Threshold</strong>
                                {high_hr['metrics']['optimal_threshold']:.2f}
                            </div>
                        </div>

                        <div class="features-list">
                            <strong>Selected Features:</strong>
                            <ul>
                                {''.join([f'<li>{f} (β={high_hr["coefficients"][f]:.4f})</li>' for f in high_hr['selected_features']])}
                            </ul>
                        </div>

                        <div class="metrics-grid">
                            <div class="metric-box">
                                <h4>Train Accuracy</h4>
                                <p>{high_hr['metrics']['train_accuracy']:.3f}</p>
                            </div>
                            <div class="metric-box">
                                <h4>Test Accuracy</h4>
                                <p>{high_hr['metrics']['test_accuracy']:.3f}</p>
                            </div>
                            <div class="metric-box">
                                <h4>Precision</h4>
                                <p>{high_hr['metrics']['precision']:.3f}</p>
                            </div>
                            <div class="metric-box">
                                <h4>Recall</h4>
                                <p>{high_hr['metrics']['recall']:.3f}</p>
                            </div>
                            <div class="metric-box">
                                <h4>F1-Score</h4>
                                <p>{high_hr['metrics']['f1']:.3f}</p>
                            </div>
                            <div class="metric-box">
                                <h4>ROC-AUC</h4>
                                <p>{high_hr['metrics']['roc_auc']:.3f}</p>
                            </div>
                        </div>

                        <div class="plot-container">
                            <h4>Feature Importance</h4>
                            <img src="data:image/png;base64,{high_hr['plots']['feature_importance']}" alt="Feature Importance">
                        </div>

                        <div class="plot-container">
                            <h4>Confusion Matrix</h4>
                            <img src="data:image/png;base64,{high_hr['plots']['confusion_matrix']}" alt="Confusion Matrix">
                        </div>

                        <div class="plot-container">
                            <h4>ROC Curve</h4>
                            <img src="data:image/png;base64,{high_hr['plots']['roc_curve']}" alt="ROC Curve">
                        </div>

                        <div class="plot-container">
                            <h4>Precision-Recall Curve</h4>
                            <img src="data:image/png;base64,{high_hr['plots']['pr_curve']}" alt="Precision-Recall Curve">
                        </div>

                        <div class="plot-container">
                            <h4>Threshold Optimization</h4>
                            <img src="data:image/png;base64,{high_hr['plots']['threshold_opt']}" alt="Threshold Optimization">
                        </div>
"""
        else:
            html += """
                        <p style="color: #721c24; padding: 20px; background: #f8d7da; border-radius: 4px;">
                            No data available for High HR Activity
                        </p>
"""

        html += """
                    </div>
                </div>
            </div>
"""

    html += """
        </div>

        <div class="footer">
            <p>Logistic Regression Analysis | HRV Metrics + Clinical Covariates</p>
            <p style="margin-top: 8px;">Low HR Activity (Sleep) | High HR Activity (Awake)</p>
        </div>
    </div>
</body>
</html>
"""

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"\n{'='*70}")
    print(f"HTML report saved to '{output_file}'")
    print(f"{'='*70}")


# ===========================================================================
# MAIN EXECUTION
# ===========================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("LOGISTIC REGRESSION - DUAL STATE FLARE PREDICTION")
    print("="*70)
    print(f"Forward Selection: {APPLY_FORWARD_SELECTION}")
    print(f"Capping: {APPLY_CAPPING}")
    print(f"SMOTE: {APPLY_SMOTE}")
    print(f"Scaling: {APPLY_SCALING}")
    print(f"Threshold Optimization: {APPLY_THRESHOLDOPT}")
    print(f"Covariates: {', '.join(COVARIATES)}")
    print("="*70 + "\n")

    # Define HRV metrics
    hrv_metrics = [
        "HRV_RMSSD", "HRV_SDNN", "HRV_HTI",
        "HRV_ULF", "HRV_VLF", "HRV_LF", "HRV_HF", "HRV_VHF",
        "HRV_TP", "HRV_LFHF", "HRV_LFn", "HRV_HFn", "HRV_LnHF"
    ]

    # Load data
    clinical_df, df_low_hr, df_high_hr = load_and_prepare_data(hrv_metrics)

    # Use hrv_metrics as hrv_features
    hrv_features = hrv_metrics

    # Store all results
    all_results = {}

    # Process each target
    for target in TARGETS:
        print(f"\n\n{'#'*70}")
        print(f"# PROCESSING TARGET: {target}")
        print(f"{'#'*70}\n")

        all_results[target] = {}

        # Process both HR activity states
        for state_name, hrv_df in [('low_hr', df_low_hr), ('high_hr', df_high_hr)]:
            state_label = 'Low HR Activity' if state_name == 'low_hr' else 'High HR Activity'

            print(f"\n{'='*70}")
            print(f"Processing: {target} - {state_label}")
            print(f"{'='*70}")

            # Prepare data
            data = prepare_target_data(clinical_df, hrv_df, target, hrv_features, state_label)

            if data is None:
                all_results[target][state_name] = None
                continue

            # Forward feature selection
            selected_features, selection_history = perform_forward_selection(
                data['X'], data['y'], {
                    'use_smote': APPLY_SMOTE,
                    'cv_folds': CV_FOLDS,
                    'random_state': RANDOM_STATE
                }
            )

            if not selected_features:
                print(f"✗ No features selected for {target} - {state_label}")
                all_results[target][state_name] = None
                continue

            # Train final model
            model_results = train_final_model(
                data['X'], data['y'], selected_features, state_label, target
            )

            # Generate plots
            print(f"\n✓ Generating plots for {target} - {state_label}...")

            plots = {}
            plots['feature_importance'] = fig_to_base64(
                create_feature_importance_plot(model_results['coefficients'], target, state_label)
            )
            plots['confusion_matrix'] = fig_to_base64(
                create_confusion_matrix_plot(model_results['metrics']['confusion_matrix'], target, state_label)
            )
            plots['roc_curve'] = fig_to_base64(
                create_roc_curve_plot(
                    model_results['y_test'],
                    model_results['y_pred_proba_test'],
                    model_results['metrics']['roc_auc'],
                    target, state_label
                )
            )
            plots['pr_curve'] = fig_to_base64(
                create_precision_recall_curve_plot(
                    model_results['y_test'],
                    model_results['y_pred_proba_test'],
                    model_results['metrics']['avg_precision'],
                    target, state_label
                )
            )
            plots['threshold_opt'] = fig_to_base64(
                create_threshold_optimization_plot(
                    model_results['thresholds_list'],
                    model_results['f1_scores_list'],
                    model_results['metrics']['optimal_threshold'],
                    target, state_label
                )
            )

            model_results['plots'] = plots
            all_results[target][state_name] = model_results

    # Generate HTML report
    print(f"\n\n{'='*70}")
    print("GENERATING HTML REPORT")
    print(f"{'='*70}")
    create_html_report(all_results, output_file='logistic_regression_dual_state_90m.html')

    # Save CSV summary
    print(f"\n{'='*70}")
    print("GENERATING CSV SUMMARY")
    print(f"{'='*70}")

    summary_data = []
    for target in TARGETS:
        if target not in all_results:
            continue

        for state_name in ['low_hr', 'high_hr']:
            if state_name in all_results[target] and all_results[target][state_name] is not None:
                result = all_results[target][state_name]
                state_label = 'Low HR Activity' if state_name == 'low_hr' else 'High HR Activity'

                summary_data.append({
                    'Target': target,
                    'HR_Activity': state_label,
                    'N_Features': len(result['selected_features']),
                    'Features': ', '.join(result['selected_features']),
                    'SMOTE_Applied': result['smote_applied'],
                    'Optimal_Threshold': result['metrics']['optimal_threshold'],
                    'Train_Accuracy': result['metrics']['train_accuracy'],
                    'Test_Accuracy': result['metrics']['test_accuracy'],
                    'Precision': result['metrics']['precision'],
                    'Recall': result['metrics']['recall'],
                    'F1_Score': result['metrics']['f1'],
                    'ROC_AUC': result['metrics']['roc_auc'],
                    'Avg_Precision': result['metrics']['avg_precision'],
                    'C': result['best_params']['C'],
                    'L1_Ratio': result['best_params']['l1_ratio'],
                    'Class_Weight': result['best_params']['class_weight']
                })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('logistic_regression_summary_dual_state.csv', index=False)
    print("✓ CSV summary saved to 'logistic_regression_summary_dual_state.csv'")

    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print("\nGenerated:")
    print("  - logistic_regression_dual_state_90m.html")
    print("  - logistic_regression_summary_dual_state.csv")
    print(f"\nCoverage: {len([t for t in TARGETS if t in all_results])} targets × 2 states")
    print()
