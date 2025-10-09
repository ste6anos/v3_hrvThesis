from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from v3_utils import StatsUtils
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

clinical_data_df = pd.read_csv(r"C:\Users\spbtu\Documents\dataset_psath\bbi_metadata\clinical_d32_T0_v2.csv")
crp_df = StatsUtils.extract_crp_data_from_csv(r"C:\Users\spbtu\Documents\dataset_psath\bbi_metadata\clinical_d32_T0_v2.csv")

d32_patients = clinical_data_df['patientid'].tolist()

df_aw = pd.read_csv(r"C:\Users\spbtu\Documents\hrvThesis\window1500m_hrv_mean_measurements_aw.csv", index_col=0)
df_sl = pd.read_csv(r"C:\Users\spbtu\Documents\hrvThesis\window1500m_hrv_mean_measurements_sl.csv", index_col=0)


aw_and_crp_df = pd.merge(df_aw, crp_df, on="patientid")
sl_and_crp_df = pd.merge(df_sl, crp_df, on="patientid")

df_clean = sl_and_crp_df.dropna(subset=['DOC_PAT_CRP_LEVEL'])
df_clean = df_clean[df_clean["DOC_PAT_CRP_LEVEL"]<=100]
df_clean = df_clean.dropna()

print("Logistic regression: windos1500m_sl, target = crp_binary\n")
# Define features and target

all_features = [
    "HRV_RMSSD", "HRV_SDNN", "HRV_HTI",
    "HRV_ULF", "HRV_VLF", "HRV_LF", "HRV_HF", "HRV_VHF",
    "HRV_TP", "HRV_LFHF", "HRV_LFn", "HRV_HFn", "HRV_LnHF"
]

y = (df_clean['DOC_PAT_CRP_LEVEL'] >= 4).astype(int)


# Forward selection parameters

selected_features = []
remaining_features = all_features.copy()
best_score = 0
max_features = len(all_features)  # or stop earlier if you want
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Grid for hyperparameter search
param_grid = {
    'C': np.arange(0.1, 10.1, 0.5),      # coarse grid for speed
    'l1_ratio': np.arange(0.1, 1.1, 0.2)
}


# Forward selection loop

while remaining_features and len(selected_features) < max_features:
    scores = []
    for feat in remaining_features:
        current_features = selected_features + [feat]
        X = df_clean[current_features]
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Logistic Regression Elastic Net
        model = LogisticRegression(
            penalty='elasticnet',
            solver='saga',
            max_iter=10000,
            random_state=42
        )
        
        # Grid search for best C/l1_ratio
        grid_search = GridSearchCV(model, param_grid, cv=kf, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_scaled, y)
        
        scores.append({
            'feature': feat,
            'cv_score': grid_search.best_score_,
            'best_C': grid_search.best_params_['C'],
            'best_l1_ratio': grid_search.best_params_['l1_ratio']
        })
    
    # Pick the feature that gives the best CV accuracy when added
    best_candidate = max(scores, key=lambda x: x['cv_score'])
    
    # If adding improves score, accept it
    if best_candidate['cv_score'] > best_score:
        selected_features.append(best_candidate['feature'])
        remaining_features.remove(best_candidate['feature'])
        best_score = best_candidate['cv_score']
        print(f"Added feature: {best_candidate['feature']}, CV accuracy: {best_score:.3f}, best C: {best_candidate['best_C']}, best l1_ratio: {best_candidate['best_l1_ratio']}")
    else:
        # Stop if no improvement
        break


# Retrain model with selected features

X_final = df_clean[selected_features]
scaler = StandardScaler()
X_final_scaled = scaler.fit_transform(X_final)

X_train, X_test, y_train, y_test = train_test_split(
    X_final_scaled, y, test_size=0.2, random_state=42, stratify=y
)

final_model = LogisticRegression(
    penalty='elasticnet',
    solver='saga',
    max_iter=10000,
    C=best_candidate['best_C'],
    l1_ratio=best_candidate['best_l1_ratio'],
    random_state=42
)

final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print("Test accuracy: {:.3f}".format(acc))


