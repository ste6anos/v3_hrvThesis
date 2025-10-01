import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, ttest_ind, shapiro, pearsonr, spearmanr
from v3_utils import FileUtils



# def extract_crp_data_from_csv(file_path):
#     df = pd.read_csv(file_path, index_col=0)
#     columns = ['patientid', 'DOC_PAT_CRP_DATE', 'DOC_PAT_CRP_LEVEL', 'DOC_PAT_CRP_UNIT']
#     crp_df = df[columns].copy()
#     crp_df['DOC_PAT_CRP_LEVEL'] = crp_df['DOC_PAT_CRP_LEVEL'].where(df['DOC_PAT_CRP_UNIT'] != 'mg/dL', df['DOC_PAT_CRP_LEVEL']*10)
#     crp_df = crp_df[["patientid", "DOC_PAT_CRP_LEVEL"]]
#     return crp_df

# def hypothesis_test_binary_target(df1, df2):
# # Mann-Whitney U (if not normal samples) or ttesst (if normal samples)
#     results = {}
#     metrics = df1.select_dtypes(include='number').columns
#     for metric in metrics:

#         _, p1 = shapiro(df1[metric].dropna())
#         _, p2 = shapiro(df2[metric].dropna())
        
#         if p1 > 0.05 and p2 > 0.05:
#             t_stat, p_value = ttest_ind(df1[metric].dropna(), df2[metric].dropna(), equal_var=False)
#             results[metric] = {"method": "t_test", "p-value": p_value, "p1": p1, "p2": p2}
#         else:
#             u_stat, p_value = mannwhitneyu(df1[metric].dropna(), df2[metric].dropna(), alternative="two-sided")
#             results[metric] = {"method": "u_test", "p-value": p_value, "p1": p1, "p2": p2}

        
#     results_df = pd.DataFrame(results).T 
#     results_df = results_df.sort_values("p-value")
#     return results_df

# def correlation_test(df):
    
#     metric_cols = df.columns[:-1]

#     target_name = df.columns[-1]
#     target_col = df[target_name]
    
#     df = df.dropna(subset=[target_name])
#     numeric_cols = df.select_dtypes(include=[np.number]).columns
#     metric_cols = [col for col in numeric_cols if col != target_name]

#     results = {}  
#     for metric in metric_cols:

#         temp_df = df.dropna(subset=[metric])
        
#         metric_clean = temp_df[metric]
#         target_aligned = temp_df[target_name]
#         _, p_metric = shapiro(metric_clean)
#         _, p_target = shapiro(target_aligned)
        
        
#         if p_metric > 0.05 and p_target > 0.05:
#             # Both normal: Pearson test
#             corr_stat, p_value = pearsonr(metric_clean, target_aligned)
#             method = "pearson"
#         else:
#             # Spearman
#             corr_stat, p_value = spearmanr(metric_clean, target_aligned)
#             method = "spearman"
        
#         results[metric] = {
#             "method": method,
#             "correlation": corr_stat,
#             "p-value": p_value,
#             "p_metric_normal": p_metric,
#             "p_target_normal": p_target
#         }
    
#     if not results: 
#         return pd.DataFrame()
    
#     results_df = pd.DataFrame(results).T
#     results_df = results_df.sort_values("p-value")
#     return results_df
        




# Load the processed metrics and clinical data
clinical_data_df = pd.read_csv(r"C:\Users\spbtu\Documents\dataset_psath\bbi_metadata\clinical_d32_T0_v2.csv")
crp_df = FileUtils.extract_crp_data_from_csv(r"C:\Users\spbtu\Documents\dataset_psath\bbi_metadata\clinical_d32_T0_v2.csv")

d32_patients = clinical_data_df['patientid'].tolist()

df_aw = pd.read_csv(r"C:\Users\spbtu\Documents\hrvThesis\window300m_hrv_mean_measurements_aw.csv", index_col=0)
df_sl = pd.read_csv(r"C:\Users\spbtu\Documents\hrvThesis\window300m_hrv_mean_measurements_sl.csv", index_col=0)
df_sl.to_csv("dfsl.csv")

metric_true_patiendid_df = clinical_data_df.loc[clinical_data_df["DOC_FLARE"] == "Yes", "patientid"]
metric_false_patiendid_df = clinical_data_df.loc[clinical_data_df["DOC_FLARE"] == "No", "patientid"]

#-------------------Hypothesis test----------------------------
aw_true_df = df_aw[df_aw["patientid"].isin(metric_true_patiendid_df)]
aw_true_df = aw_true_df[aw_true_df["state"] == "awake"]
aw_false_df = df_aw[df_aw["patientid"].isin(metric_false_patiendid_df)]
aw_false_df = aw_false_df[aw_false_df["state"] == "awake"]

sl_true_df = df_sl[df_sl["patientid"].isin(metric_true_patiendid_df)]
sl_true_df = sl_true_df[sl_true_df["state"] == "sleeping"]
sl_false_df = df_sl[df_sl["patientid"].isin(metric_false_patiendid_df)]
sl_false_df = sl_false_df[sl_false_df["state"] == "sleeping"]

aw_true_df.to_csv("aw_true_df.csv")
aw_false_df.to_csv("aw_false_df.csv")



print("\nawake\n", FileUtils.hypothesis_test_binary_target(aw_true_df, aw_false_df))
print("\nsleep\n", FileUtils.hypothesis_test_binary_target(sl_true_df, sl_false_df))


#----------------Correlation Test--------------------------

aw_and_crp_df = pd.merge(df_aw, crp_df, on="patientid")
sl_and_crp_df = pd.merge(df_sl, crp_df, on="patientid")

# print(aw_and_crp_df.head())
# print(aw_and_crp_df[aw_and_crp_df.columns[-1]].describe())

# last_col_name = aw_and_crp_df.columns[-1]  # Get the name of the last column
# aw_and_crp_df[last_col_name].plot(kind='box',  title=f'bins of {last_col_name}')
# # plt.xlabel(last_col_name)
# plt.ylabel(last_col_name)
# plt.show()  # Displays the plot (in notebooks/scripts)

print("\n awake - crp correlation\n", FileUtils.correlation_test(aw_and_crp_df))
print("\n sleeping - crp correlation\n", FileUtils.correlation_test(aw_and_crp_df))




# logistic regression lasso, ridge, elastic net





