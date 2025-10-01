import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, ttest_ind, shapiro, pearsonr, spearmanr
from v3_utils import FileUtils


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






