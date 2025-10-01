import os
import pandas as pd
import matplotlib.pyplot as plt
import math

all_patient_mean_hrv_measurement_path1 = r'C:\Users\spbtu\Documents\hrvThesis\patient_metrics_windowsize600m\aw'
all_patient_list1 = os.listdir(all_patient_mean_hrv_measurement_path1)
all_hrv_list1 = []

all_patient_mean_hrv_measurement_path2 = r'C:\Users\spbtu\Documents\hrvThesis\patient_metrics_windowsize600m\sl'
all_patient_list2 = os.listdir(all_patient_mean_hrv_measurement_path2)
all_hrv_list2 = []

# Load data for df1
for pnt in all_patient_list1:
    if pnt == 'p2010001' or pnt == 'p2070001':
        continue

    pnt_parquet_path = os.path.join(all_patient_mean_hrv_measurement_path1, pnt)
    pnt_mean_hrv_df = pd.read_parquet(pnt_parquet_path)
    all_hrv_list1.append(pnt_mean_hrv_df)

# Load data for df2 (adjust path or logic as needed for your second dataset)
for pnt in all_patient_list2:
    if pnt == 'p2010001' or pnt == 'p2070001':
        continue

    pnt_parquet_path = os.path.join(all_patient_mean_hrv_measurement_path2, pnt)  # Modify path if different
    pnt_mean_hrv_df = pd.read_parquet(pnt_parquet_path)
    all_hrv_list2.append(pnt_mean_hrv_df)

df1 = pd.concat(all_hrv_list1, ignore_index=True)
df2 = pd.concat(all_hrv_list2, ignore_index=True)

# Save DataFrames to CSV (optional, based on original code)
df1.to_csv("window600m_hrv_mean_measurements_aw.csv")
df2.to_csv("window600m_hrv_mean_measurements_sl.csv")

# Get columns excluding the last 5
plot_columns = df1.columns[:-5]  # Assuming df1 and df2 have the same columns

# Calculate the number of rows and columns for the subplot grid
n_cols = len(plot_columns)
n_rows = math.ceil(n_cols / 3)  # At most 3 plots per row
fig, axes = plt.subplots(n_rows, min(n_cols, 3), figsize=(15, 4 * n_rows))
axes = axes.flatten() if n_cols > 1 else [axes]  # Handle single column case

for idx, col in enumerate(plot_columns):
    # Plot histogram for df1 in red
    df1[col].dropna().hist(bins=20, ax=axes[idx], color='red', alpha=0.5, label='awake')
    # Plot histogram for df2 in blue
    df2[col].dropna().hist(bins=20, ax=axes[idx], color='blue', alpha=0.5, label='sleeping')
    axes[idx].set_title(f"Ιστόγραμμα για τη στήλη: {col}")
    axes[idx].set_xlabel(col)
    axes[idx].set_ylabel("Συχνότητα")
    axes[idx].grid(False)
    axes[idx].legend()

# Hide any unused subplots
for idx in range(len(plot_columns), len(axes)):
    axes[idx].set_visible(False)

# Add figure title
fig.suptitle("hystogram AW/SL, sliding_window_size = 500m (3x500)", fontsize=16)

# Increase spacing between plots
plt.tight_layout(pad=3.0, w_pad=4.0, h_pad=4.0)
# Adjust layout to prevent overlap with figure title
fig.subplots_adjust(top=0.95)
plt.show()