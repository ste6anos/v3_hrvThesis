import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import sys
import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, ttest_ind, shapiro, pearsonr, spearmanr
from v3_utils import StatsUtils
import pandas as pd

# Assuming StatsUtils is available and imported
# from your_module import StatsUtils  # Adjust as needed

# Load the processed metrics and clinical data
clinical_data_df = pd.read_csv(r"C:\Users\spbtu\Documents\dataset_psath\bbi_metadata\clinical_d32_T0_v2.csv")
crp_df = StatsUtils.extract_crp_data_from_csv(r"C:\Users\spbtu\Documents\dataset_psath\bbi_metadata\clinical_d32_T0_v2.csv")

d32_patients = clinical_data_df['patientid'].tolist()

df_aw = pd.read_csv(r"C:\Users\spbtu\Documents\hrvThesis\window1500m_hrv_mean_measurements_aw.csv", index_col=0)
df_sl = pd.read_csv(r"C:\Users\spbtu\Documents\hrvThesis\window1500m_hrv_mean_measurements_sl.csv", index_col=0)
df_sl.to_csv("dfsl.csv")

target_list = ["DOC_FLARE","DOC_PAT_COMOR_OBESE", 
               "DEMOGR_SEX", "PAT_SMOKE_H", 
               "DOC_DACTYLITIS_BY_RHEUMA", "DOC_PSORIASIS_HIST", 
               "DOC_RHEUMA_FACTOR_NEG", "PAT_PAIN_JOINTS", "DOC_DIS_ACT", 
               "PAT_FLARE", "PAT_PASS", "PAT_REM"]

# Collect all results
results_data = []

for target in target_list:
    metric_true_patiendid_df = clinical_data_df.loc[ (clinical_data_df[target] == "Yes") |   
                                                    (clinical_data_df[target] == 1) |  
                                                    (clinical_data_df[target] == "Male") , "patientid"]
    
    metric_false_patiendid_df = clinical_data_df.loc[(clinical_data_df[target] == "No") |   
                                                    (clinical_data_df[target] == 0) |  
                                                    (clinical_data_df[target] == "Female") , "patientid"]

    aw_true_df = df_aw[df_aw["patientid"].isin(metric_true_patiendid_df)]
    aw_true_df = aw_true_df[aw_true_df["state"] == "awake"]
    aw_false_df = df_aw[df_aw["patientid"].isin(metric_false_patiendid_df)]
    aw_false_df = aw_false_df[aw_false_df["state"] == "awake"]

    sl_true_df = df_sl[df_sl["patientid"].isin(metric_true_patiendid_df)]
    sl_true_df = sl_true_df[sl_true_df["state"] == "sleeping"]
    sl_false_df = df_sl[df_sl["patientid"].isin(metric_false_patiendid_df)]
    sl_false_df = sl_false_df[sl_false_df["state"] == "sleeping"]

    print(target)
    awake_results = StatsUtils.hypothesis_test_binary_target(aw_true_df, aw_false_df)
    sleep_results = StatsUtils.hypothesis_test_binary_target(sl_true_df, sl_false_df)
    
    # Store results for each metric
    for metric in awake_results.index:
        results_data.append({
            'target': target,
            'metric': metric,
            'state': 'awake',
            'p_value': awake_results.loc[metric, 'p-value'],
            'method': awake_results.loc[metric, 'method']
        })
    
    for metric in sleep_results.index:
        results_data.append({
            'target': target,
            'metric': metric,
            'state': 'sleep',
            'p_value': sleep_results.loc[metric, 'p-value'],
            'method': sleep_results.loc[metric, 'method']
        })

# Create DataFrame
results_df = pd.DataFrame(results_data)

# Pivot to get the desired format
pivot_data = {}
for target in results_df['target'].unique():
    target_data = results_df[results_df['target'] == target]
    pivot_data[target] = {}
    
    for _, row in target_data.iterrows():
        col_name = f"{row['metric']}_{row['state']}"
        pivot_data[target][col_name] = (row['p_value'], row['method'])

# All possible metrics
all_metrics = ['HRV_RMSSD', 'HRV_SDNN', 'HRV_HTI', 'HRV_VHF', 'HRV_LFHF', 
               'HRV_LFn', 'HRV_TP', 'HRV_LF', 'HRV_ULF', 'HRV_VLF', 
               'HRV_HF', 'HRV_HFn', 'HRV_LnHF']

# Correlation tests
aw_and_crp_df = pd.merge(df_aw, crp_df, on="patientid")
sl_and_crp_df = pd.merge(df_sl, crp_df, on="patientid")

awake_corr_results = StatsUtils.correlation_test(aw_and_crp_df)
sleep_corr_results = StatsUtils.correlation_test(sl_and_crp_df)

print("\n awake - crp correlation\n", awake_corr_results)
print("\n sleeping - crp correlation\n", sleep_corr_results)

# Add CRP to pivot_data as last entry
pivot_data['CRP'] = {}

for metric in all_metrics:
    # Awake correlation
    awake_key = f"{metric}_awake"
    if metric in awake_corr_results.index:
        p_val = awake_corr_results.loc[metric, 'p-value']
        corr = awake_corr_results.loc[metric, 'correlation']  # Assuming 'correlation' column exists
        pivot_data['CRP'][awake_key] = (p_val, corr)
    
    # Sleep correlation
    sleep_key = f"{metric}_sleep"
    if metric in sleep_corr_results.index:
        p_val = sleep_corr_results.loc[metric, 'p-value']
        corr = sleep_corr_results.loc[metric, 'correlation']  # Assuming 'correlation' column exists
        pivot_data['CRP'][sleep_key] = (p_val, corr)

# Generate HTML
def get_cell_color(p_value, state):
    if p_value < 0.05:
        if state == 'awake':
            return '#add8e6'  # Light blue for significant awake
        else:  # sleep
            return '#90ee90'  # Light green for significant sleep
    else:
        if state == 'awake':
            return '#f0f0f0'  # Light gray for non-significant awake
        else:  # sleep
            return '#ffffff'  # White for non-significant sleep

def format_p_value(p_value):
    return f"{p_value:.3f}" if p_value >= 0.001 else "<0.001"

html = """
<!DOCTYPE html>
<html>
<head>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
        }
        h1 {
            color: #333;
        }
        .legend {
            margin: 20px 0;
            padding: 10px;
            background: #f5f5f5;
            border-radius: 5px;
        }
        .legend-item {
            display: inline-block;
            margin-right: 20px;
            padding: 5px 10px;
            border: 1px solid #ccc;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            font-size: 11px;
        }
        th, td {
            border: 1px solid #ccc;
            padding: 6px;
            text-align: center;
        }
        th {
            background-color: #e0e0e0;
            font-weight: bold;
            position: sticky;
            top: 0;
        }
        td:first-child, th:first-child {
            text-align: left;
            font-weight: bold;
            position: sticky;
            left: 0;
            background-color: white;
            z-index: 1;
        }
        th:first-child {
            background-color: #e0e0e0;
            z-index: 2;
        }
        .metric-header {
            font-size: 9px;
        }
    </style>
</head>
<body>
    <h1>HRV Statistical Results: windowsize = 600m, threshold_aw = 0.5, threshold_sl = (~0.3)</h1>
    <div class="legend">
        <strong>Color coding:</strong>
        <span class="legend-item" style="background: #add8e6;">Awake, p &lt; 0.05</span>
        <span class="legend-item" style="background: #90ee90;">Sleep, p &lt; 0.05</span>
        <span class="legend-item" style="background: #f0f0f0;">Awake, p ≥ 0.05</span>
        <span class="legend-item" style="background: #ffffff;">Sleep, p ≥ 0.05</span>
    </div>
    <table>
        <thead>
            <tr>
                <th>Target</th>
"""

# Add metric headers
for metric in all_metrics:
    metric_short = metric.replace('HRV_', '')
    html += f'<th class="metric-header">{metric_short}_A</th>'
    html += f'<th class="metric-header">{metric_short}_S</th>'

html += """
            </tr>
        </thead>
        <tbody>
"""

# Add data rows for binary targets
binary_targets = sorted([t for t in pivot_data.keys() if t != 'CRP'])
for target in binary_targets:
    html += f'<tr><td>{target}</td>'
    
    for metric in all_metrics:
        # Awake
        awake_key = f"{metric}_awake"
        if awake_key in pivot_data[target]:
            p_val, method = pivot_data[target][awake_key]
            color = get_cell_color(p_val, 'awake')
            method_short = method.replace('_test', '')
            html += f'<td style="background-color: {color};">{format_p_value(p_val)} ({method_short})</td>'
        else:
            html += '<td>-</td>'
        
        # Sleep
        sleep_key = f"{metric}_sleep"
        if sleep_key in pivot_data[target]:
            p_val, method = pivot_data[target][sleep_key]
            color = get_cell_color(p_val, 'sleep')
            method_short = method.replace('_test', '')
            html += f'<td style="background-color: {color};">{format_p_value(p_val)} ({method_short})</td>'
        else:
            html += '<td>-</td>'
    
    html += '</tr>\n'

# Add CRP row last
html += '<tr><td>CRP</td>'
for metric in all_metrics:
    # Awake
    awake_key = f"{metric}_awake"
    if awake_key in pivot_data['CRP']:
        p_val, corr = pivot_data['CRP'][awake_key]
        color = get_cell_color(p_val, 'awake')
        p_formatted = format_p_value(p_val)
        corr_formatted = f"{corr:.3f}"
        html += f'<td style="background-color: {color};">{p_formatted}({corr_formatted})</td>'
    else:
        html += '<td>-</td>'
    
    # Sleep
    sleep_key = f"{metric}_sleep"
    if sleep_key in pivot_data['CRP']:
        p_val, corr = pivot_data['CRP'][sleep_key]
        color = get_cell_color(p_val, 'sleep')
        p_formatted = format_p_value(p_val)
        corr_formatted = f"{corr:.3f}"
        html += f'<td style="background-color: {color};">{p_formatted}({corr_formatted})</td>'
    else:
        html += '<td>-</td>'

html += '</tr>\n'

html += """
        </tbody>
    </table>
    <div style="margin-top: 20px; font-size: 12px; color: #666;">
        <p><strong>Abbreviations:</strong> A = Awake, S = Sleep. For CRP row: p-value(correlation)</p>
    </div>
</body>
</html>
"""

# Save to file
with open('hrv_results_table600m.html', 'w', encoding='utf-8') as f:
    f.write(html)

print("Results saved to hrv_results_table.html")

# Also save as CSV for reference
csv_rows = []
for target in binary_targets:
    row = {'Target': target}
    for metric in all_metrics:
        awake_key = f"{metric}_awake"
        sleep_key = f"{metric}_sleep"
        
        if awake_key in pivot_data[target]:
            p_val, method = pivot_data[target][awake_key]
            row[f"{metric}_awake"] = f"{format_p_value(p_val)} ({method.replace('_test', '')})"
        else:
            row[f"{metric}_awake"] = "-"
            
        if sleep_key in pivot_data[target]:
            p_val, method = pivot_data[target][sleep_key]
            row[f"{metric}_sleep"] = f"{format_p_value(p_val)} ({method.replace('_test', '')})"
        else:
            row[f"{metric}_sleep"] = "-"
    
    csv_rows.append(row)

# Add CRP row last
crp_row = {'Target': 'CRP'}
for metric in all_metrics:
    awake_key = f"{metric}_awake"
    sleep_key = f"{metric}_sleep"
    
    if awake_key in pivot_data['CRP']:
        p_val, corr = pivot_data['CRP'][awake_key]
        p_formatted = format_p_value(p_val)
        corr_formatted = f"{corr:.3f}"
        crp_row[f"{metric}_awake"] = f"{p_formatted}({corr_formatted})"
    else:
        crp_row[f"{metric}_awake"] = "-"
        
    if sleep_key in pivot_data['CRP']:
        p_val, corr = pivot_data['CRP'][sleep_key]
        p_formatted = format_p_value(p_val)
        corr_formatted = f"{corr:.3f}"
        crp_row[f"{metric}_sleep"] = f"{p_formatted}({corr_formatted})"
    else:
        crp_row[f"{metric}_sleep"] = "-"

csv_rows.append(crp_row)

pd.DataFrame(csv_rows).to_csv('hrv_results_table.csv', index=False)
print("Results also saved to hrv_results_table.csv")


