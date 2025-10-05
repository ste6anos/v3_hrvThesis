###v3_metricsAnalysis.py: HRV Metrics Statistical Analysis Script
Statistical analysis script for hypothesis testing against multiple binary clinical targets and correlations with CRP:

Loads clinical metadata (clinical_d32_T0_v2.csv) and extracts CRP data using StatsUtils.extract_crp_data_from_csv.
Loads preprocessed HRV mean measurements CSVs for awake (window1500m_hrv_mean_measurements_aw.csv) and sleep (window1500m_hrv_mean_measurements_sl.csv).
For each target in target_list (e.g., "DOC_FLARE", "DEMOGR_SEX", etc.), filters patient IDs into binary groups ("Yes"/1/"Male" vs. "No"/0/"Female"), then subsets HRV data by group and state (awake/sleeping).
Performs binary hypothesis tests using StatsUtils.hypothesis_test_binary_target for each subset (prints per-target results; collects all p-values and methods into results_df).
Merges full HRV data with CRP, runs correlation tests using StatsUtils.correlation_test (prints awake/sleep vs. CRP results; integrates into pivot structure).
Pivots results into a table format, generates color-coded HTML (hrv_results_table.html) where cells are styled by state (awake/sleep) and significance (p < 0.05: light blue/green; p ≥ 0.05: light gray/white), and saves CSV (hrv_results_table.csv) for reference. CRP row added last with format "p-value(correlation)".

Note: Window size is 1500 minutes. Includes commented code for debugging (e.g., saving subsets to CSV) and boxplot visualization of CRP. Colors: Awake (p<0.05: #add8e6; p≥0.05: #f0f0f0); Sleep (p<0.05: #90ee90; p≥0.05: #ffffff).
