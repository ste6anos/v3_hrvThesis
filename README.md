# HRV Analysis Pipeline (RESULTS AVAILABLE IN THE LAST PART OF THE README FILE)

## Overview

This repository contains a pipeline for processing heart rate variability (HRV) data to compute awake and sleep metrics for patients. The pipeline processes beat-to-beat interval (BBI) data stored in Parquet files, applies preprocessing, segments data into awake and sleep periods, and calculates HRV metrics using sliding windows. Results are saved as Parquet files for further analysis. Additional utilities support statistical hypothesis testing, correlation analysis, and data visualization for HRV metrics in relation to clinical data (e.g., CRP levels and flare status).

## Repository Structure

```
├── v3_utils.py              # Core utilities for data processing and analysis
├── v3_pipeline.py           # Main pipeline orchestration script
├── v3_dfploting.py          # Visualization script for HRV metrics
├── v3_metricsAnalysis.py    # Statistical analysis and hypothesis testing
└── preprocessing.py         # HRV preprocessing rules (external dependency)
```

## Scripts Description

### v3_utils.py

The `FileUtils` class provides static methods for core data handling and analysis tasks:

#### `FileUtils.preprocess_dataframe`
Applies all HRV preprocessing rules to filter valid BBI data:
- **Rules applied in sequence**: `range_250_2000_rule` → `karlsson_rule` → `acar_rule` → `malik_rule`
- Sorts by time, extracts intervals, and combines validity checks from multiple rules
- Returns a filtered DataFrame with `time` and `bbi` columns
- Handles errors gracefully with logging

#### `FileUtils.window_check`
Validates sliding windows for sufficient BBI data using the threshold from the FLIRT paper:
- Computes window duration in milliseconds, mean BBI, and interval count
- Checks if `(N * M) / L > threshold`, where:
  - `N` = number of intervals
  - `M` = mean BBI
  - `L` = window duration

#### `FileUtils.calc_awake_sleep_dfs`
Segments data into awake and sleep periods based on heart rate:
- **Threshold**: Sleep identified when HR < 0.9 × avg_heart_rate (sleeping HR is ~20% slower than awake)
- Computes threshold from mean BBI
- Handles duplicates in indices/times
- Assigns sleeping status (1 for sleep, -1 for awake)
- Merges short segments (< `min_duration`, e.g., '3min')
- Returns separate awake and sleep DataFrames

#### `FileUtils.group_and_apply_sliding_window_calculations`
Computes HRV metrics over sliding windows:
- **Time-domain metrics**: RMSSD, SDNN
- **Frequency-domain metrics**: ULF, VLF, LF, HF
- **Default window settings**: 3 frames at 100min offset
- Bins time data, iterates over windows with progress tracking (tqdm)
- Validates with `window_check`
- Uses NeuroKit2 for calculations
- Returns the mean across valid windows
- Includes garbage collection for memory management

#### `FileUtils.extract_crp_data_from_csv`
Extracts and normalizes C-reactive protein (CRP) data from a CSV file:
- Selects relevant columns: `patientid`, `DOC_PAT_CRP_DATE`, `DOC_PAT_CRP_LEVEL`, `DOC_PAT_CRP_UNIT`
- Converts mg/dL units to mg/L by multiplying by 10
- Returns a cleaned DataFrame with `patientid` and normalized `DOC_PAT_CRP_LEVEL`

#### `FileUtils.hypothesis_test_binary_target`
Performs group comparison tests:
- Uses independent t-test if both groups are normal
- Uses Mann-Whitney U test otherwise
- Compares shared numeric metrics between two DataFrames (e.g., flare "Yes" vs. "No" groups)
- Uses Shapiro-Wilk for normality checks
- Skips small samples (<3 per group)
- Returns a sorted DataFrame of results (method, p-value, normality p-values)

#### `FileUtils.correlation_test`
Performs correlation tests:
- Uses Pearson if both variables normal
- Uses Spearman otherwise
- Tests numeric columns (excluding the last/target) against the last column (target, e.g., CRP level)
- Drops NaNs in target upfront and per-metric rows with NaNs
- Uses Shapiro-Wilk for normality
- Skips small samples (<3)
- Returns a sorted DataFrame of results (method, correlation coefficient, p-value, normality p-values, sample size)

### v3_pipeline.py

The main pipeline script that orchestrates the processing:

1. Loads clinical metadata and BBI files
2. Filters patients based on clinical data and available BBI recordings
3. Processes each patient's data using `FileUtils` methods
4. Saves results in Parquet format

### v3_dfploting.py

Visualization script for generating histograms:

1. Loads HRV mean measurement data from directories for awake (aw) and sleep (sl) states (Parquet files per patient)
2. Concatenates patient data into two DataFrames (`df1` for awake, `df2` for sleep), skipping specific patients (e.g., 'p2010001', 'p2070001')
3. Saves concatenated DataFrames as CSVs
4. Generates overlaid histograms for all numeric columns (excluding last 5):
   - Uses Matplotlib subplots (up to 3 per row)
   - Colors: red for awake, blue for sleep
   - Includes Greek labels for titles/x/y axes and legend
   - Adjusts layout for spacing and saves/displays the figure

## v3_metricsAnalysis.py: HRV Metrics Statistical Analysis Script

Statistical analysis script for hypothesis testing against multiple binary clinical targets and correlations with CRP:

1. Loads clinical metadata (`clinical_d32_T0_v2.csv`) and extracts CRP data using `StatsUtils.extract_crp_data_from_csv`.
2. Loads preprocessed HRV mean measurements CSVs for awake (`window1500m_hrv_mean_measurements_aw.csv`) and sleep (`window1500m_hrv_mean_measurements_sl.csv`).
3. For each target in `target_list` (e.g., "DOC_FLARE", "DEMOGR_SEX", etc.), filters patient IDs into binary groups ("Yes"/1/"Male" vs. "No"/0/"Female"), then subsets HRV data by group and state (awake/sleeping).
4. Performs binary hypothesis tests using `StatsUtils.hypothesis_test_binary_target` for each subset (prints per-target results; collects all p-values and methods into `results_df`).
5. Merges full HRV data with CRP, runs correlation tests using `StatsUtils.correlation_test` (prints awake/sleep vs. CRP results; integrates into pivot structure).
6. Pivots results into a table format, generates color-coded HTML (`hrv_results_table.html`) where cells are styled by state (awake/sleep) and significance (p < 0.05: light blue/green; p ≥ 0.05: light gray/white), and saves CSV (`hrv_results_table.csv`) for reference. CRP row added last with format "p-value(correlation)".

Note: Window size is 1500 minutes. Includes commented code for debugging (e.g., saving subsets to CSV) and boxplot visualization of CRP. Colors: Awake (p<0.05: #add8e6; p≥0.05: #f0f0f0); Sleep (p<0.05: #90ee90; p≥0.05: #ffffff).

### preprocessing.py

Defines the `HRVPreprocessor` class for preprocessing BBI data with rules:
- `range_250_2000`
- `karlsson`
- `acar`
- `malik`

## Notes and Considerations

### Data Directory Structure
- Ensure the data directory structure matches the paths specified in `v3_pipeline.py` and `v3_metricsAnalysis.py`
- Required files: HRV CSVs, clinical metadata, BBI Parquet files

### Duplicate Handling
- Handle duplicate indices or time values in input data to avoid warnings
- See `v3_utils.py` for duplicate handling in `calc_awake_sleep_dfs`

### Window Check Threshold
- Adjust the `window_check_threshold` parameter in `v3_pipeline.py` or `group_and_apply_sliding_window_calculations` if different thresholds are needed for awake and sleep states

### Memory Management
- For large datasets, consider enabling patient segmentation in `v3_pipeline.py` to manage memory usage
- Garbage collection is included in sliding window calculations

### Statistical Tests
- Statistical tests in `v3_utils.py` assume numeric data and skip small samples
- Review p-value thresholds (e.g., 0.05 for normality) for your analysis

### Visualizations
- Visualizations in `v3_dfploting.py` use fixed paths and exclusions—update for new datasets






## Results

This section summarizes the HRV (Heart Rate Variability) metrics computed for different sliding window sizes (90m, 300m, 600m, and 1500m) using the analysis pipeline in `v3_metricsAnalysis.py`. For each window size:

- **HRV Metrics Plot**: Visualizes the distribution of key HRV metrics (e.g., RMSSD, SDNN, LF/HF ratio) across awake (threshold 0.5) and sleep (threshold 0.45) states.
- **Hypothesis Test Table**: Displays results from binary hypothesis equality tests against clinical targets (e.g., flare status, sex, smoking history), including p-values and test methods. Significant results (p < 0.05) are color-coded by state (awake: light blue; sleep: light green).

Results are generated in the `outputs/` directory and visualized below for quick reference.

### 90m Window

**HRV Metrics Plot**  
![HRV Metrics - 90m](https://github.com/ste6anos/v3_hrvThesis/blob/main/outputs/plot_w90m_threshold_aw05_sl045.png?raw=true)

**Hypothesis Test Table**   the t-test or u-test is referred as (u), (t)
![Hypothesis Tests - 90m](https://github.com/ste6anos/v3_hrvThesis/blob/main/outputs/W90M.png)

### 300m Window

**HRV Metrics Plot**   the t-test or u-test is referred as (u), (t)
![HRV Metrics - 300m](https://github.com/ste6anos/v3_hrvThesis/blob/main/outputs/hystogram_windowduration300m.png?raw=true)

**Hypothesis Test Table**  
![Hypothesis Tests - 300m](https://github.com/ste6anos/v3_hrvThesis/blob/main/outputs/W300M.png?raw=true)

### 600m Window

**HRV Metrics Plot**  
![HRV Metrics - 600m](https://github.com/ste6anos/v3_hrvThesis/blob/main/outputs/plot_w600m.png?raw=true)

**Hypothesis Test Table**   the t-test or u-test is referred as (u), (t)
![Hypothesis Tests - 600m](https://github.com/ste6anos/v3_hrvThesis/blob/main/outputs/W600M.png?raw=true)

### 1500m Window

**HRV Metrics Plot**  
![HRV Metrics - 1500m](https://github.com/ste6anos/v3_hrvThesis/blob/main/outputs/wind1500mSL025AW050.png?raw=true)

**Hypothesis Test Table**   the t-test or u-test is referred as (u), (t)
![Hypothesis Tests - 1500m](https://github.com/ste6anos/v3_hrvThesis/blob/main/outputs/W1500M.png?raw=true)

### Key Insights
- **Metrics Included**: RMSSD, SDNN, HTI, VHF, LF/HF, LFn, TP, LF, ULF, VLF, HF, HFn, LnHF.
- **Testing**: Binary t-tests or Mann-Whitney U-tests (auto-selected based on normality) for each target vs. HRV metrics, separated by awake/sleep states.
- **Correlations**: CRP correlations are appended as the final row in tables, formatted as `p-value(correlation)`.
- **Significance**: Cells highlight p < 0.05 results for easy scanning.

For raw data and scripts, see the [outputs/](https://github.com/ste6anos/v3_hrvThesis/tree/main/outputs) folder. Run `v3_metricsAnalysis.py` with adjusted window parameters to regenerate.

