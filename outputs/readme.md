## Results

This section summarizes the HRV (Heart Rate Variability) metrics computed for different sliding window sizes (90m, 300m, 600m, and 1500m) using the analysis pipeline in `v3_metricsAnalysis.py`. For each window size:

- **HRV Metrics Plot**: Visualizes the distribution of key HRV metrics (e.g., RMSSD, SDNN, LF/HF ratio) across awake (threshold 0.5) and sleep (threshold 0.45) states.
- **Hypothesis Test Table**: Displays results from binary hypothesis equality tests against clinical targets (e.g., flare status, sex, smoking history), including p-values and test methods. Significant results (p < 0.05) are color-coded by state (awake: light blue; sleep: light green).
- **Correlation Test**: A correlation test on CRP values has been added to the last row of the results table.

Results are generated in the `outputs/` directory and visualized below for quick reference.

**The thresholds for long-sized windows are lower in the sleeping state because the window_check function is applied to whole-day data, where the sleep segment is relatively smaller than the awake segment. For example, if the window size is 1500 min (= 25 h) and the subject normally sleeps for 8 h, the threshold is <0.31 by definition.**

### 90m Window

**HRV Metrics Plot**  
![HRV Metrics - 90m](https://github.com/ste6anos/v3_hrvThesis/blob/main/outputs/plot_w90m_threshold_aw05_sl045.png?raw=true)

**Hypothesis Test Table**   the t-test or u-test is referred as (u), (t)
![Hypothesis Tests - 90m](https://github.com/ste6anos/v3_hrvThesis/blob/main/outputs/W90M.png)

### 300m Window

**HRV Metrics Plot**   
![HRV Metrics - 300m](https://github.com/ste6anos/v3_hrvThesis/blob/main/outputs/hystogram_windowduration300m.png?raw=true)

**Hypothesis Test Table**  the t-test or u-test is referred as (u), (t)
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
- **Testing**: Binary t-tests, as (t), or Mann-Whitney U-tests, as (u), (auto-selected based on normality) for each target vs. HRV metrics, separated by awake/sleep states.
- **Correlations**: CRP correlations are appended as the final row in tables, formatted as `p-value(correlation)`.
- **Significance**: Cells highlight p < 0.05 results for easy scanning.

For raw data and scripts, see the [outputs/](https://github.com/ste6anos/v3_hrvThesis/tree/main/outputs) folder. Run `v3_csvMetricsAnalysis.py` and `v3_dfploting` with adjusted window parameters to regenerate.
