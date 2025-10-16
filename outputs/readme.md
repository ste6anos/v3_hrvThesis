### *Results*:

```
Key Findings on hypothesis tests:
Primary Clinical Focus: DOC_FLARE and PAT_FLARE
These targets were prioritized as they best represent the overall clinical picture.
- RMSSD during awake state consistently differentiates DOC_FLARE across all window sizes
- PAT_FLARE shows similar differentiation with RMSSD during awake state, though with 
  slight uncertainty (p-values of 0.052 for 600m and 1500m window sizes, approaching significance)
- Other metrics show inconsistent patterns across different conditions
- Notable tendencies observed:
  ~VHF, ULF, and LnHF during sleep classify PAT_FLARE in long-lasting windows (and in the shortest 90m window)
  ~SDNN and HTI during sleep classify DOC_FLARE in long-lasting windows

CRP Value Analysis:
- Correlation tests reveal no evidence of linear relationship (Pearson/Spearman) 
  between any HRV metrics and CRP values
- Logistic regression using CRP threshold of 4 mg/L (high/low classification)
  shows limited separation capability with average accuracy of 0.65
- No concrete evidence of HRV metrics' predictive ability for CRP levels
```
------------------------------------------------

## **1)** Hypothesis/Correlation tests and hrv_metric plots per windowsize

This section summarizes the HRV (Heart Rate Variability) metrics computed for different sliding window sizes (90m, 180m, 300m, 600m, and 1500m) using the analysis pipeline in `v3_metricsAnalysis.py`. For each window size:

- **HRV Metrics Plot**: Visualizes the distribution of key HRV metrics (e.g., RMSSD, SDNN, LF/HF ratio) across awake (threshold 0.5) and sleep (threshold 0.45) states.
- **Hypothesis Test Table**: Displays results from binary hypothesis equality tests against clinical targets (e.g., flare status, sex, smoking history), including p-values and test methods. Significant results (p < 0.05) are color-coded by state (awake: light blue; sleep: light green). The t-test or u-test is referred as (t) & (u) in the results table.
- **Correlation Test**: A correlation test on CRP values has been added to the last row of the results table.

Results are generated in the `outputs/` directory and visualized below for quick reference.

**The thresholds for long-sized windows are lower in the sleeping state because the window_check function is applied to whole-day data, where the sleep segment is relatively smaller than the awake segment. For example, if the window size is 1500 min (= 25 h) and the subject normally sleeps for 8 h, the threshold is <0.31 by definition.**

### 90m Window

**HRV Metrics Plot**  
![HRV Metrics - 90m](https://github.com/ste6anos/v3_hrvThesis/blob/main/outputs/plot_w90m_threshold_aw05_sl045.png?raw=true)

**Hypothesis Test Table**  
![Hypothesis Tests - 90m](https://github.com/ste6anos/v3_hrvThesis/blob/main/outputs/W90M.png)

### 180m Window

**HRV Metrics Plot**  
![HRV Metrics - 180m](https://github.com/ste6anos/v3_hrvThesis/blob/main/outputs/plot_w180m_aw05_sl045.png?raw=true)

**Hypothesis Test Table**  
![Hypothesis Tests - 180m](https://github.com/ste6anos/v3_hrvThesis/blob/main/outputs/W180M.png)

### 300m Window

**HRV Metrics Plot**   
![HRV Metrics - 300m](https://github.com/ste6anos/v3_hrvThesis/blob/main/outputs/plot_w300m_aw05_sl045.png?raw=true)

**Hypothesis Test Table**  
![Hypothesis Tests - 300m](https://github.com/ste6anos/v3_hrvThesis/blob/main/outputs/W300M.png?raw=true)

### 600m Window

**HRV Metrics Plot**  
![HRV Metrics - 600m](https://github.com/ste6anos/v3_hrvThesis/blob/main/outputs/plot_w600m.png?raw=true)

**Hypothesis Test Table**   
![Hypothesis Tests - 600m](https://github.com/ste6anos/v3_hrvThesis/blob/main/outputs/W600M.png?raw=true)

### 1500m Window

**HRV Metrics Plot**  
![HRV Metrics - 1500m](https://github.com/ste6anos/v3_hrvThesis/blob/main/outputs/wind1500mSL025AW050.png?raw=true)

**Hypothesis Test Table**  
![Hypothesis Tests - 1500m](https://github.com/ste6anos/v3_hrvThesis/blob/main/outputs/W1500M.png?raw=true)

## **2)** Logistic Regression 

**Classification Performance Across All Window Sizes**  
![Logistic Regression Results](https://github.com/ste6anos/v3_hrvThesis/blob/main/outputs/logisticRegression_results.png?raw=true)
### Key Insights
- **Metrics Included**: RMSSD, SDNN, HTI, VHF, LF/HF, LFn, TP, LF, ULF, VLF, HF, HFn, LnHF.
- **Testing**: Binary t-tests, as (t), or Mann-Whitney U-tests, as (u), (auto-selected based on normality) for each target vs. HRV metrics, separated by awake/sleep states.
- **Correlations**: CRP correlations are appended as the final row in tables, formatted as `p-value(correlation)`.
- **Significance**: Cells highlight p < 0.05 results for easy scanning.

For raw data and scripts, see the [outputs/](https://github.com/ste6anos/v3_hrvThesis/tree/main/outputs) folder. Run `v3_csvMetricsAnalysis.py` and `v3_dfploting` with adjusted window parameters to regenerate.
