# HRV Analysis Pipeline

## Overview
This repository contains a pipeline for processing heart rate variability (HRV) data to compute awake and sleep metrics for patients. The pipeline processes beat-to-beat interval (BBI) data stored in Parquet files, applies preprocessing, segments data into awake and sleep periods, and calculates HRV metrics using sliding windows. Results are saved as Parquet files for further analysis.

## Repository Structure

- **`v3_utils.py`**: Contains the `FileUtils` class with static methods for data preprocessing, awake/sleep segmentation, and HRV metric calculations.
- **`v3_pipeline.py`**: The main pipeline script that orchestrates the processing of patient BBI data, integrating clinical metadata and saving results.
- **`preprocessing.py`**: Defines the `HRVPreprocessor` class for applying HRV preprocessing rules to BBI data (assumed external dependency; not included here).

## Scripts Description

- **v3_utils.py**:
  - `FileUtils.preprocess_dataframe`: Applies HRV preprocessing rules (e.g., 'malik') to filter valid BBI data.
  - `FileUtils.calc_awake_sleep_dfs`: Segments data into awake and sleep periods based on BBI thresholds.
  - `FileUtils.window_check`: Validates sliding windows for sufficient BBI data.
  - `FileUtils.group_and_apply_sliding_window_calculations`: Computes HRV metrics (e.g., RMSSD, SDNN) over sliding windows.

- **v3_pipeline.py**:
  - Loads clinical metadata and BBI files.
  - Filters patients based on clinical data and available BBI recordings.
  - Processes each patient’s data using `FileUtils` methods, saving results in Parquet format.

- **preprocessing.py**:
  - Defines the `HRVPreprocessor` class for preprocessing BBI data with rules like 'range_250_2000', 'karlsson', 'acar', and 'malik' (implementation not provided).

## Notes

- Ensure the data directory structure matches the paths specified in `v3_pipeline.py`.
- Handle duplicate indices or time values in input data to avoid warnings (see `v3_utils.py` for duplicate handling).
- Adjust the `window_check_threshold` parameter in `v3_pipeline.py` if different thresholds are needed for awake and sleep states.
- For large datasets, consider enabling patient segmentation in `v3_pipeline.py` to manage memory usage.

