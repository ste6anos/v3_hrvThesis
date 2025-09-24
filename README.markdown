# HRV Analysis Pipeline

## Overview
This repository contains a pipeline for processing heart rate variability (HRV) data to compute awake and sleep metrics for patients. The pipeline processes beat-to-beat interval (BBI) data stored in Parquet files, applies preprocessing, segments data into awake and sleep periods, and calculates HRV metrics using sliding windows. Results are saved as Parquet files for further analysis.

## Repository Structure

- **`v3_utils.py`**: Contains the `FileUtils` class with static methods for data preprocessing, awake/sleep segmentation, and HRV metric calculations.
- **`v3_pipeline.py`**: The main pipeline script that orchestrates the processing of patient BBI data, integrating clinical metadata and saving results.
- **`preprocessing.py`**: Defines the `HRVPreprocessor` class for applying HRV preprocessing rules to BBI data (assumed external dependency; not included here).

## Prerequisites

- **Python**: Version 3.8+
- **Dependencies**:
  ```bash
  pip install pandas numpy python-dateutil tqdm neurokit2
  ```
- **Data**:
  - BBI data in Parquet format, organized by patient ID in a directory (e.g., `dataset_psath/bbi/<patient_id>`).
  - Clinical metadata in CSV format (e.g., `dataset_psath/bbi_metadata/clinical_d32_T0_v2.csv`).

## Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the data directory paths in `v3_pipeline.py` match your local setup:
   ```python
   all_patient_bbi_path = Path(r'<path-to-bbi-data>')
   clinical_data_path = Path(r'<path-to-clinical-data>')
   ```

## Usage

Run the pipeline to process patient data and generate HRV metrics:

```bash
python v3_pipeline.py
```

- **Input**: BBI Parquet files and clinical metadata CSV.
- **Output**: HRV metrics saved as Parquet files in `patient_metrics_windowduration1500m/aw/` and `patient_metrics_windowduration1500m/sl/` directories.
- **Logs**: Processing details and errors are logged to the console.

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

## Troubleshooting

- **FileNotFoundError**: Ensure parent directories exist or use `parents=True` in `Path.mkdir` calls.
- **SettingWithCopyWarning**: Handled in `FileUtils` by using `.copy()` and `.loc` for DataFrame assignments.
- **Parameter Mismatches**: Verify that `window_check_threshold` is used correctly in `v3_pipeline.py` when calling `FileUtils.group_and_apply_sliding_window_calculations`.

For further assistance, contact the repository maintainer or open an issue.