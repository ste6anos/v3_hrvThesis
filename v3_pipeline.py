import os
from pathlib import Path
import pandas as pd
from dateutil import parser
from datetime import datetime, timedelta
import logging
from tqdm import tqdm
import gc
from v3_utils import FileUtils  
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Define paths using pathlib for cross-platform compatibility
all_patient_bbi_path = Path(r'C:\Users\spbtu\Documents\dataset_psath\bbi')
clinical_data_path = Path(r'C:\Users\spbtu\Documents\dataset_psath\bbi_metadata\clinical_d32_T0_v2.csv')

# Load clinical data
clinical_data_df = pd.read_csv(clinical_data_path)
d32_patients = clinical_data_df['patientid'].tolist()
all_patient_bbi_list = [p for p in all_patient_bbi_path.iterdir() if p.is_dir()]

# Keep only patients with clinical data and BBI recordings
patients = sorted(list(set(d32_patients) & {p.name for p in all_patient_bbi_list}))
patients = [p for p in patients if p not in ['p2070001']]  # Exclude problematic cases

# Filter clinical DataFrame
clinical_data_df_v2 = clinical_data_df[clinical_data_df['patientid'].isin(patients)]



window_offset = 500
window_size = 3

# Create output directories
output_dir_aw = Path(r"patient_metrics_windowduration1500m/aw")
output_dir_sl = Path(r"patient_metrics_windowduration1500m/sl")
output_dir_aw.mkdir(parents=True, exist_ok=True)  # Add parents=True
output_dir_sl.mkdir(parents=True, exist_ok=True)  # Add parents=True

# Process patients
for pnt in tqdm(patients, desc="Processing patients", leave=True):
    # Initialize lists for awake and sleep data
    aw_dfs = []
    sl_dfs = []

    # Get register date
    register_date_series = clinical_data_df_v2[clinical_data_df_v2['patientid'] == pnt]['final_register_date']
    if register_date_series.empty:
        logger.info(f"No register date found for patient {pnt}. Skipping.")
        continue

    try:
        register_date_dt = parser.parse(register_date_series.values[0])
    except (ValueError, TypeError) as e:
        logger.info(f"Invalid register date for patient {pnt}: {e}. Skipping.")
        continue

    patient_bbi_path = all_patient_bbi_path / pnt
    patient_bbi_files = [f for f in patient_bbi_path.glob('*.parquet')]

    for file in tqdm(patient_bbi_files, desc=f"Processing files for patient {pnt}", leave=False):
        try:
            file_date = file.stem[:10]
            file_date_dt = parser.parse(file_date)
        except ValueError as e:
            logger.warning(f"Invalid date format in file {file.name} for patient {pnt}: {e}. Skipping.")
            continue

        if file_date_dt <= register_date_dt + timedelta(weeks=2):
            try:
                df = pd.read_parquet(file)
                df = FileUtils.preprocess_dataframe(df, rule='malik')
                aw, sl = FileUtils.calc_awake_sleep_dfs(df, file.name, min_duration='3min')

                if not aw.empty:
                    aw_dfs.append(aw)
                if not sl.empty:
                    sl_dfs.append(sl)
            except (ValueError, TypeError, OSError) as e:
                logger.error(f"Error processing file {file.name} for patient {pnt}: {e}")
                continue

    # Concatenate DataFrames if non-empty
    pnt_aw_df = pd.concat(aw_dfs, ignore_index=True) if aw_dfs else pd.DataFrame(columns=['time', 'bbi'])
    pnt_sl_df = pd.concat(sl_dfs, ignore_index=True) if sl_dfs else pd.DataFrame(columns=['time', 'bbi'])

    # Awake intervals processing
    aw_output_path = output_dir_aw / f"{pnt}.parquet"
    if not aw_output_path.exists():
        try:
            pnt_aw_metrics = FileUtils.group_and_apply_sliding_window_calculations(pnt_aw_df, window_offset, window_size, window_check_threshold=0.5)
            if not pnt_aw_metrics.empty:
                pnt_aw_metrics = pnt_aw_metrics.to_frame().T
                pnt_aw_metrics['patientid'] = pnt
                pnt_aw_metrics['state'] = 'awake'
                pnt_aw_metrics.to_parquet(aw_output_path, compression='snappy')
            else:
                logger.info(f"No awake metrics for patient {pnt}.")
        except (ValueError, TypeError, OSError) as e:
            logger.error(f"Failed to calculate sliding window metrics for patient {pnt} (awake): {e}")

    # Sleep intervals processing
    sl_output_path = output_dir_sl / f"{pnt}.parquet"
    if not sl_output_path.exists():
        try:
            #lower threshold because the limits of implementation 
            pnt_sl_metrics = FileUtils.group_and_apply_sliding_window_calculations(pnt_sl_df, window_offset, window_size, window_check_threshold=0.25)
            if not pnt_sl_metrics.empty:
                pnt_sl_metrics = pnt_sl_metrics.to_frame().T
                pnt_sl_metrics['patientid'] = pnt
                pnt_sl_metrics['state'] = 'sleeping'
                pnt_sl_metrics.to_parquet(sl_output_path, compression='snappy')
            else:
                logger.info(f"No sleep metrics for patient {pnt}.")
        except (ValueError, TypeError, OSError) as e:
            logger.error(f"Failed to calculate sliding window metrics for patient {pnt} (sleeping): {e}")

    # Clean up memory
    del pnt_aw_df, pnt_sl_df, aw_dfs, sl_dfs
    gc.collect()