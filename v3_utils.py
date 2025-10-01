import os
from datetime import datetime, timedelta
from dateutil import parser
import pandas as pd
import numpy as np
from preprocessing import HRVPreprocessor  # Assumed external dependency
import logging
from tqdm import tqdm
import neurokit2 as nk
import gc

logger = logging.getLogger(__name__)

class FileUtils:
    @staticmethod
    def preprocess_dataframe(df, rule='malik', **kwargs):
        """Apply HRV preprocessing to a DataFrame and filter valid rows.

        Args:
            df (pd.DataFrame): DataFrame with 'time' and 'bbi' columns.
            rule (str): HRV preprocessing rule to apply (default: 'malik').
            **kwargs: Additional arguments for HRVPreprocessor.

        Returns:
            pd.DataFrame: Filtered DataFrame with 'time' and 'bbi' columns.
        """
        if df.empty or not {'time', 'bbi'}.issubset(df.columns):
            logger.warning("Empty DataFrame or missing required columns in preprocess_dataframe.")
            return pd.DataFrame(columns=['time', 'bbi'])

        # Sort by time (avoid resetting index unless necessary)
        df = df.sort_values('time')

        # Extract intervals
        intervals = df['bbi'].values

        # Apply HRV preprocessing with multiple rules
        try:
            _, intervals_isvalid1, _ = HRVPreprocessor.preprocess(intervals, rule='range_250_2000', **kwargs)
            _, intervals_isvalid2, _ = HRVPreprocessor.preprocess(intervals, rule='karlsson', **kwargs)
            _, intervals_isvalid3, _ = HRVPreprocessor.preprocess(intervals, rule='acar', **kwargs)
            _, intervals_isvalid4, _ = HRVPreprocessor.preprocess(intervals, rule=rule, **kwargs)

            # Combine preprocessing results
            intervals_isvalid = intervals_isvalid1 & intervals_isvalid2 & intervals_isvalid3 & intervals_isvalid4

            # Filter valid rows
            df['satisfiesThePreprocessingRule'] = intervals_isvalid
            df = df[df['satisfiesThePreprocessingRule']][['time', 'bbi']].copy()
            return df
        except (ValueError, TypeError) as e:
            logger.error(f"Error in HRV preprocessing: {e}")
            return pd.DataFrame(columns=['time', 'bbi'])

    @staticmethod
    def calc_awake_sleep_dfs(df, file, min_duration='3min'):
        """Calculate sleeping and awake DataFrames from the whole day DataFrame.

        Args:
            df (pd.DataFrame): DataFrame with 'time' and 'bbi' columns.
            file (str): Filename for logging purposes.
            min_duration (str): Minimum duration for a sleep/awake segment (e.g., '3min').

        Returns:
            tuple: (awake_df, sleep_df) - DataFrames with awake and sleep segments.
        """
        if df.empty or not {'time', 'bbi'}.issubset(df.columns):
            logger.warning(f"Empty DataFrame or missing required columns for file {file}. Skipping.")
            return pd.DataFrame(columns=['time', 'bbi']), pd.DataFrame(columns=['time', 'bbi'])

        df = df.copy()

        # Compute threshold once
        threshold = 0.9 * df['bbi'].mean()

        # Handle duplicate indices
        if df.index.duplicated().any():
            logger.info(f"Duplicate indices found in file {file}. Removing duplicates.")
            df = df[~df.index.duplicated(keep='first')]

        # Handle duplicate 'time' values
        if df['time'].duplicated().any():
            logger.warning(f"Duplicate 'time' values found in file {file}. Removing duplicates.")
            df = df[~df['time'].duplicated(keep='first')]

        # Assign sleeping status
        df.loc[:, 'sleeping'] = (df['bbi'] < threshold).map({True: 1, False: -1})

        # Merge short segments
        df = df.reset_index(drop=True)
        change_points = (df['sleeping'] != df['sleeping'].shift()).cumsum()
        for _, group in df.groupby(change_points):
            duration = group['time'].iloc[-1] - group['time'].iloc[0]
            if duration < pd.Timedelta(min_duration):
                prev_idx = group.index[0] - 1
                if prev_idx >= 0:  # Avoid invalid index
                    df.loc[group.index, 'sleeping'] = df.loc[prev_idx, 'sleeping']

        aw = df[df['sleeping'] == -1][['time', 'bbi']].copy()
        sl = df[df['sleeping'] == 1][['time', 'bbi']].copy()

        return aw, sl

    @staticmethod
    def window_check(df, window_duration, threshold):
        """Check if the time window has the necessary amount of BBIs.

        Args:
            df (pd.DataFrame): DataFrame with 'time' and 'bbi' columns.
            window_duration (int): Time duration of the window in minutes.
            threshold (float): Threshold for accepting the window sample.

        Returns:
            bool: True if the window meets the threshold, False otherwise.
        """
        if df.empty or not {'time', 'bbi'}.issubset(df.columns):
            return False

        # Duration of the window in milliseconds
        L = window_duration * 60 * 1000
        # Mean BBI
        M = df['bbi'].mean()
        # Number of intervals
        N = len(df)

        return (N * M) / L > threshold

    @staticmethod
    def group_and_apply_sliding_window_calculations(df, window_offset=100, window_size=3, window_check_threshold=0.5):
        """Apply HRV metrics calculation on sliding windows, 3 frames at a time.

        Args:
            df (pd.DataFrame): DataFrame with 'time' and 'bbi' columns.
            window_offset (int): Offset duration between consecutive windows in minutes.
            window_size (int): Number of frames in a window.
            window_check_threshold (float): Threshold for window_check.

        Returns:
            pd.Series: Mean of HRV measures across valid windows, or empty Series if none.
        """
        if df.empty or not {'time', 'bbi'}.issubset(df.columns):
            logger.warning("Empty DataFrame or missing required columns in sliding window calculations.")
            return pd.Series()

        window_duration = window_offset * window_size
        window_offset = f'{window_offset}min'

        if not pd.api.types.is_datetime64_any_dtype(df['time']):
            df['time'] = pd.to_datetime(df['time'])

        df = df.sort_values('time').reset_index(drop=True)
        df['time_bin'] = df['time'].dt.floor(window_offset)

        grouped = df.groupby('time_bin')
        if grouped.ngroups == 0:
            logger.info("No valid time bins for sliding window calculations.")
            return pd.Series()

        bin_times = grouped['time'].first().reset_index(drop=True)
        results = []

        for i in tqdm(range(len(bin_times) - window_size + 1), desc="Processing sliding windows", leave=False):
            start_time = bin_times.iloc[i]
            end_time = start_time + pd.Timedelta(minutes=window_duration)

            window_df = df[(df['time'] >= start_time) & (df['time'] < end_time)][['time', 'bbi']]

            if FileUtils.window_check(window_df, window_duration, window_check_threshold):
                t = window_df['time'].to_numpy()
                x = window_df['bbi'].to_numpy()

                RRI_dict = {'RRI': x, 'RRI_time': t}

                try:
                    hrv_td = nk.hrv_time(RRI_dict, show=False)[["HRV_RMSSD", "HRV_SDNN", "HRV_HTI"]]
                    hrv_fd = nk.hrv_frequency(RRI_dict, show=False)[
                        ["HRV_ULF", "HRV_VLF", "HRV_LF", "HRV_HF", "HRV_VHF", "HRV_TP",
                         "HRV_LFHF", "HRV_LFn", "HRV_HFn", "HRV_LnHF"]
                    ]

                    temp_df = pd.concat([hrv_td, hrv_fd], axis=1)
                    results.append(temp_df)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Error computing HRV metrics for window starting at {start_time}: {e}")
                    continue

                del window_df, hrv_td, hrv_fd, temp_df
                gc.collect()

        if results:
            results_df = pd.concat(results, ignore_index=True)
            return results_df.mean()
        else:
            logger.info("No valid windows for HRV calculations.")

            return pd.Series()

    @staticmethod
    def extract_crp_data_from_csv(file_path):
        """
        Extracts and normalizes CRP data from a CSV file.
        
        Converts mg/dL to mg/L by multiplying by 10.
        
        Args:
            file_path (str): Path to the CSV file.
        
        Returns:
            pd.DataFrame: Cleaned CRP DataFrame with 'patientid' and 'DOC_PAT_CRP_LEVEL'.
        """
        df = pd.read_csv(file_path, index_col=0)
        columns = ['patientid', 'DOC_PAT_CRP_DATE', 'DOC_PAT_CRP_LEVEL', 'DOC_PAT_CRP_UNIT']
        crp_df = df[columns].copy()
        crp_df['DOC_PAT_CRP_LEVEL'] = crp_df['DOC_PAT_CRP_LEVEL'].where(
            df['DOC_PAT_CRP_UNIT'] != 'mg/dL', df['DOC_PAT_CRP_LEVEL'] * 10
        )
        crp_df = crp_df[["patientid", "DOC_PAT_CRP_LEVEL"]]
        return crp_df

    @staticmethod
    def hypothesis_test_binary_target(df1, df2):
        """
        Performs group comparison tests (t-test or Mann-Whitney U) for each numeric metric
        between two DataFrames (e.g., true vs. false groups).
        
        Uses normality (Shapiro-Wilk) to choose parametric/non-parametric test.
        
        Args:
            df1, df2 (pd.DataFrame): Groups to compare.
        
        Returns:
            pd.DataFrame: Sorted results by p-value.
        """
        results = {}
        metrics = df1.select_dtypes(include='number').columns.intersection(df2.select_dtypes(include='number').columns)
        
        for metric in metrics:
            data1 = df1[metric].dropna()
            data2 = df2[metric].dropna()
            if len(data1) < 3 or len(data2) < 3:
                continue  # Skip small samples
            
            _, p1 = shapiro(data1)
            _, p2 = shapiro(data2)
            
            if p1 > 0.05 and p2 > 0.05:
                t_stat, p_value = ttest_ind(data1, data2, equal_var=False)
                method = "t_test"
            else:
                u_stat, p_value = mannwhitneyu(data1, data2, alternative="two-sided")
                method = "u_test"
            
            results[metric] = {"method": method, "p-value": p_value, "p1": p1, "p2": p2}
        
        if not results:
            return pd.DataFrame()
        
        results_df = pd.DataFrame(results).T
        results_df = results_df.sort_values("p-value")
        return results_df
    

    @staticmethod
    def correlation_test(df):
        """
        Performs correlation tests (Pearson or Spearman) between numeric columns (except last)
        and the last column (target) of the DataFrame.
        
        Drops NaNs in target upfront; per-metric, drops rows with metric NaNs.
        Uses normality to choose method.
        
        Args:
            df (pd.DataFrame): DataFrame with metrics + target as last column.
        
        Returns:
            pd.DataFrame: Sorted results by p-value.
        """
        if df.empty or len(df.columns) < 2:
            return pd.DataFrame()
        
        target_name = df.columns[-1]
        
        # Drop rows with NaN in target upfront
        df = df.dropna(subset=[target_name])
        if df.empty or len(df) < 3:
            return pd.DataFrame()
        
        # Numeric metrics (exclude target)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        metric_cols = [col for col in numeric_cols if col != target_name]
        
        if not metric_cols:
            return pd.DataFrame()
        
        results = {}
        for metric in metric_cols:
            temp_df = df.dropna(subset=[metric])
            if len(temp_df) < 3:
                continue  # Skip small samples
            
            metric_clean = temp_df[metric]
            target_aligned = temp_df[target_name]
            _, p_metric = shapiro(metric_clean)
            _, p_target = shapiro(target_aligned)
            
            if p_metric > 0.05 and p_target > 0.05:
                corr_stat, p_value = pearsonr(metric_clean, target_aligned)
                method = "pearson"
            else:
                corr_result = spearmanr(metric_clean, target_aligned)
                corr_stat = corr_result.correlation
                p_value = corr_result.pvalue
                method = "spearman"
            
            results[metric] = {
                "method": method,
                "correlation": corr_stat,
                "p-value": p_value,
                "p_metric_normal": p_metric,
                "p_target_normal": p_target,
                "n_samples": len(metric_clean)
            }
        
        if not results:
            return pd.DataFrame()
        
        results_df = pd.DataFrame(results).T
        results_df = results_df.sort_values("p-value")
        return results_df
