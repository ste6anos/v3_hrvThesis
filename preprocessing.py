import numpy as np

class HRVPreprocessor:
    """A class for preprocessing RR intervals using various rules.

    This class provides methods to filter RR intervals based on different preprocessing rules,
    such as Malik's rule, Kamarth's rule, Acar's rule, and Karlsson's rule.
    """

    @staticmethod
    def _validate_intervals(intervals):
        """Validate input intervals and handle edge cases.

        Parameters
        ----------
        intervals : array-like
            Array of RR intervals (in milliseconds).

        Returns
        -------
        numpy.ndarray or tuple
            Validated array of intervals, or (empty array, 0) if empty, or (intervals, 0) if single interval.

        Raises
        ------
        ValueError
            If intervals contain non-positive numbers or invalid types.
        """
        intervals = np.array(intervals, dtype=float)
        if len(intervals) == 0:
            return np.array([]), 0
        if len(intervals) == 1:
            return intervals, 0
        if not np.all(intervals > 0):
            raise ValueError("Intervals must be positive numbers.")
        if np.any(np.isnan(intervals)):
            raise ValueError("Intervals cannot contain NaN values.")
        return intervals

    @staticmethod
    def preprocess(intervals, rule="malik", **kwargs):
        """Generic preprocessing function for RR intervals.

        Parameters
        ----------
        intervals : array-like
            Array of RR intervals (in milliseconds).
        rule : str, optional
            The preprocessing rule to apply. Options: 'malik', 'kamarth', 'acar', 'karlsson', 'range_250_2000'.
            Default is 'malik'.
        **kwargs
            Additional arguments for specific rules (e.g., max_diff_percent for Malik's rule,
            max_interval_percentage_diff for Acar's or Karlsson's rule).

        Returns
        -------
        valid_intervals : numpy.ndarray
            Filtered intervals that satisfy the rule.
        intervals_isvalid : numpy.ndarray
            Boolean, True if the value satisly the rule. False for NaNs and not applying the rule.
        false_count : int
            Number of intervals marked as invalid.
        

        Raises
        ------
        ValueError
            If the specified rule is unknown or input is invalid.
        """
        result = HRVPreprocessor._validate_intervals(intervals)
        if isinstance(result, tuple):  # Edge cases: empty or single interval
            return result

        intervals = result  # Validated intervals
        if rule == "malik":
            return HRVPreprocessor.malik_rule(intervals, **kwargs)
        elif rule == "kamarth":
            return HRVPreprocessor.kamarth_rule(intervals)
        elif rule == "acar":
            return HRVPreprocessor.acar_rule(intervals, **kwargs)
        elif rule == "karlsson":
            return HRVPreprocessor.karlsson_rule(intervals, **kwargs)
        elif rule == "range_250_2000":
            return HRVPreprocessor.out_of_range_250ms2000ms(intervals)
        else:
            raise ValueError(f"Unknown preprocessing rule: {rule}")

    @staticmethod
    def malik_rule(intervals, max_diff_percent=20):
        """Apply Malik's rule for preprocessing RR intervals.

        Parameters
        ----------
        intervals : array-like
            Array of RR intervals (in milliseconds).
        max_diff_percent : float, optional
            Maximum allowed percentage difference between consecutive intervals. Default is 20.

        Returns
        -------
        valid_intervals : numpy.ndarray
            Filtered intervals that satisfy the rule.
        false_count : int
            Number of intervals marked as invalid.

        Raises
        ------
        ValueError
            If max_diff_percent is not a non-negative number.
        """
        if not isinstance(max_diff_percent, (int, float)) or max_diff_percent < 0:
            raise ValueError("max_diff_percent must be a non-negative number")

        intervals_isvalid = np.ones(len(intervals), dtype=bool)
        intervals_isvalid[0] = False
        intervals_isvalid[np.isnan(intervals)] = False

        percent_diff = np.abs(np.diff(intervals)) / intervals[:-1] * 100
        diff_valid = percent_diff <= max_diff_percent
        intervals_isvalid[1:] = diff_valid

        valid_intervals = intervals[intervals_isvalid]
        false_count = np.sum(~intervals_isvalid)

        return valid_intervals, intervals_isvalid, false_count

    @staticmethod
    def kamarth_rule(intervals):
        """Apply Kamarth's rule for preprocessing RR intervals.

        Parameters
        ----------
        intervals : array-like
            Array of RR intervals (in milliseconds).

        Returns
        -------
        valid_intervals : numpy.ndarray
            Filtered intervals that satisfy the rule.
        false_count : int
            Number of intervals marked as invalid.
        """
        max_increase_percent = 32.5
        max_decrease_percent = 24.5

        intervals_isvalid = np.ones(len(intervals), dtype=bool)
        intervals_isvalid[0] = False
        intervals_isvalid[np.isnan(intervals)] = False

        percent_diff = np.abs(np.diff(intervals)) / intervals[:-1] * 100
        diff_valid = np.ones(len(intervals) - 1, dtype=bool)
        if len(intervals) > 2:
            diff_valid[:-1] &= (percent_diff[1:] - percent_diff[:-1]) <= max_increase_percent
            diff_valid[1:] &= (percent_diff[:-1] - percent_diff[1:]) <= max_decrease_percent

        intervals_isvalid[1:] = diff_valid
        valid_intervals = intervals[intervals_isvalid]
        false_count = np.sum(~intervals_isvalid)

        return valid_intervals, intervals_isvalid, false_count

    @staticmethod
    def acar_rule(intervals, max_interval_percentage_diff=20):
        """Apply Acar's rule for preprocessing RR intervals.

        Parameters
        ----------
        intervals : array-like
            Array of RR intervals (in milliseconds).
        max_interval_percentage_diff : float, optional
            Maximum allowed percentage difference from the rolling mean. Default is 20.

        Returns
        -------
        valid_intervals : numpy.ndarray
            Filtered intervals that satisfy the rule.
        false_count : int
            Number of intervals marked as invalid.
        """
        if not isinstance(max_interval_percentage_diff, (int, float)) or max_interval_percentage_diff < 0:
            raise ValueError("max_interval_percentage_diff must be a non-negative number")

        intervals_isvalid = np.ones(len(intervals), dtype=bool)
        intervals_isvalid[0] = False
        intervals_isvalid[np.isnan(intervals)] = False

        rolling_mean_last_nine_intervals = np.zeros(len(intervals))
        for i in range(len(intervals)):
            start = max(0, i - 9 + 1)
            rolling_mean_last_nine_intervals[i] = np.mean(intervals[start:i + 1])

        percentage_diff = np.abs((intervals - rolling_mean_last_nine_intervals) / rolling_mean_last_nine_intervals) * 100
        intervals_isvalid = percentage_diff <= max_interval_percentage_diff

        valid_intervals = intervals[intervals_isvalid]
        false_count = np.sum(~intervals_isvalid)

        return valid_intervals, intervals_isvalid, false_count

    @staticmethod
    def karlsson_rule(intervals, max_interval_percentage_diff=0.2):
        """Apply Karlsson's rule for preprocessing RR intervals.

        Parameters
        ----------
        intervals : array-like
            Array of RR intervals (in milliseconds).
        max_interval_percentage_diff : float, optional
            Maximum allowed percentage difference from the mean of neighboring intervals. Default is 0.2.

        Returns
        -------
        valid_intervals : numpy.ndarray
            Filtered intervals that satisfy the rule.
        false_count : int
            Number of intervals marked as invalid.
        """
        if not isinstance(max_interval_percentage_diff, (int, float)) or max_interval_percentage_diff < 0:
            raise ValueError("max_interval_percentage_diff must be a non-negative number")

        intervals_isvalid = np.ones(len(intervals), dtype=bool)
        intervals_isvalid[0] = False
        intervals_isvalid[np.isnan(intervals)] = False

        for i in range(1, len(intervals) - 1):
            if intervals_isvalid[i]:
                mean_neighbors = (intervals[i - 1] + intervals[i + 1]) / 2
                if abs(intervals[i] - mean_neighbors) > max_interval_percentage_diff * mean_neighbors:
                    intervals_isvalid[i] = False

        valid_intervals = intervals[intervals_isvalid]
        false_count = np.sum(~intervals_isvalid)

        return valid_intervals, intervals_isvalid, false_count

    @staticmethod
    def out_of_range_250ms2000ms(intervals):
        """Filter intervals outside the 250ms-2000ms range.

        Parameters
        ----------
        intervals : array-like
            Array of RR intervals (in milliseconds).

        Returns
        -------
        valid_intervals : numpy.ndarray
            Filtered intervals within the range.
        false_count : int
            Number of intervals marked as invalid.
        """
        intervals_isvalid = (intervals >= 250) & (intervals <= 2000)
        valid_intervals = intervals[intervals_isvalid]
        false_count = np.sum(~intervals_isvalid)
        return valid_intervals, intervals_isvalid, false_count