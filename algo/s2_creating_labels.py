import pandas as pd
import numpy as np
import statsmodels.api as sm1
from utils.path_info import label_data_path
from pathlib import Path


def s2_load_labels(file_name):
    labels = pd.read_csv(Path(label_data_path,  f"{file_name}.csv"), index_col=0, parse_dates=True)
    return labels


def s2_creating_labels(prices, is_save_labels=True, file_name="default"):
    labels = trend_labels(prices["close"], [6, 12, 24], look_forward=True)
    if is_save_labels:
        labels.to_csv(Path(label_data_path, f"{file_name}.csv"))
    return labels


def tValLinR(close):
    """
    Calculate the t-value and coefficients of the slope from a linear regression of the time series.

    Parameters:
    - close (pd.Series): A pandas series of closing prices.

    Returns:
    - tuple: (t-value of the slope coefficient, coefficients of the regression)
    """
    x = np.ones((close.shape[0], 2))
    x[:, 1] = np.arange(close.shape[0])
    ols = sm1.OLS(close, x).fit()
    return ols.tvalues[1], ols.params


def trend_labels(price_series, observation_span, look_forward=True):
    """
    Generate labels for segments of a time series based on the trend (slope) over a specified observation span.

    Parameters:
    - price_series (pd.Series): A pandas series of prices or any numerical values, indexed by dates or integers.
    - observation_span list: A list defining the observation periods.
    - look_forward (bool): If True, the function analyzes forward trends. If False, it analyzes backward trends.

    Returns:
    - pd.DataFrame: A DataFrame with columns ['t1', 'tVal', 'bin', 'windowSize'].
    """
    # Initialize a DataFrame to store the results with the same index as the price_series
    out = pd.DataFrame(index=price_series.index, columns=['t1', 'tVal', 'bin', 'windowSize'])

    # Define the range of horizons (windows) to test
    hrzns = observation_span

    # Loop through each index in the price series
    for idx in price_series.index:
        # Dictionary to store the t-values for each horizon
        tval_dict = {}

        # Get the location (integer index) of the current index in the series
        iloc0 = price_series.index.get_loc(idx)

        # Skip the index if there isn't enough data to look forward or backward
        if look_forward and iloc0 > len(price_series) - observation_span[1]:
            continue
        if not look_forward and iloc0 < observation_span[1]:
            continue

        # Loop through each horizon in the specified range
        for hrzn in hrzns:
            if look_forward:
                # Define the window for forward-looking analysis
                dt1 = idx  # Start date
                dt2 = min(iloc0 + hrzn, len(price_series) - 1)  # End date
                dt2 = price_series.index[dt2]
            else:
                # Define the window for backward-looking analysis
                dt1 = max(iloc0 - hrzn, 0)  # Start date
                dt1 = price_series.index[dt1]
                dt2 = idx  # End date

            # Extract the segment of the series for the current window
            df1 = price_series.loc[dt1:dt2]

            # Calculate and store the t-value for the linear trend of this segment
            tval_dict[hrzn], _ = tValLinR(df1.values)

        # Find the horizon with the highest absolute t-value
        max_hrzn = max(tval_dict, key=lambda x: abs(tval_dict[x]))

        # Determine the end date of the window for the maximum t-value
        if look_forward:
            max_dt1 = min(iloc0 + max_hrzn, len(price_series) - 1)
            max_dt1 = price_series.index[max_dt1]
        else:
            max_dt1 = max(iloc0 - max_hrzn, 0)
            max_dt1 = price_series.index[max_dt1]

        # Store the results in the DataFrame
        out.loc[idx, ['t1', 'tVal', 'bin', 'windowSize']] = max_dt1, tval_dict[max_hrzn], np.sign(
            tval_dict[max_hrzn]), max_hrzn

    # Convert 't1' to datetime if the index is of datetime type
    if isinstance(price_series.index, pd.DatetimeIndex):
        out['t1'] = pd.to_datetime(out['t1'])

    # Convert 'bin' to a numeric type, downcasting to the smallest signed integer
    out['bin'] = pd.to_numeric(out['bin'], downcast='signed')

    # Handle extreme t-values by setting a maximum threshold
    tValueVariance = out['tVal'].values.var()
    tMax = min(20, tValueVariance)
    out.loc[out['tVal'] > tMax, 'tVal'] = tMax
    out.loc[out['tVal'] < -tMax, 'tVal'] = -tMax

    # Drop rows with NaN values in the 'bin' column and return the DataFrame
    return out.dropna(subset=['bin'])
