import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


###################
# Hourly Features #
###################
def returns(df):
    return (df['close'] - df['open']) / df['open']


def liquidity_proxy(df):
    return (df['high'] - df['low']) / df['volume'].replace(0, np.nan)


def realized_volatility(df, window=24):
    return df['returns'].rolling(window=window).std()


def trend_duration(df, ema_window=12):
    ema = df['close'].ewm(span=ema_window, adjust=False).mean()
    trend = df['close'] > ema
    trend_duration = trend.astype(int).groupby((trend != trend.shift()).cumsum()).cumsum()
    return trend_duration.where(trend, -trend_duration)


def volume_volatility_ratio(df, vol_window=24):
    realized_vol = df['returns'].rolling(window=vol_window).std()
    return df['volume'] / realized_vol.replace(0, np.nan)


def compute_hmm_regime(df, return_col='returns', n_states=3, random_state=42):
    """
    Fits a Gaussian HMM to the return series and returns a state label series.

    Parameters:
        df (pd.DataFrame): DataFrame containing the return column.
        return_col (str): Name of the column to use for HMM (default = 'returns').
        n_states (int): Number of hidden states (default = 3).
        random_state (int): Random seed for reproducibility.

    Returns:
        pd.Series: Series of HMM state labels aligned with df index.
    """
    returns = df[return_col].dropna().values.reshape(-1, 1)
    model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=100, random_state=random_state)
    model.fit(returns)
    state_series = pd.Series(np.nan, index=df.index)
    state_series.loc[df[return_col].notna()] = model.predict(returns)
    return state_series.astype('Int64')  # Nullable integer type for NaN compatibility


##################
# Daily Features #
##################
def rolling_volatility(series, window=5):
    return series.pct_change().rolling(window).std()


def log_return(series, period=1):
    return np.log(series / series.shift(period))


def curve_skewness(df):
    """
    Computes curve skewness as (BTC4 - BTC2) - (BTC2 - BTC1).
    Positive: back end steeper - possible carry.
    Negative: front end steep  - stress or front-loaded expectations.
    """
    front_slope = df["BTC2 Curncy"] - df["BTC1 Curncy"]
    back_slope = df["BTC4 Curncy"] - df["BTC2 Curncy"]
    return back_slope - front_slope


def curve_curvature(df, front_col, mid_col, back_col, normalize=True):
    curvature = (df[front_col] + df[back_col]) / 2 - df[mid_col]
    if normalize:
        curvature = curvature / df[front_col]
    return curvature


def slope_long(df, long_cols=None, near_col="BTC1 Curncy"):
    """
    Compute slope proxy = (long_future - BTC1) / BTC1,
    where long_future is BTC11, or BTC10, or BTC9 — whichever is available first row-wise.
    """
    # Start with NaNs
    long_future = pd.Series(index=df.index, dtype=float)

    # Fill long_future with BTC11 if available, else BTC10, else BTC9
    for col in long_cols:
        long_future = long_future.combine_first(df[col])

    # Calculate slope using the best available long_future
    slope = (long_future - df[near_col]) / df[near_col]

    return slope


def slope(df, near_col, far_col, normalize=True):
    slope = df[far_col] - df[near_col]
    if normalize:
        slope = slope / df[near_col]
    return slope


def delta(series):
    """Computes first-order difference."""
    return series.diff()


def ratio(df, numerator_col="BTC1 Curncy", denominator_col="XAU Curncy"):
    """Computes ratio."""
    return df[numerator_col] / df[denominator_col]


def btc_nearby_volume(df, btc1_col="BTC1 Curncy", btc2_col="BTC2 Curncy"):
    """
    Compute BTC nearby volume by summing BTC1 and BTC2.
    Smooths out rollover-driven volume spikes.
    """
    return df[btc1_col] + df[btc2_col]


def btc_total_volume(df, btc_prefix="BTC", contracts=11):
    """
    Compute total BTC volume across available front contracts (e.g., BTC1 to BTC6).
    """
    btc_cols = [f"{btc_prefix}{i} Curncy" for i in range(1, contracts + 1) if f"{btc_prefix}{i} Curncy" in df.columns]
    return df[btc_cols].sum(axis=1)


def roll_activity_ratio(df, btc1_col="BTC1 Curncy", btc2_col="BTC2 Curncy"):
    """
    Compute the ratio of BTC2 volume to nearby volume (BTC1 + BTC2).
    Peaks near expiry as traders roll from BTC1 to BTC2.
    """
    nearby = df[btc1_col] + df[btc2_col]
    return df[btc2_col] / nearby.replace(0, np.nan)
