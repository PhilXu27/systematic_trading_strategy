import numpy as np
import pandas as pd

def slope(df, near_col, far_col, normalize=True):
    slope = df[far_col] - df[near_col]
    if normalize:
        slope = slope / df[near_col]
    return slope

def curve_curvature(df, front_col, mid_col, back_col, normalize=True):
    curvature = (df[front_col] + df[back_col]) / 2 - df[mid_col]
    if normalize:
        curvature = curvature / df[front_col]
    return curvature

def log_return(series, period=1):
    return np.log(series / series.shift(period))

def rolling_volatility(series, window=5):
    return series.pct_change().rolling(window).std()

def curve_skewness(df):
    """
    Computes curve skewness as (BTC4 - BTC2) - (BTC2 - BTC1).
    Positive: back end steeper - possible carry.
    Negative: front end steep  - stress or front-loaded expectations.
    """
    front_slope = df["BTC2 Curncy"] - df["BTC1 Curncy"]
    back_slope = df["BTC4 Curncy"] - df["BTC2 Curncy"]
    return back_slope - front_slope

def slope_long(df, long_cols=["BTC11 Curncy", "BTC10 Curncy", "BTC9 Curncy", "BTC8 Curncy"], near_col="BTC1 Curncy"):
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

def generate_curve_features(df):
    features = pd.DataFrame(index=df.index)

    # Slope features
    features["slope_1m"] = slope(df, "BTC1 Curncy", "BTC2 Curncy", normalize=True)
    features["slope_3m"] = slope(df, "BTC1 Curncy", "BTC4 Curncy", normalize=True)
    features["slope_10m_proxy"] = slope_long(df)

    # Curve curvature
    features["curve_curvature"] = curve_curvature(df, "BTC1 Curncy", "BTC2 Curncy", "BTC3 Curncy", normalize=False)
    features["curve_curvature_norm"] = curve_curvature(df, "BTC1 Curncy", "BTC2 Curncy", "BTC3 Curncy", normalize=True)

    # Curve skewness
    features["curve_skewness"] = curve_skewness(df)

    # BTC1 price and momentum
    btc1 = df["BTC1 Curncy"]
    features["btc1_price"] = btc1
    features["btc1_ret_1d"] = log_return(btc1, period=1)
    features["btc1_ret_3d"] = log_return(btc1, period=3)
    features["btc1_ret_5d"] = log_return(btc1, period=5)

    # BTC1 volatility (rolling 5-day)
    features["btc1_vol_5d"] = rolling_volatility(btc1, window=5)
    return features

def generate_macro_features(df):
    """Computes a comprehensive macro feature set."""
    features = pd.DataFrame(index=df.index)

    # DXY features
    features["delta_dxy"] = delta(df["DXY Index"])
    features["dxy_ret_1d"] = log_return(df["DXY Index"], period=1)

    # Yield curve slope (US 10Y - 2Y)
    features["yield_slope"] = slope(df, "GT2 Govt", "GT10 Govt", normalize=False)

    # Long-term yield change
    features["delta_gt10"] = delta(df["GT10 Govt"])

    # Gold and oil deltas
    features["delta_gold"] = delta(df["XAU Curncy"])
    features["delta_oil"] = delta(df["CO1 Comdty"])

    # BTC/gold ratio
    features["btc_gold_ratio"] = ratio(df, "BTC1 Curncy", "XAU Curncy")

    # Macro volatility (example: DXY and GT10)
    features["vol_dxy_5d"] = rolling_volatility(df["DXY Index"], window=5)
    features["vol_gt10_5d"] = rolling_volatility(df["GT10 Govt"], window=5)

    # Short rate slope (1Y - 3M)
    features["short_yield_slope"] = slope(df, "GB3 Govt", "GB12 Govt", normalize=False)

    # Volatility of short-term rates (3M)
    features["vol_gb3_5d"] = rolling_volatility(df["GB3 Govt"], window=5)

    return features

def generate_energy_features(df):
    df = df[["BTC1 Curncy", "ST27IA Index", "ST27AU Index", "ST27RA Index"]].dropna()
    features = pd.DataFrame(index=df.index)

    # Returns the ratio of industrial electricity price to BTC price.
    features["mining_cost_proxy"] = ratio(df, "ST27IA Index", "BTC1 Curncy")

    # Returns log return of general electricity prices as an inflation proxy.
    features["inflation_pressure"] = log_return(df["ST27AU Index"], period=1)

    # Returns the spread between residential and industrial electricity prices.
    features["consumer_stress"] = slope(df, "ST27IA Index", "ST27RA Index", normalize=False)

    return features

def generate_volume_features(df):
    features = pd.DataFrame(index=df.index)

    features["btc_nearby_volume"] = btc_nearby_volume(df)
    features["btc_total_volume"] = btc_total_volume(df, contracts=11)
    features["roll_activity_ratio"] = roll_activity_ratio(df)
    features['oil_volume'] = df['CO1 Comdty']

    return features

if __name__ == "__main__":
    from utils.path_info import external_data_path
    from pathlib import Path
    adjusted_prices = pd.read_csv(Path(external_data_path, "adjusted_price.csv"), parse_dates=True, dayfirst=True, index_col=0)
    volume = pd.read_csv(Path(external_data_path, "volume.csv"), parse_dates=True, dayfirst=True, index_col=0)
    volume = volume.reindex(adjusted_prices.index)
    prices_ffill_cols = [
        "BTC1 Curncy", "BTC2 Curncy", "BTC3 Curncy", "BTC4 Curncy", "DXY Index",
        "GB3 Govt", "GB6 Govt", "GB12 Govt", "GT2 Govt", "GT5 Govt", "GT10 Govt",
        "XAU Curncy", "CO1 Comdty"
    ]

    volume_ffill_cols = ["BTC1 Curncy", "BTC2 Curncy", "CO1 Comdty"]

    adjusted_prices[prices_ffill_cols] = adjusted_prices[prices_ffill_cols].ffill(limit=2)
    volume[volume_ffill_cols] = volume[volume_ffill_cols].ffill(limit=2)
    curve_features = generate_curve_features(adjusted_prices)
    macro_features = generate_macro_features(adjusted_prices)

    energy_features = generate_energy_features(adjusted_prices)
    energy_features = energy_features.reindex(curve_features.index)
    energy_features.ffill(inplace=True)

    volume_features = generate_volume_features(volume)
    features = pd.concat([curve_features, macro_features, energy_features, volume_features], axis=1)
