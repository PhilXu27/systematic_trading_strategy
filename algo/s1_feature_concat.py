import pandas as pd
import numpy as np
from s1_feature_engineering import generate_price_volume_feature
from s1_daily_feature import generate_curve_features, generate_macro_features, generate_volume_features, generate_energy_features

import pandas as pd

def align_daily_to_hourly(df_hourly, df_daily):
    """
    Align a daily-frequency DataFrame (df_daily) to the hourly-frequency index of df_hourly.
    
    Parameters:
    - df_hourly: pd.DataFrame
        DataFrame with a DatetimeIndex at hourly frequency.
    - df_daily: pd.DataFrame
        DataFrame with a DatetimeIndex at daily frequency (dates only, no time).
    
    Returns:
    - pd.DataFrame
        Combined DataFrame with df_daily values forward-filled across matching dates in df_hourly's index.
    """

    # Ensure datetime index
    df_hourly.index = pd.to_datetime(df_hourly.index)
    df_daily.index = pd.to_datetime(df_daily.index)

    # Create a mapping from date to daily data row (as Series)
    daily_dict = {d.date(): row for d, row in df_daily.iterrows()}

    # Align by mapping each timestamp to its corresponding daily value
    daily_aligned_list = df_hourly.index.map(
        lambda x: daily_dict.get(x.date(), pd.Series(index=df_daily.columns, dtype='float64'))
    )

    # Convert list of Series to DataFrame
    df_daily_aligned = pd.DataFrame(daily_aligned_list.tolist(), index=df_hourly.index)

    # Concatenate along columns
    df_combined = pd.concat([df_hourly, df_daily_aligned], axis=1)

    return df_combined

if __name__ == "__main__":
    prices = pd.read_csv("data/prices.csv", parse_dates=True, index_col=0)
    hourly_features = generate_price_volume_feature(prices)

    adjusted_prices = pd.read_csv("data/adjusted_price.csv", parse_dates=True, dayfirst=True, index_col=0)
    volume = pd.read_csv("data/volume.csv", parse_dates=True, dayfirst=True, index_col=0)
    volume = volume.reindex(adjusted_prices.index)
    prices_ffill_cols = ["BTC1 Curncy",
                  "BTC2 Curncy",
                  "BTC3 Curncy",
                  "BTC4 Curncy",
                  "DXY Index",
                  "GB3 Govt",
                  "GB6 Govt",
                  "GB12 Govt",
                  "GT2 Govt",
                  "GT5 Govt",
                  "GT10 Govt",
                  "XAU Curncy",
                  "CO1 Comdty"]
    volume_ffill_cols = ["BTC1 Curncy", "BTC2 Curncy", "CO1 Comdty"]
    adjusted_prices[prices_ffill_cols] = adjusted_prices[prices_ffill_cols].ffill(limit=2)
    volume[volume_ffill_cols] = volume[volume_ffill_cols].ffill(limit=2)
    curve_features = generate_curve_features(adjusted_prices)
    macro_features = generate_macro_features(adjusted_prices)

    energy_features = generate_energy_features(adjusted_prices)
    energy_features = energy_features.reindex(curve_features.index)
    energy_features.ffill(inplace=True)

    volume_features = generate_volume_features(volume)
    daily_features = pd.concat([curve_features, macro_features, energy_features, volume_features], axis=1)

    hourly_features = hourly_features.shift(1)
    daily_features = daily_features.shift(1)

    features = align_daily_to_hourly(hourly_features, daily_features)

    print(features)
