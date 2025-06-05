import pandas as pd
from algo.s1_helper import (
    returns, liquidity_proxy, realized_volatility, trend_duration, volume_volatility_ratio, compute_hmm_regime,
    slope, slope_long, curve_curvature, curve_skewness, log_return, rolling_volatility, delta, ratio,
    btc_nearby_volume, btc_total_volume, roll_activity_ratio
)
import numpy as np
from ta.momentum import ROCIndicator, RSIIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.trend import EMAIndicator, MACD
from pathlib import Path
from utils.path_info import external_data_path


def s1_feature_engineering(prices, start, end):
    # features = simple_features(prices)
    adjusted_prices = pd.read_csv(
        Path(external_data_path, "adjusted_price.csv"),
        parse_dates=True, dayfirst=True, index_col=0
    )
    adjusted_prices = adjusted_prices.loc[start:end]
    volume = pd.read_csv(
        Path(external_data_path, "volume.csv"),
        parse_dates=True, dayfirst=True, index_col=0
    )
    volume = volume.loc[start:end]

    hourly_features = generic_hourly_features(prices)
    daily_features = generic_daily_features(adjusted_prices, volume)
    monthly_features = generic_monthly_features(adjusted_prices)

    hourly_features = hourly_features.shift(1)
    daily_features = daily_features.shift(1)
    monthly_features = monthly_features.shift(1)

    monthly_features = monthly_features.resample("h").last().ffill()
    daily_features = daily_features.resample("h").last().ffill()
    # features = align_daily_to_hourly(hourly_features, daily_features)
    features = pd.concat([hourly_features, daily_features, monthly_features], axis=1)
    features = features.dropna()
    return features


def simple_features(prices):
    features = pd.DataFrame(index=prices.index)
    return_data = prices[["close"]].pct_change()
    features["daily_rtn"] = return_data["close"].fillna(0)
    features["daily_rtn_2hr"] = prices[["close"]].pct_change(2).fillna(0)
    features["daily_rtn_3hr"] = prices[["close"]].pct_change(3).fillna(0)
    features["daily_rtn_4hr"] = prices[["close"]].pct_change(4).fillna(0)
    features["range_over_close_lag_1"] = ((prices["high"] - prices["low"]) / prices["close"]).shift(-1)
    features["range_over_close_lag_1"] = features["range_over_close_lag_1"].fillna(
        features["range_over_close_lag_1"].median()
    )
    return features


def generic_hourly_features(prices):
    price_volume_features = generate_price_volume_feature(prices)
    features_hourly = pd.concat([price_volume_features], axis=1)
    features_hourly['hour_of_day'] = features_hourly.index.hour
    features_hourly['day_of_week'] = features_hourly.index.dayofweek
    features_hourly['is_weekend'] = features_hourly.index.dayofweek.isin([5, 6]).astype(int)
    features_hourly['is_month_end'] = features_hourly.index.is_month_end.astype(int)
    return features_hourly


def generic_daily_features(adjusted_prices, volume):
    curve_features = generate_curve_features(adjusted_prices)
    macro_features = generate_macro_features(adjusted_prices)
    volume_features = generate_volume_features(volume)
    vix = pd.read_csv(Path("data", "external", "VIX Index.csv"), parse_dates=True, index_col=0)
    vix = vix.reindex(curve_features.index)
    features_daily = pd.concat([curve_features, macro_features, volume_features, vix], axis=1)
    return features_daily


def generic_monthly_features(adjusted_prices):
    energy_features = generate_energy_features(adjusted_prices)
    features_monthly = pd.concat([energy_features], axis=1)
    return features_monthly


def generate_curve_features(df):
    features = pd.DataFrame(index=df.index)
    # Slope features
    features["slope_1m"] = slope(df, "BTC1 Curncy", "BTC2 Curncy", normalize=True)
    features["slope_3m"] = slope(df, "BTC1 Curncy", "BTC4 Curncy", normalize=True)
    # features["slope_10m_proxy"] = slope_long(
    #     df, long_cols=["BTC11 Curncy", "BTC10 Curncy", "BTC9 Curncy", "BTC8 Curncy"]
    # )

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
    features["btc1_ret_10d"] = log_return(btc1, period=10)
    features["btc1_ret_30d"] = log_return(btc1, period=30)

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


def generate_price_volume_feature(data):
    data["returns"] = returns(data)
    features = pd.DataFrame(index=data.index)

    # Basic features
    features['volume'] = data['volume']
    features['roc_12'] = ROCIndicator(close=data['close'], window=12).roc()
    features['rsi_24'] = RSIIndicator(close=data['close'], window=24).rsi()

    # Volatility and trend
    features['atr_24'] = AverageTrueRange(high=data['high'], low=data['low'], close=data['close'], window=24).average_true_range()
    bb = BollingerBands(close=data['close'], window=20, window_dev=2)
    features['bb_width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
    features['ema_6'] = EMAIndicator(close=data['close'], window=6).ema_indicator()
    features['ema_24'] = EMAIndicator(close=data['close'], window=24).ema_indicator()
    macd = MACD(close=data['close'])
    features['macd'] = macd.macd()
    features['macd_signal'] = macd.macd_signal()

    # SMAs and crossover
    features['sma_6'] = data['close'].rolling(6).mean()
    features['sma_24'] = data['close'].rolling(24).mean()
    features['cross_signal'] = np.where(features['sma_6'] > features['sma_24'], 1, np.where(features['sma_6'] < features['sma_24'], -1, 0))

    # Serial correlation
    for lag in [1, 2, 5]:
        features[f'autocorr_{lag}'] = data['returns'].rolling(20).corr(data['returns'].shift(lag))

    # Lagged returns & volume
    for lag in [1, 2, 3, 5,]:
        features[f'lag_return_{lag}'] = data['returns'].shift(lag)
    for lag in [1, 2, 3]:
        features[f'lag_volume_{lag}'] = data['volume'].shift(lag)

    features['liquidity_proxy'] = liquidity_proxy(data)
    features['realized_vol_24'] = realized_volatility(data)
    features['trend_duration_12'] = trend_duration(data)
    features['volume_vol_ratio_24'] = volume_volatility_ratio(data)
    features['hmm_state'] = compute_hmm_regime(data, return_col='returns', n_states=3)

    # Cleanup
    features.ffill(inplace=True)
    return features


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
