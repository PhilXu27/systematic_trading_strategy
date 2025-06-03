import pandas as pd
import numpy as np
from ta.momentum import ROCIndicator, RSIIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.trend import EMAIndicator, MACD
from hmmlearn.hmm import GaussianHMM

def s1_feature_engineering(prices):
    features = simple_features(prices)
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

def generate_price_volume_feature(data):
    data['returns'] = returns(data)
    features = data.copy()

    # Basic features
    features['roc_12'] = ROCIndicator(close=data['close'], window=12).roc()
    features['rsi_14'] = RSIIndicator(close=data['close'], window=14).rsi()

    # Volatility and trend
    features['atr_14'] = AverageTrueRange(high=data['high'], low=data['low'], close=data['close'], window=14).average_true_range()
    bb = BollingerBands(close=data['close'], window=20, window_dev=2)
    features['bb_width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
    features['ema_12'] = EMAIndicator(close=data['close'], window=12).ema_indicator()
    features['ema_26'] = EMAIndicator(close=data['close'], window=26).ema_indicator()
    macd = MACD(close=data['close'])
    features['macd'] = macd.macd()
    features['macd_signal'] = macd.macd_signal()

    # SMAs and crossover
    features['sma_50'] = data['close'].rolling(50).mean()
    features['sma_200'] = data['close'].rolling(200).mean()
    features['cross_signal'] = np.where(features['sma_50'] > features['sma_200'], 1, np.where(features['sma_50'] < features['sma_200'], -1, 0))

    # Serial correlation
    for lag in [1, 2, 5]:
        features[f'autocorr_{lag}'] = data['returns'].rolling(20).corr(data['returns'].shift(lag))

    # Lagged returns & volume
    for lag in [1, 2, 3, 5]:
        features[f'lag_return_{lag}'] = data['returns'].shift(lag)
    for lag in [1, 2, 3]:
        features[f'lag_volume_{lag}'] = data['volume'].shift(lag)

    features['liquidity_proxy'] = liquidity_proxy(data)
    features['realized_vol_24'] = realized_volatility(data)
    features['trend_duration_12'] = trend_duration(data)
    features['volume_vol_ratio_24'] = volume_volatility_ratio(data)
    features['hmm_state'] = compute_hmm_regime(data, return_col='returns', n_states=3)

    # Final cleanup
    features.ffill(inplace=True)

    return features

if __name__ == "__main__":
    # Load your dataset
    data = pd.read_csv("data/prices.csv", parse_dates=True, index_col=0)
    features = generate_price_volume_feature(data)
    print(features)
    # features.to_csv("data/basic_features.csv")