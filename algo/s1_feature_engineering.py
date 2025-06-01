import pandas as pd


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

