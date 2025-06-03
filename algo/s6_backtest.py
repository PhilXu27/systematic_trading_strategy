import pandas as pd
from configs.GLOBAL_CONFIG import GLOBAL_RANDOM_STATE
from configs.MODEL_CONFIG import MODEL_CONFIG
from xgboost import XGBClassifier


def backtest_data_prepare(labels, features):
    bins = labels[["bin"]]
    bins = bins.applymap(lambda x: 1.0 if x == 1.0 else 0.0)
    bins.columns = ["label"]
    valid_time_index = bins.index.tolist()
    features = features.loc[valid_time_index]
    data_all = pd.concat([bins, features], axis=1)
    return data_all

def s6_backtest(labels, features, test_start, test_end, pred_frequency, training_mode, validation_percentage, **kwargs):
    data_all = backtest_data_prepare(labels, features)
    ################
    # Check kwargs #
    ################
    valid_model_class = ["XGBClassifier", "RandomForestClassifier"]
    backtest_test_models = kwargs.get("backtest_test_models")
    assert [i in MODEL_CONFIG.keys() for i in backtest_test_models]
    assert [i.get("model_class") in valid_model_class for i in backtest_test_models]

    rebalance_timestamps = pd.date_range(start=test_start, end=test_end, freq=pred_frequency)
    if training_mode == "expanding_window":
        expanding_window_backtest(data_all, rebalance_timestamps, validation_percentage, backtest_test_models)
    elif training_mode == "rolling_window":
        rolling_window_backtest()
        raise NotImplementedError
    return


def expanding_window_backtest(data_all, rebalance_timestamps, validation_percentage, test_models):
    for model in test_models:
        model_setting = MODEL_CONFIG[model]
        print("Running model {}".format(model))
        for i in range(len(rebalance_timestamps) - 1):
            rebalance_ts = rebalance_timestamps[i]
            next_rebalance_ts = rebalance_timestamps[i + 1]
            train_data = data_all[data_all.index <= rebalance_ts]
            validation_size = int(len(train_data) * validation_percentage)
            val_data = train_data.iloc[-validation_size:]
            hyper_train_data = train_data.iloc[:-validation_size]
            test_data = data_all[(data_all.index > rebalance_ts) & (data_all.index <= next_rebalance_ts)]

            X_train, y_train = train_data.iloc[:, 1:], train_data.iloc[:, 0]
            X_hyper_train, y_hyper_train = hyper_train_data.iloc[:, 1:], hyper_train_data.iloc[:, 0]
            X_val, y_val = val_data.iloc[:, 1:], val_data.iloc[:, 0]
            X_test, y_test = test_data.iloc[:, 1:], test_data.iloc[:, 0]
            ######################
            # Hyper param tuning #
            ######################

    return

def rolling_window_backtest(**kwargs):
    window_size = kwargs.get("window_size")
    return


def simple_random_forest(X_train, y_train, X_test, y_test, **kwargs):

    return
