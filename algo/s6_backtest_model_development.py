import pandas as pd
from pathlib import Path
from utils.path_info import results_path, signals_data_path
from configs.MODEL_CONFIG import MODEL_CONFIG
from xgboost import XGBClassifier
from algo.s3_model_development import time_series_cv
from sklearn.ensemble import RandomForestClassifier
from multiprocessing import Pool
import multiprocessing


def backtest_data_prepare(labels, features):
    bins = labels[["bin"]]
    bins = bins.applymap(lambda x: 1.0 if x == 1.0 else 0.0)
    bins.columns = ["label"]
    valid_time_index = bins.index.tolist()
    features = features.loc[valid_time_index]
    data_all = pd.concat([bins, features], axis=1)
    return data_all


def s6_load_backtest_signals(**kwargs):
    backtest_test_models = kwargs.get("backtest_test_models")
    all_predictions = {}
    for model in backtest_test_models:
        signals = pd.read_csv(Path(signals_data_path, model + ".csv"), index_col=0, parse_dates=True)
        all_predictions[model] = signals
    return all_predictions


def s6_backtest_model_development(
        labels, features,
        test_start, test_end,
        retrain_frequency, rebalance_frequency,
        training_mode,
        validation_percentage,
        **kwargs
):
    data_all = backtest_data_prepare(labels, features)
    ################
    # Check kwargs #
    ################
    valid_model_class = ["XGBClassifier", "RandomForestClassifier"]
    backtest_test_models = kwargs.get("backtest_test_models")
    assert all([i in MODEL_CONFIG.keys() for i in backtest_test_models])
    assert all([MODEL_CONFIG.get(i).get("model_class") in valid_model_class for i in backtest_test_models])

    retrain_timestamps = pd.date_range(start=test_start, end=test_end, freq=retrain_frequency)
    rebalance_timestamps = pd.date_range(start=test_start, end=test_end, freq=rebalance_frequency)
    # Make sure every time in retrain also in rebalance, if it is, do nothing, otherwise, add it to rebalance.
    combined_timestamps = pd.Index(sorted(set(rebalance_timestamps).union(set(retrain_timestamps))))
    rebalance_timestamps = combined_timestamps

    if training_mode == "expanding_window":
        all_predictions = expanding_window_backtest(
            data_all, retrain_timestamps, rebalance_timestamps, validation_percentage, backtest_test_models
        )
    elif training_mode == "rolling_window":
        window_size = kwargs.get("window_size", 30 * 24 * 12)
        warmup_size = kwargs.get("warmup_size", 30 * 24)
        all_predictions = rolling_window_backtest(
            data_all, retrain_timestamps, rebalance_timestamps, validation_percentage, backtest_test_models,
            window_size, warmup_size
        )
    elif training_mode == "parallel_expanding_window":
        all_predictions = parallel_backtest(
            training_mode, data_all, retrain_timestamps, rebalance_timestamps, validation_percentage, backtest_test_models
        )
    elif training_mode == "parallel_rolling_window":
        window_size = kwargs.get("window_size", 30 * 24 * 12)
        warmup_size = kwargs.get("warmup_size", 30 * 24)
        all_predictions = parallel_backtest(
            training_mode, data_all, retrain_timestamps, rebalance_timestamps, validation_percentage, backtest_test_models,
            window_size, warmup_size
        )
    else:
        raise ValueError
    for model in backtest_test_models:
        signals = all_predictions[model]
        signals.to_csv(Path(signals_data_path, f"{model}.csv"))
    return all_predictions


def run_batch(model_name, model_config, batch_index, data_all, validation_percentage, rebalance_ts_batch):
    results = []
    curr_model = None

    for i in range(len(rebalance_ts_batch) - 1):
        rebalance_ts = rebalance_ts_batch[i]
        next_rebalance_ts = rebalance_ts_batch[i + 1]
        train_data = data_all[data_all.index <= rebalance_ts]
        test_data = data_all[(data_all.index > rebalance_ts) & (data_all.index <= next_rebalance_ts)]

        if test_data.empty:
            continue

        X_train, y_train = train_data.iloc[:, 1:], train_data.iloc[:, 0]
        X_test, y_test = test_data.iloc[:, 1:], test_data.iloc[:, 0]

        if i == 0:  # Only retrain at start of batch
            val_size = int(len(train_data) * validation_percentage)
            val_data = train_data.iloc[-val_size:]
            hyper_train_data = train_data.iloc[:-val_size]

            X_val, y_val = val_data.iloc[:, 1:], val_data.iloc[:, 0]
            X_hyper_train, y_hyper_train = hyper_train_data.iloc[:, 1:], hyper_train_data.iloc[:, 0]

            curr_model, y_pred = eval(model_config["model_func"])(
                eval(model_config["model_class"]),
                model_config["base_params"],
                model_config["param_grid"],
                X_train, y_train,
                X_hyper_train, y_hyper_train,
                X_val, y_val,
                X_test, y_test
            )
        else:
            curr_model.fit(X_train, y_train)
            y_pred = curr_model.predict(X_test)

        curr_pred = pd.DataFrame(index=y_test.index)
        curr_pred["true"] = y_test
        curr_pred["pred"] = y_pred
        results.append(curr_pred)

    result_df = pd.concat(results).sort_index()
    result_df["model"] = model_name
    return model_name, batch_index, result_df


def run_expanding_batch(model_name, model_config, batch_index, data_all, validation_percentage, rebalance_ts_batch):
    results = []
    curr_model = None

    for i in range(len(rebalance_ts_batch) - 1):
        rebalance_ts = rebalance_ts_batch[i]
        next_rebalance_ts = rebalance_ts_batch[i + 1]
        train_data = data_all[data_all.index <= rebalance_ts]
        test_data = data_all[(data_all.index > rebalance_ts) & (data_all.index <= next_rebalance_ts)]

        if test_data.empty:
            continue

        X_train, y_train = train_data.iloc[:, 1:], train_data.iloc[:, 0]
        X_test, y_test = test_data.iloc[:, 1:], test_data.iloc[:, 0]

        if i == 0:  # Only retrain at start of batch
            val_size = int(len(train_data) * validation_percentage)
            val_data = train_data.iloc[-val_size:]
            hyper_train_data = train_data.iloc[:-val_size]

            X_val, y_val = val_data.iloc[:, 1:], val_data.iloc[:, 0]
            X_hyper_train, y_hyper_train = hyper_train_data.iloc[:, 1:], hyper_train_data.iloc[:, 0]

            curr_model, y_pred = eval(model_config["model_func"])(
                eval(model_config["model_class"]),
                model_config["base_params"],
                model_config["param_grid"],
                X_train, y_train,
                X_hyper_train, y_hyper_train,
                X_val, y_val,
                X_test, y_test
            )
        else:
            curr_model.fit(X_train, y_train)
            y_pred = curr_model.predict(X_test)

        curr_pred = pd.DataFrame(index=y_test.index)
        curr_pred["true"] = y_test
        curr_pred["pred"] = y_pred
        results.append(curr_pred)

    result_df = pd.concat(results).sort_index()
    result_df["model"] = model_name
    return model_name, batch_index, result_df


def run_rolling_batch(
        model_name, model_config, batch_index, data_all, validation_percentage, rebalance_ts_batch,
        window_size, warmup_size
):
    results = []
    curr_model = None

    for i in range(len(rebalance_ts_batch) - 1):
        rebalance_ts = rebalance_ts_batch[i]
        next_rebalance_ts = rebalance_ts_batch[i + 1]

        rebalance_idx = data_all.index.get_loc(rebalance_ts)
        if rebalance_idx < warmup_size:
            continue
        if rebalance_idx <= window_size:
            train_data = data_all.iloc[:rebalance_idx + 1]
        else:
            train_data = data_all.iloc[rebalance_idx - window_size + 1:rebalance_idx + 1]
        test_data = data_all[(data_all.index > rebalance_ts) & (data_all.index <= next_rebalance_ts)]
        if test_data.empty:
            continue

        X_train, y_train = train_data.iloc[:, 1:], train_data.iloc[:, 0]
        X_test, y_test = test_data.iloc[:, 1:], test_data.iloc[:, 0]

        if i == 0:  # Only retrain at start of batch
            val_size = int(len(train_data) * validation_percentage)
            val_data = train_data.iloc[-val_size:]
            hyper_train_data = train_data.iloc[:-val_size]

            X_val, y_val = val_data.iloc[:, 1:], val_data.iloc[:, 0]
            X_hyper_train, y_hyper_train = hyper_train_data.iloc[:, 1:], hyper_train_data.iloc[:, 0]

            curr_model, y_pred = eval(model_config["model_func"])(
                eval(model_config["model_class"]),
                model_config["base_params"],
                model_config["param_grid"],
                X_train, y_train,
                X_hyper_train, y_hyper_train,
                X_val, y_val,
                X_test, y_test
            )
        else:
            curr_model.fit(X_train, y_train)
            y_pred = curr_model.predict(X_test)
        curr_pred = pd.DataFrame(index=y_test.index)
        curr_pred["true"] = y_test
        curr_pred["pred"] = y_pred
        results.append(curr_pred)

    result_df = pd.concat(results).sort_index()
    result_df["model"] = model_name
    return model_name, batch_index, result_df


def parallel_backtest(
        training_mode, data_all, retrain_timestamps, rebalance_timestamps, validation_percentage, test_models,
        window_size=None, warmup_size=None
):
    all_predictions = {}

    if training_mode == "parallel_expanding_window":
        run_func = run_expanding_batch
    elif training_mode == "parallel_rolling_window":
        run_func = run_rolling_batch  # todo
    else:
        raise ValueError

    for model in test_models:
        model_config = MODEL_CONFIG[model]
        batch_ranges = []

        for i in range(len(retrain_timestamps) - 1):
            start = retrain_timestamps[i]
            end = retrain_timestamps[i + 1]
            rebalance_ts_batch = [ts for ts in rebalance_timestamps if start <= ts <= end]
            if len(rebalance_ts_batch) < 2:
                raise ValueError("Check for the empty batches.")
            if training_mode == "parallel_expanding_window":
                batch_ranges.append((model, model_config, i, data_all, validation_percentage, rebalance_ts_batch))
            elif training_mode == "parallel_rolling_window":
                batch_ranges.append((model, model_config, i, data_all, validation_percentage, rebalance_ts_batch, window_size, warmup_size))
            else:
                raise ValueError

        num_cpus = multiprocessing.cpu_count()
        with Pool(processes=min(len(batch_ranges), num_cpus)) as pool:
            results = pool.starmap(run_func, batch_ranges)

        model_dfs = [_[2] for _ in sorted(results, key=lambda x: x[1])]
        full_model_df = pd.concat(model_dfs).sort_index()
        full_model_df.to_csv(Path(results_path, "test", f"{model}.csv"))
        all_predictions[model] = full_model_df

    return all_predictions

def expanding_window_backtest(
        data_all, retrain_timestamps, rebalance_timestamps,
        validation_percentage, test_models
):
    all_predictions = {}
    for model in test_models:
        model_setting = MODEL_CONFIG[model]

        assert "model_func" in model_setting.keys()
        assert "model_class" in model_setting.keys()
        assert "base_params" in model_setting.keys()
        assert "param_grid" in model_setting.keys()
        model_func = model_setting.get("model_func")
        model_class = eval(model_setting.get("model_class"))
        base_params = model_setting.get("base_params")
        param_grid = model_setting.get("param_grid")

        pred_df = pd.DataFrame()
        curr_model = None
        print("Running model {}".format(model))
        for i in range(len(rebalance_timestamps) - 1):

            rebalance_ts = rebalance_timestamps[i]
            next_rebalance_ts = rebalance_timestamps[i + 1]
            print(f"We rebalance on {rebalance_ts}")

            train_data = data_all[data_all.index <= rebalance_ts]
            test_data = data_all[(data_all.index > rebalance_ts) & (data_all.index <= next_rebalance_ts)]

            X_train, y_train = train_data.iloc[:, 1:], train_data.iloc[:, 0]
            X_test, y_test = test_data.iloc[:, 1:], test_data.iloc[:, 0]
            if rebalance_ts in retrain_timestamps or not curr_model:
                print(f"We retrain on {rebalance_ts}")
                # init or we reach the timestamp that we need to do retrain
                validation_size = int(len(train_data) * validation_percentage)
                val_data = train_data.iloc[-validation_size:]
                hyper_train_data = train_data.iloc[:-validation_size]
                X_hyper_train, y_hyper_train = hyper_train_data.iloc[:, 1:], hyper_train_data.iloc[:, 0]
                X_val, y_val = val_data.iloc[:, 1:], val_data.iloc[:, 0]

                #####################################################
                # Hyper param tuning, model training and prediction #
                #####################################################
                curr_model, y_pred = eval(model_func)(
                    model_class, base_params, param_grid,
                    X_train, y_train, X_hyper_train, y_hyper_train,
                    X_val, y_val, X_test, y_test
                )
            else:
                ##################################################
                # Model training with same params and prediction #
                ##################################################
                curr_model.fit(X_train, y_train)
                y_pred = curr_model.predict(X_test)

            curr_pred = pd.DataFrame(index=y_test.index, columns=["pred", "true"])
            curr_pred["true"] = y_test
            curr_pred["pred"] = y_pred
            pred_df = pd.concat([pred_df, curr_pred], axis=0)
        all_predictions[model] = pred_df
    return all_predictions

def rolling_window_backtest(
        data_all, retrain_timestamps, rebalance_timestamps, validation_percentage, test_models, window_size, warmup_size
):

    all_predictions = {}
    for model in test_models:
        model_setting = MODEL_CONFIG[model]

        assert "model_func" in model_setting.keys()
        assert "model_class" in model_setting.keys()
        assert "base_params" in model_setting.keys()
        assert "param_grid" in model_setting.keys()
        model_func = model_setting.get("model_func")
        model_class = eval(model_setting.get("model_class"))
        base_params = model_setting.get("base_params")
        param_grid = model_setting.get("param_grid")

        pred_df = pd.DataFrame()
        curr_model = None
        print("Running model {}".format(model))
        for i in range(len(rebalance_timestamps) - 1):

            rebalance_ts = rebalance_timestamps[i]
            next_rebalance_ts = rebalance_timestamps[i + 1]
            print(f"We rebalance on {rebalance_ts}")

            rebalance_idx = data_all.index.get_loc(rebalance_ts)

            if rebalance_idx < warmup_size:
                print(f"Skipping prediction at {rebalance_ts} due to warmup period")
                continue

            if rebalance_idx <= window_size:
                # Expanding window during warmup phase (t ∈ (warmup_size, window_size])
                train_data = data_all.iloc[:rebalance_idx + 1]
            else:
                # Fixed rolling window for t > window_size
                train_data = data_all.iloc[rebalance_idx - window_size + 1:rebalance_idx + 1]

            test_data = data_all[(data_all.index > rebalance_ts) & (data_all.index <= next_rebalance_ts)]

            X_train, y_train = train_data.iloc[:, 1:], train_data.iloc[:, 0]
            X_test, y_test = test_data.iloc[:, 1:], test_data.iloc[:, 0]
            if rebalance_ts in retrain_timestamps or not curr_model:
                print(f"We retrain on {rebalance_ts}")
                # init or we reach the timestamp that we need to do retrain
                validation_size = int(len(train_data) * validation_percentage)
                val_data = train_data.iloc[-validation_size:]
                hyper_train_data = train_data.iloc[:-validation_size]
                X_hyper_train, y_hyper_train = hyper_train_data.iloc[:, 1:], hyper_train_data.iloc[:, 0]
                X_val, y_val = val_data.iloc[:, 1:], val_data.iloc[:, 0]

                #####################################################
                # Hyper param tuning, model training and prediction #
                #####################################################
                curr_model, y_pred = eval(model_func)(
                    model_class, base_params, param_grid,
                    X_train, y_train, X_hyper_train, y_hyper_train,
                    X_val, y_val, X_test, y_test
                )
            else:
                ##################################################
                # Model training with same params and prediction #
                ##################################################
                curr_model.fit(X_train, y_train)
                y_pred = curr_model.predict(X_test)

            curr_pred = pd.DataFrame(index=y_test.index, columns=["pred", "true"])
            curr_pred["true"] = y_test
            curr_pred["pred"] = y_pred
            pred_df = pd.concat([pred_df, curr_pred], axis=0)
        all_predictions[model] = pred_df
    return all_predictions


def simple_random_forest(
        model_class, base_params, param_grid,
        X_train, y_train, X_hyper_train, y_hyper_train,
        X_val, y_val, X_test, y_test
):
    best_rf, best_rf_score, best_params = time_series_cv(
        model_class=model_class, base_params=base_params, param_grid=param_grid,
        X_train=X_hyper_train, y_train=y_hyper_train, X_val=X_val, y_val=y_val
    )
    rf_model = RandomForestClassifier(**best_params)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    return rf_model, y_pred

def simple_xgb_boost(
        model_class, base_params, param_grid,
        X_train, y_train, X_hyper_train, y_hyper_train,
        X_val, y_val, X_test, y_test
):
    best_rf, best_rf_score, best_params = time_series_cv(
        model_class=model_class, base_params=base_params, param_grid=param_grid,
        X_train=X_hyper_train, y_train=y_hyper_train, X_val=X_val, y_val=y_val
    )
    xgb_model = XGBClassifier(**best_params)
    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)
    return xgb_model, y_pred

