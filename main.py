from algo.s0_data_preparing import s0_data_prepare
from algo.s1_feature_engineering import s1_feature_engineering, s1_load_features
from algo.s2_creating_labels import s2_creating_labels, s2_load_labels
from algo.s25_pre_run import s25_pre_run
from algo.s3_model_development import s3_model_development
from algo.s4_feature_important_analysis import s4_feature_important_analysis
from algo.s5_model_evaluation import s5_model_evaluation
from algo.s6_backtest_model_development import s6_backtest_model_development, s6_load_backtest_signals
from algo.s7_backtest_portfolio_formation import s7_backtest_portfolio_formation
from pathlib import Path
from utils.path_info import data_path
import pandas as pd


def main(
        is_generate_features=False, is_generate_labels=True, is_backtest_only=False, is_generate_backtest_signals=True,
        **kwargs
):
    general_start = "2018-01-01" #  "2018-01-01", "2020-01-01"
    warmup_end = "2018-03-31"
    train_end = "2020-12-31"
    test_start = "2021-01-01"
    general_end = "2021-12-31"  # "2021-12-31"
    validation_percentage = 0.2
    #####################
    # 0. Data Preparing #
    #####################
    prices = s0_data_prepare()
    prices = prices.loc[general_start: general_end]
    ##########################
    # 1. Feature Engineering #
    ##########################
    if is_generate_features:
        features = s1_feature_engineering(prices, general_start, general_end)
    else:
        features = s1_load_features()

    feature_start, feature_end = (
        features.index[0].strftime("%Y-%m-%d %H:%M:%S"),
        features.index[-1].strftime("%Y-%m-%d %H:%M:%S")
    )
    #######################
    # 2.  Creating Labels #
    #######################
    if is_generate_labels:
        curr_save_file = kwargs.get("curr_label_file", "default")
        is_save_labels = kwargs.get("is_save_labels", True)
        labels, forward_looking_labels = s2_creating_labels(
            prices, is_save_labels=is_save_labels, file_name=curr_save_file
        )
    else:
        curr_label_file = kwargs.get("curr_label_file", "default")
        labels, forward_looking_labels = s2_load_labels(curr_label_file)
    label_start, label_end = (
        labels.index[0].strftime("%Y-%m-%d %H:%M:%S"),
        labels.index[-1].strftime("%Y-%m-%d %H:%M:%S")
    )

    experiment_start = max(label_start, feature_start, warmup_end)
    experiment_end = min(label_end, feature_end)
    valid_timestamp = pd.date_range(start=experiment_start, end=experiment_end, freq="h")

    labels = labels.loc[valid_timestamp]
    forward_looking_labels = forward_looking_labels.loc[valid_timestamp]
    features = features.loc[valid_timestamp]
    #########################
    # 2.5. Train Test Split #
    #########################
    data_all, experiment_data_dict = s25_pre_run(labels, features, test_start, general_end, validation_percentage)

    if not is_backtest_only:
        ########################
        # 3. Model Development #
        ########################
        models = s3_model_development(experiment_data_dict)

        #################################
        # 4. Feature Important Analysis #
        #################################
        feature_important_results = s4_feature_important_analysis(models, experiment_data_dict)

        #######################
        # 5. Model Evaluation #
        #######################
        evaluation_results = s5_model_evaluation(models, experiment_data_dict)
    else:
        ############################################
        # 6. Backtest & Backtest Model Development #
        ############################################
        rebalance_frequency = "24h"
        retrain_frequency = "144h"
        training_mode = "parallel_expanding_window"
        assert training_mode in ["expanding_window", "rolling_window", "parallel_expanding_window", ""]
        if is_generate_backtest_signals:
            backtest_model_predictions = s6_backtest_model_development(
                labels, features,
                test_start, general_end,
                retrain_frequency, rebalance_frequency,
                training_mode,
                validation_percentage,
                **kwargs
            )
        else:
            backtest_model_predictions = s6_load_backtest_signals(**kwargs)
        ############################################
        # 7. Backtest & Backtest Model Development #
        ############################################

        s7_backtest_portfolio_formation(prices, backtest_model_predictions, forward_looking_labels)

    return


if __name__ == '__main__':
    main(
        is_generate_features=False,
        is_generate_labels=False,
        is_generate_backtest_signals=False,
        is_backtest_only=True,
        **{
            "curr_label_file": "labels_6_12_24",
            "backtest_test_models": ["xgboost_simple"]
        }
    )
