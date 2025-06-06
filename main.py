import pandas as pd

from algo.s0_data_preparing import s0_data_prepare
from algo.s1_feature_engineering import s1_feature_engineering, s1_load_features
from algo.s25_pre_run import s25_pre_run
from algo.s2_creating_labels import s2_creating_labels, s2_load_labels
from algo.s3_model_development import s3_model_development
from algo.s4_feature_important_analysis import s4_feature_important_analysis, s4_cluster_level_importance_analysis
from algo.s5_model_evaluation import s5_model_evaluation
from algo.s6_backtest_model_development import s6_backtest_model_development, s6_load_backtest_signals
from algo.s7_backtest_portfolio_formation import s7_backtest_portfolio_formation
from algo.s8_portfolio_analysis import s8_portfolio_analysis


def main(
        training_mode="parallel_rolling_window",
        is_generate_features=False, is_generate_labels=True,
        is_backtest_only=False, is_generate_backtest_signals=True,
        **kwargs
):
    general_start = "2018-01-01" #  "2018-01-01", "2020-01-01"
    warmup_end = "2018-03-31"
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
    labels = forward_looking_labels
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
        cluster_feature_important_results = s4_cluster_level_importance_analysis(models, experiment_data_dict)
        #######################
        # 5. Model Evaluation #
        #######################
        evaluation_results = s5_model_evaluation(models, experiment_data_dict)
    else:
        ############################################
        # 6. Backtest & Backtest Model Development #
        ############################################
        rebalance_frequency = "120h"
        retrain_frequency = "720h"
        assert training_mode in [
            "expanding_window", "rolling_window",
            "parallel_expanding_window", "parallel_rolling_window"
        ]
        backtest_save_prefix = f"rebalance_{rebalance_frequency}_retrain_{retrain_frequency}_mode_{training_mode}"

        if is_generate_backtest_signals:
            backtest_model_predictions = s6_backtest_model_development(
                labels, features,
                test_start, general_end,
                retrain_frequency, rebalance_frequency,
                training_mode,
                validation_percentage,
                backtest_save_prefix,
                **kwargs
            )
        else:
            backtest_model_predictions = s6_load_backtest_signals(backtest_save_prefix, **kwargs)
        ###################################
        # 7. Backtest Portfolio Formation #
        ###################################
        portfolio_info, portfolio_values = s7_backtest_portfolio_formation(
            prices, backtest_model_predictions, forward_looking_labels, backtest_save_prefix
        )
        ###################################
        # 7. Backtest Portfolio Analytics #
        ###################################
        s8_portfolio_analysis(portfolio_values, backtest_save_prefix)

    return


if __name__ == '__main__':
    curr_test_models = ["xgb_boost", "random_forest", "gradient_boost", "lightgbm"]

    main(
        is_generate_features=True,
        is_generate_labels=True,
        is_generate_backtest_signals=True,
        is_backtest_only=True,  # If you want to run backtest, turn it on; if you want to tune the model, turn if off.
        training_mode="parallel_rolling_window",  # "parallel_expanding_window", "parallel_rolling_window"
        **{
            "curr_label_file": "labels_6_12_18_24_48",
            "backtest_test_models": curr_test_models,
        }
    )
