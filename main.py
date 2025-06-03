from algo.s0_data_preparing import s0_data_prepare
from algo.s1_feature_engineering import s1_feature_engineering
from algo.s2_creating_labels import s2_creating_labels, s2_load_labels
from algo.s25_pre_run import s25_pre_run
from algo.s3_model_development import s3_model_development
from algo.s4_feature_important_analysis import s4_feature_important_analysis
from algo.s5_model_evaluation import s5_model_evaluation
from algo.s6_backtest import s6_backtest
from pathlib import Path
from utils.path_info import data_path
import pandas as pd


def main(is_local_features=False, is_local_labels=True, backtest_only=False, **kwargs):
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
    if not is_local_features:
        features = s1_feature_engineering(prices, general_start, general_end)
    else:
        # features = s1_load_features()
        raise NotImplementedError

    feature_start, feature_end = (
        features.index[0].strftime("%Y-%m-%d %H:%M:%S"),
        features.index[-1].strftime("%Y-%m-%d %H:%M:%S")
    )
    #######################
    # 2.  Creating Labels #
    #######################
    if not is_local_labels:
        labels = s2_creating_labels(prices, **kwargs)
    else:
        curr_label_file = kwargs.get("curr_label_file")
        labels = s2_load_labels(curr_label_file)
    label_start, label_end = (
        labels.index[0].strftime("%Y-%m-%d %H:%M:%S"),
        labels.index[-1].strftime("%Y-%m-%d %H:%M:%S")
    )

    experiment_start = max(label_start, feature_start, warmup_end)
    experiment_end = min(label_end, feature_end)
    valid_timestamp = pd.date_range(start=experiment_start, end=experiment_end, freq="h")

    labels = labels.loc[valid_timestamp]
    features = features.loc[valid_timestamp]
    #########################
    # 2.5. Train Test Split #
    #########################
    data_all, experiment_data_dict = s25_pre_run(labels, features, test_start, general_end, validation_percentage)

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

    ############################################
    # 6. Backtest & Backtest Model Development #
    ############################################
    pred_frequency = "6h"
    training_mode = "one_time_prediction"
    # s6_backtest()
    return


if __name__ == '__main__':
    main(
        **{
            "curr_label_file": "labels_6_12_24",
            "backtest_test_models": ["xgboost_simple"]
        }
    )
