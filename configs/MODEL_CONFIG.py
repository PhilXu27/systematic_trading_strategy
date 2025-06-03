from configs.GLOBAL_CONFIG import GLOBAL_RANDOM_STATE

MODEL_CONFIG = {
    "xgboost_simple": {
        "model_class": "XGBClassifier",
        "base_params": {
            "random_state": GLOBAL_RANDOM_STATE,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "use_label_encoder": False
        },
        "param_grid": {
            "learning_rate": [0.01, 0.05],
            "n_estimators": [100, 200],
            "max_depth": [3, 6],
            "min_child_weight": [5]
        }
    }
}
