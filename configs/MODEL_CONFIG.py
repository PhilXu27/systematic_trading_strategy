from configs.GLOBAL_CONFIG import GLOBAL_RANDOM_STATE

MODEL_CONFIG = {
    "simple_xgb_boost": {
        "model_func": "simple_xgb_boost",
        "model_class": "XGBClassifier",
        "base_params": {
            "random_state": GLOBAL_RANDOM_STATE,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
        },
        "param_grid": {
            "learning_rate": [0.01],
            "n_estimators": [100, 200],
            "max_depth": [3, 6],
            "min_child_weight": [5]
        }
    },
    "simple_random_forest": {
        "model_func": "simple_random_forest",
        "model_class": "RandomForestClassifier",
        "base_params": {
            "random_state": GLOBAL_RANDOM_STATE,
            "min_samples_leaf": 2,
            "max_features": "sqrt"
        },
        "param_grid": {
            "max_depth": [None, 10, 20],
            "n_estimators": [100, 200],
            "min_samples_split": [2, 5]
        }
    }
}
