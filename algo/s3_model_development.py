import lightgbm as lgb
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, log_loss
from sklearn.model_selection import ParameterGrid
from sklearn.svm import SVC

from configs.GLOBAL_CONFIG import GLOBAL_RANDOM_STATE


def s3_model_development(experiment_data_dict):
    print("S3 Model Development: One Time Prediction, Starts")
    X_train, y_train = experiment_data_dict.get("X_train"), experiment_data_dict.get("y_train")
    X_hyper_train, y_hyper_train = experiment_data_dict.get("X_hyper_train"), experiment_data_dict.get("y_hyper_train")
    X_val, y_val = experiment_data_dict.get("X_val"), experiment_data_dict.get("y_val")
    X_test, y_test = experiment_data_dict.get("X_test"), experiment_data_dict.get("y_test")

    rf_model = random_forest_one_time(X_train, y_train, X_hyper_train, y_hyper_train, X_val, y_val, X_test, y_test)
    gb_model = gradient_boosting_one_time(X_train, y_train, X_hyper_train, y_hyper_train, X_val, y_val, X_test, y_test)
    xgb_model = xgboost_one_time(X_train, y_train, X_hyper_train, y_hyper_train, X_val, y_val, X_test, y_test)
    lgbm_model = lightgbm_one_time(X_train, y_train, X_hyper_train, y_hyper_train, X_val, y_val, X_test, y_test)

    models = {
        'Random Forest': rf_model,
        'Gradient Boosting Model': gb_model,
        'XGB Model': xgb_model,
        'LGBM Model': lgbm_model,
    }
    print("S3 Model Development: One Time Prediction, Ends")
    return models


def random_forest_one_time(X_train, y_train, X_hyper_train, y_hyper_train, X_val, y_val, X_test, y_test):
    print("Training Random Forest...")
    param_grid = {
        'max_depth': [20],
        'n_estimators': [100],
        'min_samples_split': [5],
    }
    # param_grid = {
    #     'max_depth': [None, 10, 20],
    #     'n_estimators': [100, 200, 500],
    #     'min_samples_split': [2, 5],
    # }
    base_params = {
        "random_state": GLOBAL_RANDOM_STATE, "min_samples_leaf": 2, "max_features": "sqrt",
    }
    best_rf, best_rf_score, best_params = time_series_cv(
        model_class=RandomForestClassifier, base_params=base_params, param_grid=param_grid,
        X_train=X_hyper_train, y_train=y_hyper_train, X_val=X_val, y_val=y_val
    )
    # Predict on test set
    rf_model = RandomForestClassifier(**best_params)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)

    test_acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {test_acc:.4f}")
    return rf_model


def gradient_boosting_one_time(X_train, y_train, X_hyper_train, y_hyper_train, X_val, y_val, X_test, y_test):
    print("Training Gradient Boosting...")
    param_grid = {
        'max_depth': [20],
        'n_estimators': [100],
        'min_samples_split': [5],
    }
    # param_grid = {
    #     'max_depth': [None, 10, 20],
    #     'n_estimators': [100, 200, 500],
    #     'min_samples_split': [2, 5],
    # }
    base_params = {
        "random_state": GLOBAL_RANDOM_STATE,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
    }

    best_rf, best_rf_score, best_params = time_series_cv(model_class=GradientBoostingClassifier,
                                                         base_params=base_params,
                                                         param_grid=param_grid, X_train=X_hyper_train,
                                                         y_train=y_hyper_train, X_val=X_val, y_val=y_val)
    # Predict on test set
    gb_model = GradientBoostingClassifier(**best_params)
    gb_model.fit(X_train, y_train)
    y_pred = gb_model.predict(X_test)

    test_acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {test_acc:.4f}")
    return gb_model


def xgboost_one_time(X_train, y_train, X_hyper_train, y_hyper_train, X_val, y_val, X_test, y_test):
    print("Training XGBoost...")
    # param_grid = {
    #     'learning_rate': [0.01, 0.05, 0.1],
    #     'n_estimators': [100, 200],
    #     'max_depth': [3, 4, 6],
    #     'min_child_weight': [1, 5],
    # }
    param_grid = {
        'learning_rate': [0.05],
        'n_estimators': [200],
        'max_depth': [4],
        'min_child_weight': [5],
    }
    base_params = {
        'random_state': GLOBAL_RANDOM_STATE,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
    }
    best_rf, best_rf_score, best_params = time_series_cv(model_class=xgb.XGBClassifier, base_params=base_params,
                                                         param_grid=param_grid, X_train=X_hyper_train,
                                                         y_train=y_hyper_train, X_val=X_val, y_val=y_val)
    # Predict on test set
    xgb_model = xgb.XGBClassifier(**best_params)
    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)

    test_acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {test_acc:.4f}")
    return xgb_model


def lightgbm_one_time(X_train, y_train, X_hyper_train, y_hyper_train, X_val, y_val, X_test, y_test):
    print("Training LightGBM...")
    # param_grid = {
    #     'learning_rate': [0.01, 0.05, 0.1],
    #     'n_estimators': [100, 200],
    #     'max_depth': [3, 4, 6],
    #     'min_child_weight': [1, 5],
    # }
    param_grid = {
        'learning_rate': [0.01],
        'n_estimators': [100],
        'max_depth': [3],
        'min_child_weight': [5],
    }
    base_params = {
        'random_state': GLOBAL_RANDOM_STATE,
        'verbose': -1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'objective': 'binary',
    }
    best_rf, best_rf_score, best_params = time_series_cv(model_class=lgb.LGBMClassifier, base_params=base_params,
                                                         param_grid=param_grid, X_train=X_hyper_train,
                                                         y_train=y_hyper_train, X_val=X_val, y_val=y_val)
    # Predict on test set
    lgbm_model = lgb.LGBMClassifier(**best_params)
    lgbm_model.fit(X_train, y_train)
    y_pred = lgbm_model.predict(X_test)

    test_acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {test_acc:.4f}")
    return lgbm_model


def svm_one_time(X_train, y_train, X_hyper_train, y_hyper_train, X_val, y_val, X_test, y_test):
    print("Training SVM...")
    # param_grid = {
    #     'C': [0.1, 1, 10],
    #     'gamma': ['scale', 'auto', 0.01, 0.1],
    # }
    param_grid = {
        'C': [0.1],
        'gamma': [100],
    }
    base_params = {
        'random_state': GLOBAL_RANDOM_STATE,
        'probability': True,
        'kernel': 'rbf',  # you can choose one from ['precomputed', 'linear', 'rbf', 'poly', 'sigmoid']
    }
    best_rf, best_rf_score, best_params = time_series_cv(model_class=SVC, base_params=base_params,
                                                         param_grid=param_grid, X_train=X_hyper_train,
                                                         y_train=y_hyper_train, X_val=X_val, y_val=y_val)
    # Predict on test set
    svm_model = SVC(**best_params)
    svm_model.fit(X_train, y_train)
    y_pred = svm_model.predict(X_test)

    test_acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {test_acc:.4f}")
    return svm_model


def time_series_cv(model_class, base_params, param_grid, X_train, y_train, X_val, y_val, principle="score"):
    assert principle in ["score", "AUC", "F1"]

    best_model = None
    best_score = -np.inf
    best_params = None
    for params in ParameterGrid(param_grid):
        combined_params = {**base_params, **params}
        print(f"Working on {combined_params}")
        model = model_class(**combined_params)
        model.fit(X_train, y_train)
        if principle == "score":
            score = model.score(X_val, y_val)  # or AUC, F1, etc.
        elif principle == "accuracy":
            # by default the model.score is accuracy, but in case the user intentionally type in accuracy.
            y_pred = model.predict(X_val)
            score = accuracy_score(y_val, y_pred)
        elif principle == "AUC":
            y_proba = model.predict_proba(X_val)[:, 1]
            score = roc_auc_score(y_val, y_proba)
        elif principle == "F1":
            y_pred = model.predict(X_val)
            score = f1_score(y_val, y_pred)
        elif principle == "log_loss":
            y_proba = model.predict_proba(X_val)
            score = -log_loss(y_val, y_proba)
        else:
            raise ValueError(f"Unknown principle: {principle}")

        if score > best_score:
            best_score = score
            best_model = model
            best_params = combined_params
    print(f"Best Score is: {best_score}, with params: {best_params}")
    return best_model, best_score, best_params
