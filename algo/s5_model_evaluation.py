from sklearn.metrics import roc_auc_score, accuracy_score, \
    precision_score, recall_score, f1_score


def s5_model_evaluation(models, experiment_data_dict):
    X_train, y_train = experiment_data_dict["X_train"], experiment_data_dict["y_train"]
    X_test, y_test = experiment_data_dict["X_test"], experiment_data_dict["y_test"]

    evaluation_results = []
    for model_name, model in models.items():
        print(f"\nEvaluating {model_name}...")
        metrics = evaluate_model(model, X_train, y_train, X_test, y_test, model_name)
        evaluation_results.append(metrics)
        print(f"{model_name} - Test AUC: {metrics['Test AUC']:.4f}")
    return evaluation_results


def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    """Comprehensive model evaluation"""
    # Make predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    train_proba = model.predict_proba(X_train)[:, 1]
    test_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    metrics = {
        'Model': model_name,
        'Train Accuracy': accuracy_score(y_train, train_pred),
        'Test Accuracy': accuracy_score(y_test, test_pred),
        'Train Precision': precision_score(y_train, train_pred),
        'Test Precision': precision_score(y_test, test_pred),
        'Train Recall': recall_score(y_train, train_pred),
        'Test Recall': recall_score(y_test, test_pred),
        'Train F1': f1_score(y_train, train_pred),
        'Test F1': f1_score(y_test, test_pred),
        'Train AUC': roc_auc_score(y_train, train_proba),
        'Test AUC': roc_auc_score(y_test, test_proba),
        'test_proba': test_proba
    }

    # Store predictions for ROC curve

    return metrics
