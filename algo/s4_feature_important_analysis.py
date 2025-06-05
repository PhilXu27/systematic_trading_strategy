import pandas as pd
import matplotlib.pyplot as plt
from configs.GLOBAL_CONFIG import GLOBAL_RANDOM_STATE
from sklearn.inspection import permutation_importance

def s4_feature_important_analysis(models, experiment_data_dict):
    print("S4 Feature Important Analysis, Starts")
    X_train, y_train = experiment_data_dict["X_train"], experiment_data_dict["y_train"]
    X_test, y_test = experiment_data_dict["X_test"], experiment_data_dict["y_test"]

    feature_important_results = {}
    for model_name, model in models.items():
        fi_metrics = {
            "mdi": plot_mdi_importance(model, model_name, X_train),
            "pfi": plot_permutation_importance(model, model_name, X_train, y_train, X_test, y_test)
        }
        feature_important_results[model_name] = fi_metrics
    print("S4 Feature Important Analysis, Ends")
    return feature_important_results

def plot_mdi_importance(model, model_name, X_train, n_features=20):
    """Plot MDI feature importance for tree-based models"""
    mdi_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    # Plot top features
    plt.figure(figsize=(10, 8))
    top_features = mdi_importance.head(n_features)

    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.gca().invert_yaxis()
    plt.xlabel('Importance')
    plt.title(f'{model_name} - Mean Decrease in Impurity (MDI)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return mdi_importance

def plot_permutation_importance(model, model_name, X_train, y_train, X_test, y_test, n_features=10):
    """Plot permutation feature importance"""
    perm_importance = permutation_importance(
        model, X_test, y_test,
        n_repeats=10,
        random_state=GLOBAL_RANDOM_STATE,
        scoring='roc_auc'
    )

    perm_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': perm_importance.importances_mean,
        'std': perm_importance.importances_std
    }).sort_values('importance', ascending=False)

    # Plot top features
    plt.figure(figsize=(10, 8))
    top_features = perm_df.head(n_features)

    plt.barh(range(len(top_features)), top_features['importance'],
             xerr=top_features['std'], capsize=3)
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.gca().invert_yaxis()
    plt.xlabel('Importance')
    plt.title(f'{model_name} - Permutation Feature Importance')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return perm_df
