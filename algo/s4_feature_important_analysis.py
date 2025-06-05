import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
from pathlib import Path
from utils.path_info import feature_importance_results_path
import matplotlib.pyplot as plt
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans
from configs.GLOBAL_CONFIG import GLOBAL_RANDOM_STATE
from sklearn.inspection import permutation_importance
from sklearn.mixture import GaussianMixture
import seaborn as sns
from scipy.stats import spearmanr


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
    try:
        importance = model.feature_importances_
    except AttributeError:
        # Only for tree model, it has feature_importances_
        return
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
    plt.savefig(Path(feature_importance_results_path, f'{model_name}_mdi_importance.png'))

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
    plt.savefig(Path(feature_importance_results_path, f'{model_name}_pfi_importance.png'))
    return perm_df



def s4_cluster_level_importance_analysis(models, experiment_data_dict):
    X_train, y_train = experiment_data_dict["X_train"], experiment_data_dict["y_train"]
    X_test, y_test = experiment_data_dict["X_test"], experiment_data_dict["y_test"]

    distance_matrix = compute_spearman_distance_matrix(X_train)
    plot_corr_and_distance(X_train, distance_matrix)
    clusterer = OptimalClusterer(variance_threshold=0.95, max_clusters=10, random_state=42)
    clusterer.apply_pca(distance_matrix)
    optimal_clusters = clusterer.find_optimal_clusters(method='silhouette')

    scores = []
    cluster_range = range(2, clusterer.max_clusters + 1)
    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, random_state=42).fit(clusterer.pca_data)
        score = silhouette_score(clusterer.pca_data, kmeans.labels_)
        scores.append(score)
    plot_cluster_score(cluster_range, scores, optimal_clusters)
    clusters = clusterer.apply_kmeans()
    save_cluster_info(clusters)
    gmm_probabilities = clusterer.apply_gmm(reg_covar=0.1)
    plot_heatmap_gmm_prob(gmm_probabilities)

    feature_important_results = {}
    for model_name, model in models.items():
        try:
            cluster_imp_mdi = calculate_cluster_importance_mdi(model, X_train.columns, clusters)
            plot_cluster_mdi(cluster_imp_mdi, model_name)
        except AttributeError:
            cluster_imp_mdi = None
        cluster_imp_pfi = calculate_cluster_importance_pfi(model, X_train, y_train, clusters)
        plot_cluster_pfi(cluster_imp_pfi, model_name)

        fi_metrics = {
            "mdi": cluster_imp_mdi,
            "pfi": cluster_imp_pfi
        }
        feature_important_results[model_name] = fi_metrics
    return feature_important_results


def plot_cluster_mdi(cluster_imp_mdi, model_name):
    plt.figure(figsize=(10, 8))
    cluster_imp_mdi.sort_values('mean').plot(
        kind='barh',
        y='mean',
        xerr='std',
        legend=False,
        color='skyblue',
        capsize=3
    )
    plt.title(f'{model_name} - Cluster Importance (Mean Decrease Impurity)', fontsize=14)
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Cluster', fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(Path(feature_importance_results_path, f"{model_name}_cluster_mdi_importance.png"))
    return

def plot_cluster_pfi(cluster_imp_pfi, model_name):
    plt.figure(figsize=(10, 8))
    cluster_imp_pfi.sort_values('mean').plot(
        kind='barh',
        y='mean',
        xerr='std',
        legend=False,
        color='salmon',
        capsize=3
    )
    plt.title(f'{model_name} - Cluster Importance (Permutation Feature Importance)', fontsize=14)
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Cluster', fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(Path(feature_importance_results_path, f"{model_name}_cluster_pfi_importance.png"))

def calculate_cluster_importance_mdi(model, feature_names, clusters):
    """
    Calculate Mean Decrease Impurity (MDI) feature importance at the cluster level.

    Parameters:
    -----------
    model : fitted sklearn ensemble model
        Model with estimators_ attribute
    feature_names : list
        List of feature names
    clusters : dict
        Dictionary mapping cluster IDs to lists of feature names

    Returns:
    --------
    cluster_importance : pandas DataFrame
        DataFrame with mean and std of cluster importance
    """
    # Extract feature importance from all trees
    if hasattr(model, 'estimators_'):
        importances = {i: tree.feature_importances_
                      for i, tree in enumerate(model.estimators_)}
    else:
        importances = {0: model.feature_importances_}

    # Convert to DataFrame
    imp_df = pd.DataFrame.from_dict(importances, orient='index')
    imp_df.columns = feature_names

    # Replace zeros with NaN (happens when max_features=1)
    imp_df = imp_df.replace(0, np.nan)

    # Group by clusters
    cluster_importance = pd.DataFrame(columns=['mean', 'std'])

    for cluster_id, features in clusters.items():
        # Filter to include only features that exist in the model
        valid_features = [f for f in features if f in feature_names]

        if valid_features:
            # Sum importances across features in cluster for each tree
            cluster_imp = imp_df[valid_features].sum(axis=1)

            # Calculate mean and std of cluster importance
            cluster_importance.loc[f'Cluster_{cluster_id}', 'mean'] = cluster_imp.mean()
            if len(cluster_imp) > 1:
                cluster_importance.loc[f'Cluster_{cluster_id}', 'std'] = (
                    cluster_imp.std() * cluster_imp.shape[0]**-0.5
                )
            else:
                cluster_importance.loc[f'Cluster_{cluster_id}', 'std'] = 0

    # Normalize to sum to 1
    total_importance = cluster_importance['mean'].sum()
    if total_importance > 0:
        cluster_importance['mean'] = cluster_importance['mean'] / total_importance

    return cluster_importance


def calculate_cluster_importance_pfi(model, X, y, clusters, cv=5, scoring='neg_log_loss'):
    """
    Calculate Mean performance feature importance (PFI) at the cluster level.

    Parameters:
    -----------
    model : sklearn estimator
        Model with fit and predict_proba methods
    X : pandas DataFrame
        Feature matrix
    y : pandas Series or array-like
        Target variable
    clusters : dict
        Dictionary mapping cluster IDs to lists of feature names
    cv : int, default=5
        Number of cross-validation folds
    scoring : str, default='neg_log_loss'
        Scoring metric for evaluating importance

    Returns:
    --------
    cluster_importance : pandas DataFrame
        DataFrame with mean and std of cluster importance
    """
    from sklearn.model_selection import KFold
    from sklearn.metrics import log_loss

    # Set up cross-validation
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)

    # Initialize results DataFrames
    baseline_scores = pd.Series(dtype='float64')  # baseline scores
    permutation_scores = pd.DataFrame(columns=clusters.keys())  # scores after permutation

    # Perform cross-validation
    for i, (train_idx, test_idx) in enumerate(kf.split(X)):
        # Split data
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

        # Fit model
        model.fit(X_train, y_train)

        # Get baseline predictions and score
        y_pred = model.predict_proba(X_test)
        baseline_scores.loc[i] = -log_loss(y_test, y_pred, labels=model.classes_)

        # For each cluster, shuffle all its features and compute score
        for cluster_id in clusters.keys():
            # Create a copy of the test set
            X_test_permuted = X_test.copy()

            # Get cluster features that exist in the dataset
            cluster_features = [f for f in clusters[cluster_id] if f in X.columns]

            if cluster_features:  # Only process if there are valid features
                # Shuffle all features in the cluster with the same permutation
                # This preserves correlations within cluster
                permutation_idx = np.random.permutation(len(X_test))
                for feature in cluster_features:
                    X_test_permuted[feature] = X_test_permuted[feature].values[permutation_idx]

                # Get predictions on permuted data
                y_pred_permuted = model.predict_proba(X_test_permuted)

                # Calculate score
                permutation_scores.loc[i, cluster_id] = -log_loss(
                    y_test, y_pred_permuted, labels=model.classes_
                )

    # Calculate importance as normalized performance drop
    importance = (-1 * permutation_scores).add(baseline_scores, axis=0)
    importance = importance / (-1 * permutation_scores)

    # Calculate mean and std
    cluster_importance = pd.DataFrame({
        'mean': importance.mean(),
        'std': importance.std() * importance.shape[0]**-0.5
    })

    # Rename index for clarity
    cluster_importance.index = [f'Cluster_{i}' for i in cluster_importance.index]

    return cluster_importance


def plot_heatmap_gmm_prob(gmm_probabilities):
    # Plot heatmap of GMM probabilities
    plt.figure(figsize=(5, 6))
    sns.heatmap(gmm_probabilities, cmap='viridis', vmin=0, vmax=1)
    plt.title("GMM Cluster Membership Probabilities")
    plt.tight_layout()
    plt.savefig(Path(feature_importance_results_path, "gmm_probabilities.png"))
    return


def save_cluster_info(clusters):
    rows = []

    for cluster_id, features in clusters.items():
        for feature in features:
            rows.append({"cluster_id": cluster_id, "feature": feature})

    df_clusters = pd.DataFrame(rows)
    df_clusters.to_csv(Path(feature_importance_results_path, "clusters_info.csv"), index=False)
    return


def plot_cluster_score(cluster_range, scores, optimal_clusters):
    plt.figure(figsize=(6, 4))
    plt.plot(cluster_range, scores, marker='o', linestyle='-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for Different Cluster Counts')
    plt.grid(alpha=0.3)
    plt.xticks(cluster_range)
    plt.axvline(x=optimal_clusters, color='red', linestyle='--', label=f'Optimal K: {optimal_clusters}')
    plt.legend()
    plt.savefig(Path(feature_importance_results_path, f'cluster_score.png'))
    return



def plot_corr_and_distance(X_train, distance_matrix):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot correlation matrix
    sns.heatmap(X_train.corr(), cmap='coolwarm', center=0, ax=axes[0],
                cbar_kws={"shrink": .8})
    axes[0].set_title('Correlation Matrix', fontsize=14)

    # Plot distance matrix
    sns.heatmap(distance_matrix, cmap='viridis', ax=axes[1],
                cbar_kws={"shrink": .8})
    axes[1].set_title('Distance Matrix', fontsize=14)

    plt.tight_layout()
    plt.savefig(Path(feature_importance_results_path, "corr_and_distance_matrix.png"))
    return


def compute_spearman_distance_matrix(X):
    """
    Computes a distance matrix (1 - |Spearman correlation|) for the input DataFrame or array.

    Parameters:
    -----------
    X : pandas.DataFrame or np.ndarray
        Input data where rows are samples and columns are features.

    Returns:
    --------
    pd.DataFrame
        Distance matrix with feature names as both row and column labels.
    """
    # Compute Spearman correlation
    corr = spearmanr(X).correlation

    # Ensure symmetry and correct diagonal
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1)

    # Compute distance matrix
    distance_matrix = 1 - np.abs(corr)

    # Assign feature names
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns
    else:
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]

    return pd.DataFrame(distance_matrix, index=feature_names, columns=feature_names)


class OptimalClusterer:
    """
    A class that combines dimensionality reduction with clustering techniques to find
    optimal feature clusters based on a distance matrix.

    The workflow is:
    1. Apply PCA to reduce dimensionality of the distance matrix while preserving variance
    2. Find the optimal number of clusters using cluster quality metrics
    3. Apply K-means clustering to get initial cluster assignments
    4. Refine with Gaussian Mixture Model to get probabilistic cluster memberships

    Parameters:
    -----------
    variance_threshold : float, default=0.95
        The minimum explained variance to determine number of PCA components
    max_clusters : int, default=10
        The maximum number of clusters to consider when determining optimal k
    random_state : int, default=42
        Random seed for reproducibility
    """
    def __init__(self, variance_threshold=0.95, max_clusters=10, random_state=42):
        self.variance_threshold = variance_threshold
        self.max_clusters = max_clusters
        self.random_state = random_state
        self.features = None
        self.pca_data = None
        self.optimal_k = None
        self.kmeans = None

    def apply_pca(self, dist_matrix):
        """
        Apply PCA to the distance matrix and determine the optimal number of components.

        This method:
        1. Stores the feature names for later use
        2. Fits a PCA model to the distance matrix
        3. Determines how many components are needed to reach the variance threshold
        4. Stores the reduced data with optimal number of components
        5. Plots the cumulative explained variance

        Parameters:
        -----------
        dist_matrix : pandas.DataFrame
            The distance matrix between features

        Returns:
        --------
        None
        """
        self.features = dist_matrix.index
        pca = PCA(random_state=self.random_state)
        points = pca.fit_transform(dist_matrix)
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.argmax(cumulative_variance >= self.variance_threshold) + 1
        self.pca_data = points[:, :n_components]

        plt.figure(figsize=(8, 5))
        plt.plot(cumulative_variance, marker='o')
        plt.axhline(self.variance_threshold, color='red', linestyle='--')
        plt.title('PCA Variance Captured')
        plt.xlabel('Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.grid()
        plt.savefig(Path(feature_importance_results_path, "PCA_features.png"))

    def find_optimal_clusters(self, method='silhouette'):
        """
        Find the optimal number of clusters using a specified cluster quality metric.

        This method:
        1. Tests different numbers of clusters (from 2 to max_clusters)
        2. For each K value, fits a K-means model and calculates the quality score
        3. Determines the optimal K that maximizes the quality score

        Available metrics:
        - 'silhouette': Measures how similar a point is to its own cluster compared to others
        - 'calinski': Ratio of between-cluster to within-cluster dispersion
        - 'davies': Average similarity of clusters (lower is better, so score is negated)

        Parameters:
        -----------
        method : str, default='silhouette'
            The cluster quality metric to use

        Returns:
        --------
        int
            The optimal number of clusters
        """
        scores = []
        methods = {'silhouette': silhouette_score, 'calinski': calinski_harabasz_score, 'davies': davies_bouldin_score}
        for k in range(2, self.max_clusters+1):
            kmeans = KMeans(n_clusters=k, random_state=self.random_state).fit(self.pca_data)
            if method == 'davies':
                score = -methods[method](self.pca_data, kmeans.labels_)
            else:
                score = methods[method](self.pca_data, kmeans.labels_)
            scores.append(score)
        self.optimal_k = np.argmax(scores) + 2
        return self.optimal_k

    def apply_kmeans(self):
        """
        Apply K-means clustering using the optimal number of clusters.

        This method:
        1. Fits a K-means model with the optimal K
        2. Creates a dictionary mapping cluster labels to feature names

        Returns:
        --------
        dict
            Dictionary where keys are cluster labels and values are lists of feature names
        """
        self.kmeans = KMeans(n_clusters=self.optimal_k, random_state=self.random_state).fit(self.pca_data)
        clusters = {i: [] for i in range(self.optimal_k)}
        for feature, label in zip(self.features, self.kmeans.labels_):
            clusters[label].append(feature)
        return clusters

    def apply_gmm(self, reg_covar=1e-3):
        """
        Apply Gaussian Mixture Model for soft clustering.

        This method:
        1. Initializes GMM with K-means cluster centers for better convergence
        2. Fits the GMM to the PCA-reduced data
        3. Gets the probability of each feature belonging to each cluster

        Parameters:
        -----------
        reg_covar : float, default=1e-3
            Regularization parameter for covariance matrices

        Returns:
        --------
        pandas.DataFrame
            DataFrame containing the probability of each feature (rows)
            belonging to each cluster (columns)
        """
        gmm = GaussianMixture(n_components=self.optimal_k, random_state=self.random_state,
                             means_init=self.kmeans.cluster_centers_, reg_covar=reg_covar)
        gmm.fit(self.pca_data)
        gmm_probs = gmm.predict_proba(self.pca_data)
        gmm_df = pd.DataFrame(gmm_probs, index=self.features,
                             columns=[f'Cluster_{i}' for i in range(self.optimal_k)])
        return gmm_df