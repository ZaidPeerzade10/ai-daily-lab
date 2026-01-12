# AI Daily Lab — 2026-01-12

## Task
1. Generate a synthetic 2D dataset for clustering using `sklearn.datasets.make_blobs` with 1000 samples, 2 features, and 4 cluster centers. Set `random_state=42` for reproducibility.
2. Apply `sklearn.preprocessing.StandardScaler` to the generated features.
3. Perform K-Means clustering on the scaled data. Initialize `sklearn.cluster.KMeans` with `n_clusters=4` and `random_state=42` (set `n_init='auto'` or `n_init=10` to suppress warnings for older scikit-learn versions). Fit the model and obtain the cluster labels.
4. Calculate and report the `sklearn.metrics.silhouette_score` using the scaled features and the obtained cluster labels.
5. Create a scatter plot of the 2D features (either original or scaled), coloring each data point according to its assigned K-Means cluster. Ensure the plot has appropriate axis labels, a clear title (e.g., 'K-Means Clusters (Silhouette Score: X.XX)'), and a legend (if distinct colors are used).
6. Briefly explain what the Silhouette Score measures and why it's a useful metric for evaluating clustering results, especially when true labels are not available.

## Focus
Unsupervised Learning (Clustering), Data Preprocessing, Model Evaluation, Data Visualization

## Dataset
Synthetic 2D dataset with distinct clusters (`sklearn.datasets.make_blobs`)

## Hint
Remember to fit the `StandardScaler` on the data and then transform it. For K-Means, ensure you pass the scaled data. When plotting, you can use the original data for features but color them by the cluster labels derived from the scaled data.
