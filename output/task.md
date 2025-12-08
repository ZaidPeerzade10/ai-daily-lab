# AI Daily Lab — 2025-12-08

## Task
1. Generate a synthetic dataset suitable for clustering using `sklearn.datasets.make_blobs` with at least 700 samples, 6 numerical features, and 4 distinct clusters (do not use the true cluster labels for modeling).
2. Apply `sklearn.cluster.KMeans` to the generated features to discover 4 clusters. Initialize KMeans with a `random_state` for reproducibility.
3. Evaluate the quality of the discovered clusters by calculating the `sklearn.metrics.silhouette_score`.
4. To visualize the clustering, reduce the dimensionality of the original features to 2 using `sklearn.decomposition.PCA`.
5. Create a scatter plot using `matplotlib.pyplot` or `seaborn` of the 2 principal components, coloring the points based on the clusters identified by KMeans. Title the plot with the calculated Silhouette Score.

## Focus
basic AI experimentation, model evaluation, data visualization

## Dataset
Synthetic (sklearn.datasets.make_blobs)

## Hint
Remember to import `KMeans`, `silhouette_score`, and `PCA` from `sklearn`. Convert the generated data into a pandas DataFrame for easier manipulation and plotting. For visualization, you'll need `matplotlib.pyplot` or `seaborn`.
