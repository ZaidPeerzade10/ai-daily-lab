# AI Daily Lab — 2025-12-25

## Task
1. Generate a synthetic dataset using `sklearn.datasets.make_blobs` with at least 1000 samples, 5 features, and 4 distinct centers (clusters). Set `random_state=42`.
2. Apply `sklearn.preprocessing.StandardScaler` to the generated features.
3. Implement the Elbow method and Silhouette analysis: For a range of K values (e.g., from 2 to 8), train `sklearn.cluster.KMeans` (set `random_state=42`). For each K, record the `inertia_` and the `silhouette_score` (using the scaled data and the predicted cluster labels).
4. Plot both the `inertia_` values (Elbow curve) and `silhouette_score` values against the corresponding K values. Clearly label the axes and provide descriptive titles for both plots.
5. Based on your plots, identify the optimal number of clusters (K). Train a final `KMeans` model using this optimal K on the scaled data.
6. Reduce the dimensionality of the *scaled* features to 2 principal components using `sklearn.decomposition.PCA`. Transform the original scaled data and the centroids of your final `KMeans` model into this 2D PCA space.
7. Visualize the final clusters in the 2D PCA space. Plot the data points, coloring them according to their assigned cluster labels from the final `KMeans` model. Overlay the transformed cluster centroids on the plot. Ensure axes are labeled and the plot has a clear title.

## Focus
basic AI experimentation, model evaluation, data visualization

## Dataset
Synthetic data from `sklearn.datasets.make_blobs`

## Hint
Remember to transform your KMeans centroids into the PCA space to plot them correctly. For plotting, `matplotlib.pyplot.scatter` is useful. Use `plt.figure(figsize=...)` for better plot readability.
