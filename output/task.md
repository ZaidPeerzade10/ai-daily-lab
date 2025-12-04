# AI Daily Lab — 2025-12-04

## Task
1. Generate a synthetic dataset using `sklearn.datasets.make_blobs` with at least 500 samples, 4 numerical features, and 3 distinct clusters. Convert this into a pandas DataFrame, including the cluster labels as a feature (e.g., `cluster_id`).
2. Add a new categorical feature to the DataFrame (e.g., `group`) with 2-3 distinct values, randomly assigned.
3. Using `seaborn` and `matplotlib.pyplot`, create the following visualizations to explore the data:
    *   A pair plot (`sns.pairplot`) for the numerical features, coloring the points by the `cluster_id`.
    *   A set of histograms (or KDE plots) for `feature_1` and `feature_2`, separated for each unique value of the newly created `group` categorical feature (e.g., using `sns.FacetGrid` or `sns.histplot` with `hue` and `col`).
    *   A box plot (or violin plot) showing the distribution of `feature_3` across the different `cluster_id`s.
4. Ensure all plots have appropriate titles and labels.

## Focus
data visualization

## Dataset
Synthetic data generated with `sklearn.datasets.make_blobs` and additional pandas DataFrame manipulation.

## Hint
Leverage `seaborn` for high-level plotting functions like `pairplot`, `histplot`, and `boxplot`. For separated histograms/KDEs, consider `sns.FacetGrid` or using `hue` and `col` parameters in `sns.histplot`.
