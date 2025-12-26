# AI Daily Lab — 2025-12-26

## Task
1. Generate a pandas DataFrame with 800 samples, including:
    *   Three numerical features: `amount` (random positive floats, intentionally introduce some high outliers to simulate skewed transactions), `age` (random integers between 18-65), `duration_months` (random integers between 1-120).
    *   One categorical feature: `region` (e.g., 'North', 'South', 'East', 'West' with varying proportions).
    *   Ensure `amount` has a right-skewed distribution with some clear outliers (e.g., using `np.random.exponential` or adding a few large values).
2. Calculate and display comprehensive descriptive statistics for the numerical features, grouped by the `region` categorical feature (e.g., mean, median, standard deviation, min, max, quartiles).
3. Create a set of subplots (e.g., 1 row, 2 columns) to visualize the distribution of `amount` and `duration_months` across different `region` categories. Use `seaborn.boxplot` or `seaborn.violinplot` for these visualizations. Ensure plots have appropriate titles and labels.
4. Focus on the `amount` feature. Apply a `log1p` transformation (i.e., `np.log1p(feature)`) to this feature to mitigate its skewness and outliers. Create another set of side-by-side subplots showing:
    *   A histogram or Kernel Density Estimate (KDE) plot of the *original* `amount` distribution.
    *   A histogram or KDE plot of the *log1p-transformed* `amount` distribution.
    Clearly label titles to highlight the effect of the transformation.
5. Compute the pairwise correlation matrix for all numerical features in the DataFrame (using the *log1p-transformed* `amount` for the correlation calculation). Visualize this matrix using a `seaborn.heatmap` with annotations, ensuring a clear title.

## Focus
pandas / numpy, data visualization, feature engineering

## Dataset
Synthetic Pandas DataFrame (mix of numerical with skew/outliers and categorical features)

## Hint
For introducing outliers in `amount`, you could generate most values from a normal/exponential distribution and then add a small percentage of significantly larger values. For step 3, `sns.boxplot(x='region', y='amount', data=df)` in combination with `plt.subplot` or `plt.figure(figsize=...)` can be useful. For step 4, `sns.histplot` or `sns.kdeplot` are good choices. Remember to handle plotting with `matplotlib.pyplot` for subplots and figure sizing.
