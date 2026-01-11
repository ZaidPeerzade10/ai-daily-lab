# AI Daily Lab — 2026-01-11

## Task
1. Generate a synthetic binary classification dataset using `sklearn.datasets.make_classification` with 1000 samples, 10 informative features, and 2 classes (set `random_state=42`).
2. Split the dataset into training and testing sets (e.g., 70/30 split) using `sklearn.model_selection.train_test_split`.
3. Create two distinct `sklearn.pipeline.Pipeline` objects for classification:
    *   `pipeline_no_pca`: Consisting of `StandardScaler` followed by `LogisticRegression` (set `random_state=42`, `solver='liblinear'` for reproducibility).
    *   `pipeline_with_pca`: Consisting of `StandardScaler`, then `sklearn.decomposition.PCA(n_components=2)`, followed by `LogisticRegression` (set `random_state=42`, `solver='liblinear'` for reproducibility).
4. Train both `pipeline_no_pca` and `pipeline_with_pca` on the training data.
5. Predict probabilities for the positive class (class 1) on the test set using both trained pipelines. Calculate and report the `sklearn.metrics.roc_auc_score` for both models on the test set.
6. **Visualize the transformed data**: Apply the `StandardScaler` and `PCA(n_components=2)` steps from `pipeline_with_pca` (using `.fit_transform()` on training and `.transform()` on testing data) to the *test set features*. Create a scatter plot of the two principal components, coloring the points by their actual class labels (`y_test`). Add appropriate titles and labels.
7. Briefly discuss the impact of including PCA on the model's performance for this dataset and what the visualization reveals about the data separation.

## Focus
ML Pipelines, Feature Engineering (Dimensionality Reduction), Model Evaluation, Data Visualization

## Dataset
Synthetic binary classification data from `sklearn.datasets.make_classification` with 10 informative features.

## Hint
For step 6, you will need to apply the `StandardScaler` and `PCA` transformations *separately* to `X_test` using the *fitted* transformers from `pipeline_with_pca` (e.g., `pipeline_with_pca.named_steps['standardscaler'].transform(X_test)` and then apply the PCA transformer to the scaled data). This will give you the 2 principal components for visualization.
