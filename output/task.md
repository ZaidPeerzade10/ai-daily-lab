# AI Daily Lab — 2025-12-22

## Task
1. Generate a synthetic binary classification dataset using `sklearn.datasets.make_classification` with at least 1000 samples, 5 numerical features, and 1 conceptual 'high-cardinality' categorical feature. To create this categorical feature, generate a numerical feature with a large number of unique integer values (e.g., 50-100) and then convert it to string type, adding it to your feature DataFrame.
2. Split the dataset into training and testing sets (e.g., 70/30 split) using `train_test_split`.
3. Create two distinct `sklearn.pipeline.Pipeline` objects for preprocessing and modeling:
    *   `pipeline_onehot_encoding`: Use `sklearn.compose.ColumnTransformer`. For the numerical features, apply `StandardScaler`. For the high-cardinality categorical feature, apply `OneHotEncoder(handle_unknown='ignore')`.
    *   `pipeline_feature_hashing`: Use `sklearn.compose.ColumnTransformer`. For the numerical features, apply `StandardScaler`. For the high-cardinality categorical feature, apply `FeatureHasher(n_features=15, input_type='string')` (you may adjust `n_features`).
4. Both pipelines should then fit a `LogisticRegression` model (using `solver='liblinear'` and a `random_state` for reproducibility).
5. Train both pipelines on the training data and evaluate their performance on the test set. Report the `accuracy_score` and `f1_score` for each pipeline, clearly stating which encoding strategy yielded which result.
6. For both pipelines, calculate probability predictions on the test set. Create two plots using `sklearn.calibration.CalibrationDisplay.from_estimator` (one for each pipeline) to visualize model calibration. Arrange them side-by-side or clearly distinguish them with titles indicating the encoding method. Discuss briefly which model appears better calibrated based on the plots.

## Focus
Feature Engineering (Categorical Hashing vs. One-Hot), ML Pipelines, Model Calibration

## Dataset
Synthetic binary classification data with high-cardinality categorical feature.

## Hint
When using `FeatureHasher` within `ColumnTransformer`, ensure the selected column is treated as a single string per row (e.g., ensure it's a series of strings). `input_type='string'` on `FeatureHasher` will treat each item in the input sequence as a string. For `CalibrationDisplay`, you'll pass the fitted pipeline as the estimator, along with the test features and labels. Use `plt.figure()` and `ax` arguments to create multiple subplots or separate plots.
