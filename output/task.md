# AI Daily Lab — 2025-12-27

## Task
1. Generate a synthetic binary classification dataset using `sklearn.datasets.make_classification` with at least 1000 samples, 6 numerical features, and 2 classes (set `random_state` for reproducibility). Convert the features (`X`) and target (`y`) into a pandas DataFrame.
2. Introduce missing values into the DataFrame:
    *   For `feature_0`: Randomly replace approximately 15% of its values with `np.nan`.
    *   For `feature_1`: Randomly replace approximately 10% of its values with `np.nan`.
    *   For `feature_2`: Randomly replace approximately 5% of its values with `np.nan`.
3. Create an `sklearn.pipeline.Pipeline` that first applies a `sklearn.compose.ColumnTransformer` for preprocessing and then fits a `sklearn.ensemble.RandomForestClassifier` (set `random_state` for reproducibility).
    *   **Inside the `ColumnTransformer`**:
        *   For `feature_0`: Apply `SimpleImputer(strategy='mean')` followed by `StandardScaler`.
        *   For `feature_1`: Apply `KNeighborsImputer(n_neighbors=5)` followed by `StandardScaler`.
        *   For `feature_2`: Apply `SimpleImputer(strategy='median')` followed by `StandardScaler`.
        *   For the *remaining numerical features* (`feature_3` to `feature_5`): Apply `StandardScaler` directly (no imputation needed).
4. Evaluate the complete pipeline's performance using 5-fold cross-validation (`sklearn.model_selection.cross_val_score`) with `accuracy` as the scoring metric.
5. Report the mean accuracy and its standard deviation from the cross-validation.

## Focus
Data Preprocessing, Missing Value Imputation, ML Pipelines, Cross-Validation

## Dataset
Synthetic binary classification with intentionally introduced `np.nan` values in specific features.

## Hint
When constructing the `ColumnTransformer`, remember to specify lists of column names or indices for each preprocessing step. For instance, `('mean_impute_scale', Pipeline([('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())]), ['feature_0'])`. Ensure `KNeighborsImputer` is part of a sub-pipeline.
