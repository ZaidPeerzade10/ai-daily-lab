# AI Daily Lab — 2026-01-08

## Task
1. Generate a synthetic binary classification dataset using `sklearn.datasets.make_classification` with 1000 samples, 4 numerical features, and 2 classes (`random_state=42`). Convert `X` into a pandas DataFrame and add a categorical feature named `color` (e.g., 'red', 'blue', 'green', 'yellow' with random distribution, ensuring a good mix).
2. Introduce missing values into the `color` feature by randomly replacing approximately 15% of its values with `np.nan`.
3. Create an `sklearn.pipeline.Pipeline` that first applies a `sklearn.compose.ColumnTransformer` for preprocessing and then fits a `sklearn.linear_model.LogisticRegression` model (set `random_state=42`, `solver='liblinear'` for reproducibility).
    *   **Inside the `ColumnTransformer`**:
        *   For the numerical features: Apply `SimpleImputer(strategy='mean')` followed by `StandardScaler`.
        *   For the `color` categorical feature: Apply `SimpleImputer(strategy='most_frequent')` followed by `OneHotEncoder(handle_unknown='ignore')`.
4. Evaluate the complete pipeline's performance using 5-fold cross-validation (`sklearn.model_selection.cross_val_score`) with `accuracy` as the scoring metric.
5. Report the mean accuracy and its standard deviation from the cross-validation.

## Focus
ML pipelines, Feature Engineering (categorical imputation and encoding)

## Dataset
Synthetic classification data with mixed numerical and categorical features, including missing categorical values.

## Hint
Remember to define separate lists for numerical and categorical column names to pass to the `ColumnTransformer`. The `handle_unknown='ignore'` parameter in `OneHotEncoder` is good practice to prevent errors if unseen categories appear in the test set during cross-validation.
