# AI Daily Lab — 2025-12-12

## Task
1. Generate a synthetic binary classification dataset using `sklearn.datasets.make_classification` with at least 1000 samples, 4 numerical features, and 2 classes. Convert the features and target into a pandas DataFrame.
2. Select two of the original numerical features (e.g., `feature_0`, `feature_1`) and apply `pd.cut` to each of them to discretize them into 3-4 bins (e.g., `['low', 'medium', 'high']`). These new binned features should be categorical and added to the DataFrame.
3. Create an `sklearn.compose.ColumnTransformer` to preprocess the data:
    *   For the *remaining original* numerical features (those not binned): Apply `StandardScaler`.
    *   For the *newly binned* categorical features: Apply `OneHotEncoder(handle_unknown='ignore')`.
4. Construct an `sklearn.pipeline.Pipeline` that first applies this `ColumnTransformer` and then trains a `GradientBoostingClassifier` (set `random_state` for reproducibility).
5. Evaluate the complete pipeline's performance using 5-fold cross-validation (`sklearn.model_selection.cross_val_score`) with `accuracy` as the scoring metric. Report the mean accuracy and its standard deviation.

## Focus
feature engineering, ML pipelines, model evaluation

## Dataset
synthetic classification

## Hint
When setting up the `ColumnTransformer`, ensure you correctly identify which features go to `StandardScaler` (original numerical features that were *not* binned) and which go to `OneHotEncoder` (the *new* binned categorical features). Use `make_column_transformer` for easier column selection.
