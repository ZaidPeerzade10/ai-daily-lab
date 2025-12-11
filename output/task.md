# AI Daily Lab — 2025-12-11

## Task
1. Generate a synthetic regression dataset using `sklearn.datasets.make_regression` with at least 500 samples, 7 features, and a small amount of noise.
2. Implement a custom `sklearn` transformer (inheriting from `BaseEstimator`, `TransformerMixin`) named `CustomPolynomialFeatures`.
   This transformer should take a list of feature names (or indices) as an initialization argument. Its `transform` method should apply `PolynomialFeatures` (with `degree=2`, `include_bias=False`) only to the specified features, and pass through other features unchanged.
3. Create an `sklearn.pipeline.Pipeline` that uses an `sklearn.compose.ColumnTransformer`.
   *   Apply `StandardScaler` to all numerical features *not* handled by your custom transformer.
   *   Apply your `CustomPolynomialFeatures` transformer to 3-4 specific numerical features.
   *   The pipeline should then fit a `Ridge` regressor.
4. Evaluate the pipeline's performance using `sklearn.model_selection.cross_val_score` with 5-fold cross-validation and `neg_mean_squared_error` as the scoring metric.
5. Print the mean and standard deviation of the Mean Squared Error (MSE) for the pipeline (remembering to convert `neg_mean_squared_error` to positive MSE values).

## Focus
ML Pipelines / Feature Engineering

## Dataset
`sklearn.datasets.make_regression`

## Hint
For your `CustomPolynomialFeatures` transformer, the `fit` method can simply return `self`. In the `transform` method, ensure you correctly select the columns for polynomial transformation, apply it, and then combine the transformed features with the untouched features. When setting up the `ColumnTransformer`, use `make_column_selector` or explicit lists of column names/indices to route features to the correct transformers, and remember the 'remainder' parameter for handling untransformed features.
