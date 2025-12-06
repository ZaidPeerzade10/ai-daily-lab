# AI Daily Lab — 2025-12-06

## Task
1. Generate a synthetic regression dataset using `sklearn.datasets.make_regression` with at least 500 samples, 3 informative features, and a small amount of noise.
2. Create two distinct `sklearn.pipeline.Pipeline` objects:
    *   `pipeline_simple`: Consisting of `StandardScaler` followed by `LinearRegression`.
    *   `pipeline_poly`: Consisting of `PolynomialFeatures` (set `degree=2`), then `StandardScaler`, then `LinearRegression`.
3. Evaluate both pipelines using `sklearn.model_selection.cross_val_score` with 5-fold cross-validation and `neg_mean_squared_error` as the scoring metric.
4. Print the mean and standard deviation of the Mean Squared Error (MSE) for both pipelines, clearly indicating which result belongs to which pipeline. (Remember to convert `neg_mean_squared_error` to positive MSE values for interpretability).

## Focus
ML Pipelines, Feature Engineering (Polynomial Features), Model Evaluation

## Dataset
Synthetic regression dataset (`sklearn.datasets.make_regression`)

## Hint
Remember to import all necessary components from `sklearn.preprocessing`, `sklearn.linear_model`, `sklearn.pipeline`, and `sklearn.model_selection`. The `PolynomialFeatures` step should be placed before `StandardScaler` in `pipeline_poly` to ensure scaling is applied to the newly created polynomial features.
