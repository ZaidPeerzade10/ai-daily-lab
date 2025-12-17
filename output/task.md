# AI Daily Lab — 2025-12-17

## Task
1. Generate a synthetic regression dataset using `sklearn.datasets.make_regression` with at least 800 samples, 4 informative features, and a small amount of noise.
2. Create a new numerical feature named `time_of_day` for each sample, ranging from 0 to 23 (e.g., using `np.random.randint`). Add this feature to your feature matrix `X`.
3. From `time_of_day`, create two new features: `time_of_day_sin` and `time_of_day_cos`, applying sine and cosine transformations respectively (e.g., `np.sin(2 * np.pi * time_of_day / 24)`).
4. Create two `sklearn.pipeline.Pipeline` objects:
    *   `pipeline_raw_tod`: Uses `sklearn.compose.ColumnTransformer` to apply `StandardScaler` to all original `make_regression` features AND the raw `time_of_day` feature. Then fit a `Ridge` regressor.
    *   `pipeline_cyclical_tod`: Uses `sklearn.compose.ColumnTransformer` to apply `StandardScaler` to all original `make_regression` features AND the `time_of_day_sin` and `time_of_day_cos` features. The raw `time_of_day` feature should *not* be used in this pipeline.
5. Evaluate both pipelines using `sklearn.model_selection.cross_val_score` with 5-fold cross-validation and `r2` as the scoring metric.
6. Print the mean and standard deviation of the R-squared scores for both pipelines, clearly indicating the performance difference due to cyclical feature encoding.

## Focus
Feature Engineering (Cyclical Encoding), ML Pipelines, Model Evaluation

## Dataset
Synthetic regression data generated with `make_regression` and additional engineered cyclical features.

## Hint
Remember to properly define your feature sets for the `ColumnTransformer` in each pipeline. For the `pipeline_cyclical_tod`, ensure you explicitly exclude the raw `time_of_day` feature while including its sine and cosine transformations.
