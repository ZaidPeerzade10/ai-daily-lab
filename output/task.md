# AI Daily Lab — 2025-12-03

## Task
1. Generate a synthetic regression dataset using `sklearn.datasets.make_regression` with at least 500 samples and 5 features.
2. Create an `sklearn.pipeline.Pipeline` that first applies `StandardScaler` to the features and then fits a `Ridge` regressor.
3. Define a hyperparameter grid for the `Ridge` regressor within the pipeline, tuning the `alpha` parameter across at least 3 distinct values (e.g., `[0.1, 1.0, 10.0]`).
4. Use `sklearn.model_selection.GridSearchCV` with the pipeline and the defined parameter grid to find the best hyperparameters. Use `neg_mean_squared_error` as the scoring metric and apply 3-fold cross-validation.
5. Report the best hyperparameters found by `GridSearchCV` and the corresponding best score (remembering to convert the `neg_mean_squared_error` back to a positive MSE value).

## Focus
ML pipelines, basic AI experimentation, model evaluation

## Dataset
Synthetic data from `sklearn.datasets.make_regression`

## Hint
When defining the parameter grid for `GridSearchCV` for a pipeline, prefix the regressor's parameter names with its name in the pipeline (e.g., `ridge__alpha`). Remember that `GridSearchCV` returns a negative score for `neg_mean_squared_error`, so you'll need to multiply by -1 to get the actual MSE.
