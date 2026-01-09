# AI Daily Lab — 2026-01-09

## Task
1. Generate a synthetic binary classification dataset using `sklearn.datasets.make_classification` with at least 1000 samples, 10 features (e.g., `n_informative=5`, `n_redundant=3`, `n_repeated=2`), and 2 classes (set `random_state=42`).
2. Split the dataset into training and testing sets (e.g., 70/30 split) using `sklearn.model_selection.train_test_split`.
3. Create two distinct `sklearn.pipeline.Pipeline` objects:
    *   `pipeline_no_fs`: Consisting of `StandardScaler` followed by `LogisticRegression` (set `random_state=42`, `solver='liblinear'` for reproducibility).
    *   `pipeline_with_fs`: Consisting of `StandardScaler`, then `sklearn.feature_selection.SelectKBest` (e.g., select `k=5` features using `score_func=sklearn.feature_selection.f_classif`), followed by `LogisticRegression` (set `random_state=42`, `solver='liblinear'` for reproducibility).
4. Train both `pipeline_no_fs` and `pipeline_with_fs` on the training data.
5. Predict probabilities for the positive class (class 1) on the test set using both trained pipelines.
6. Calculate and report the `sklearn.metrics.roc_auc_score` for both models on the test set.
7. Briefly discuss the impact of including the feature selection step on the model's performance for this dataset, based on the reported ROC AUC scores.

## Focus
Feature Selection in ML Pipelines, Model Comparison

## Dataset
Synthetic binary classification data (`sklearn.datasets.make_classification`)

## Hint
Ensure `SelectKBest` is placed *after* `StandardScaler` in the pipeline for correct feature selection on scaled data. Remember to specify `k` (number of features to select) and `score_func` (e.g., `f_classif` for classification tasks) for `SelectKBest`.
