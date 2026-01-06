# AI Daily Lab — 2026-01-06

## Task
1. Generate a synthetic binary classification dataset using `sklearn.datasets.make_classification` with at least 1000 samples, 6 informative features, and 2 classes (set `random_state` for reproducibility). Convert the features (`X`) and target (`y`) into a pandas DataFrame.
2. Split the dataset into training and testing sets (e.g., 70/30 split) using `sklearn.model_selection.train_test_split`.
3. Create two distinct `sklearn.pipeline.Pipeline` objects for classification:
    *   `pipeline_baseline`: Consisting of `StandardScaler` followed by `LogisticRegression` (set `random_state=42`, `solver='liblinear'` for reproducibility).
    *   `pipeline_polynomial`: Consisting of `StandardScaler`, then `PolynomialFeatures(degree=2, include_bias=False)`, followed by `LogisticRegression` (set `random_state=42`, `solver='liblinear'` for reproducibility).
4. Evaluate both pipelines using 5-fold cross-validation (`sklearn.model_selection.cross_val_score`) on the training data. Use `scoring='roc_auc'` for performance comparison.
5. Report the mean and standard deviation of the ROC AUC scores for both the `pipeline_baseline` and `pipeline_polynomial`. Based on these results, briefly discuss the impact of including polynomial features on model performance for this dataset.

## Focus
feature engineering, ML pipelines, model evaluation, pandas/numpy

## Dataset
`sklearn.datasets.make_classification` for a synthetic classification problem.

## Hint
Remember that `PolynomialFeatures` can significantly increase the number of features. `include_bias=False` prevents adding a column of all ones, which is usually handled by the `LogisticRegression` intercept. Focus on comparing the ROC AUC scores for a robust evaluation of classification performance.
