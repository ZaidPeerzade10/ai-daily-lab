# AI Daily Lab — 2025-12-21

## Task
1. Generate a synthetic binary classification dataset using `sklearn.datasets.make_classification` with at least 1000 samples, 5 informative features, and a significant class imbalance (e.g., `weights=[0.9, 0.1]` for 90% majority, 10% minority). Set `random_state` for reproducibility.
2. Split the dataset into training and testing sets (e.g., 70/30 split) using `sklearn.model_selection.train_test_split`.
3. Construct an `imblearn.pipeline.Pipeline` that first applies `StandardScaler` to the features, then applies `imblearn.over_sampling.SMOTE` (set `random_state`), and finally fits a `LogisticRegression` model (set `random_state`, `solver='liblinear'`).
4. Train the pipeline on the *training data*.
5. Evaluate the trained model's performance on the *test data*. Print the `sklearn.metrics.classification_report` to show precision, recall, and f1-score for each class, paying close attention to the minority class.
6. Plot the Precision-Recall curve for the minority class on the test set using `sklearn.metrics.PrecisionRecallDisplay.from_estimator`, clearly labeling the plot with a title.

## Focus
ML pipelines, imbalanced classification, feature engineering (scaling), model evaluation (classification report, PR curve), data visualization

## Dataset
Synthetic binary classification data with class imbalance generated using `sklearn.datasets.make_classification`.

## Hint
Remember to import `SMOTE` from `imblearn.over_sampling` and `Pipeline` from `imblearn.pipeline` for proper integration of resampling techniques within a scikit-learn compatible workflow.
