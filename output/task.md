# AI Daily Lab — 2025-12-18

## Task
1. Generate a synthetic multi-class classification dataset using `sklearn.datasets.make_classification` with at least 1000 samples, 10 features (e.g., `n_informative=5`), and 3 distinct classes (set `random_state` for reproducibility).
2. Split the dataset into training and testing sets (e.g., 80/20 split) using `sklearn.model_selection.train_test_split`.
3. Train a `sklearn.ensemble.RandomForestClassifier` on the training data (set `random_state` for reproducibility).
4. Predict class labels on the test set.
5. Print a detailed classification report using `sklearn.metrics.classification_report` to show precision, recall, and f1-score for each class.
6. Plot the confusion matrix for the test set predictions using `sklearn.metrics.ConfusionMatrixDisplay.from_estimator`, ensuring appropriate labels and a title.

## Focus
model evaluation, data visualization, ML pipelines

## Dataset
sklearn.datasets.make_classification (synthetic multi-class)

## Hint
Remember that `make_classification` allows you to specify `n_classes`. For plotting the confusion matrix, `ConfusionMatrixDisplay.from_estimator` is a convenient function that directly takes your fitted model and test data.
