# AI Daily Lab — 2026-01-14

## Task
1. Generate a synthetic imbalanced binary classification dataset using `sklearn.datasets.make_classification` with 1200 samples, 8 features, 2 classes, and `weights=[0.9, 0.1]` to create an imbalance. Set `random_state=42` for reproducibility.
2. Split the dataset into training and testing sets (e.g., 70/30 split) using `sklearn.model_selection.train_test_split`.
3. Create an `imblearn.pipeline.Pipeline` that includes the following steps:
    *   `sklearn.preprocessing.StandardScaler`
    *   `imblearn.over_sampling.SMOTE` (set `random_state=42`)
    *   `sklearn.linear_model.LogisticRegression` (set `random_state=42`, `solver='liblinear'`)
4. Train the pipeline on the training data.
5. Predict probabilities for the positive class (class 1) on the scaled test set. Then, convert these probabilities to binary class labels (0 or 1) using a threshold of 0.5.
6. Calculate and print the `sklearn.metrics.classification_report` for the test set predictions.
7. Plot the Precision-Recall curve for the model on the test set using `sklearn.metrics.PrecisionRecallDisplay.from_predictions`. Ensure the plot has a clear title, labels, and displays the average precision score.

## Focus
ML pipelines / model evaluation (Imbalanced Classification)

## Dataset
Synthetic imbalanced binary classification data

## Hint
Remember to import `SMOTE` from `imblearn.over_sampling` and `Pipeline` from `imblearn.pipeline`. `imblearn.pipeline.Pipeline` is essential for correctly applying resampling techniques like SMOTE only to the training data within each cross-validation fold or during fitting. For the Precision-Recall curve, `PrecisionRecallDisplay.from_predictions` is very convenient.
