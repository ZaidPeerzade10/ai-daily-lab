# AI Daily Lab — 2025-12-05

## Task
1. Generate a synthetic binary classification dataset using `sklearn.datasets.make_classification` (e.g., 1000 samples, 10 features, 2 informative features, 2 classes).
2. Split the dataset into training and testing sets (e.g., 80/20 split) using `train_test_split`.
3. Train a `LogisticRegression` model on the training data.
4. Predict class labels and class probabilities for the positive class on the test set.
5. Calculate and print the following evaluation metrics for the test set predictions: Accuracy, Precision, Recall, F1-score, and ROC AUC score.
6. Plot the Receiver Operating Characteristic (ROC) curve for the model using `matplotlib.pyplot`, clearly labeling axes and adding a title. Include the AUC score in the plot legend.

## Focus
model evaluation

## Dataset
synthetic classification (sklearn.datasets.make_classification)

## Hint
Remember to import necessary metrics from `sklearn.metrics` such as `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `roc_auc_score`, and `roc_curve`. For plotting the ROC curve, you'll need the false positive rates (fpr) and true positive rates (tpr) from `roc_curve`.
