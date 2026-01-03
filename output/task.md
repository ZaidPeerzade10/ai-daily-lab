# AI Daily Lab — 2026-01-03

## Task
1. Generate a synthetic binary classification dataset using `sklearn.datasets.make_classification` with at least 1000 samples, 4 informative features, and 2 classes (set `random_state` for reproducibility).
2. Split the dataset into training and testing sets (e.g., 70/30 split) using `sklearn.model_selection.train_test_split`.
3. Standardize the features using `sklearn.preprocessing.StandardScaler` on the training data, then transform both training and testing sets.
4. Build a simple sequential neural network model using `tensorflow.keras.models.Sequential` for binary classification. The model should have:
    *   An input layer matching the number of features.
    *   One hidden `Dense` layer with `relu` activation (e.g., 16-32 units).
    *   An output `Dense` layer with a single unit and `sigmoid` activation.
5. Compile the model using the `adam` optimizer and `binary_crossentropy` loss.
6. Train the model on the scaled training data for a suitable number of epochs (e.g., 30-50) and a batch size.
7. Predict probabilities on the scaled test set. Convert these probabilities to binary class labels (0 or 1) using a threshold (e.g., 0.5).
8. Print the `sklearn.metrics.classification_report` for the test set predictions.
9. Generate and plot a confusion matrix using `sklearn.metrics.ConfusionMatrixDisplay.from_predictions` for the test set, clearly labeling the plot with a title (e.g., 'Confusion Matrix for Keras Binary Classifier').

## Focus
Basic AI experimentation, Model Evaluation, Data Visualization

## Dataset
Synthetic binary classification dataset

## Hint
For Keras, ensure your target `y` is in the correct shape (e.g., `y_train.reshape(-1, 1)`) and data types (`float32`) for training. Remember to use `model.predict` to get probabilities and then `(probabilities > 0.5).astype(int)` to convert them to class labels before generating the confusion matrix and classification report.
