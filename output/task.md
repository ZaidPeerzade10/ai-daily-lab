# AI Daily Lab — 2026-01-04

## Task
1. Generate a synthetic binary classification dataset using `sklearn.datasets.make_classification` with at least 1000 samples, 5 informative features, and 2 classes (set `random_state` for reproducibility).
2. Split the dataset into training and testing sets (e.g., 70/30 split) using `sklearn.model_selection.train_test_split`.
3. Define a function `create_keras_model(optimizer='adam', units=32)` that builds and compiles a `tensorflow.keras.models.Sequential` model. This model should have an input layer matching the number of features, one `Dense` hidden layer with `relu` activation (using `units` as a parameter), and a `Dense` output layer with `sigmoid` activation. Compile it with `binary_crossentropy` loss.
4. Wrap this Keras model using `tensorflow.keras.wrappers.scikit_learn.KerasClassifier`.
5. Construct an `sklearn.pipeline.Pipeline` that first applies `sklearn.preprocessing.StandardScaler` and then uses the wrapped `KerasClassifier`.
6. Define a hyperparameter grid for `sklearn.model_selection.GridSearchCV` to tune the following parameters of the Keras model within the pipeline:
    *   `kerasclassifier__batch_size` (e.g., `[32, 64]`)
    *   `kerasclassifier__epochs` (e.g., `[10, 20]`)
    *   `kerasclassifier__model__units` (e.g., `[16, 32]`)
    *   `kerasclassifier__optimizer` (e.g., `['adam', 'rmsprop']`)
7. Perform `GridSearchCV` with 3-fold cross-validation and `scoring='roc_auc'` on the training data. (Set `n_jobs=-1` for faster execution).
8. Report the `best_params_` and `best_score_` from `GridSearchCV`. Then, using the `best_estimator_`, predict class labels on the test set and print the `sklearn.metrics.classification_report`.

## Focus
basic AI experimentation

## Dataset
Synthetic binary classification data using `sklearn.datasets.make_classification`.

## Hint
Remember to use `tensorflow.keras.wrappers.scikit_learn.KerasClassifier` to integrate your Keras model with scikit-learn's `Pipeline` and `GridSearchCV`. When defining the `param_grid` for `GridSearchCV`, specify Keras model parameters that are arguments to your `create_keras_model` function with the `kerasclassifier__model__` prefix (e.g., `kerasclassifier__model__units`). Other parameters of `KerasClassifier` itself (like `batch_size`, `epochs`, `optimizer`) should use the `kerasclassifier__` prefix.
