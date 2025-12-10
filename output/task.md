# AI Daily Lab — 2025-12-10

## Task
1. Generate a synthetic binary classification dataset using `sklearn.datasets.make_moons` with at least 1000 samples, `noise=0.1`, and `random_state=42`.
2. Split the dataset into training and testing sets (e.g., 80/20 split) using `sklearn.model_selection.train_test_split`.
3. Build a simple feedforward neural network using `tf.keras.Sequential`:
    *   An input `tf.keras.layers.Dense` layer suitable for the number of features.
    *   A hidden `Dense` layer with 32 units and `relu` activation.
    *   Another hidden `Dense` layer with 16 units and `relu` activation.
    *   An output `Dense` layer with 1 unit and `sigmoid` activation.
4. Compile the model with `optimizer='adam'`, `loss='binary_crossentropy'`, and `metrics=['accuracy']`.
5. Train the model on the training data for a fixed number of epochs (e.g., 50) with a batch size (e.g., 32), storing the training history.
6. Evaluate the trained model on the test set and print the test accuracy.
7. Plot the training and validation accuracy and loss over epochs from the training history using `matplotlib.pyplot`, clearly labeling the axes and providing a title.

## Focus
basic AI experimentation

## Dataset
synthetic binary classification data (`make_moons`)

## Hint
Ensure your input layer `input_shape` matches your feature dimension. Remember to import `tensorflow` and `matplotlib.pyplot`.
