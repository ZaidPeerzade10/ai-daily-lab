# AI Daily Lab — 2026-01-23

## Task
1. **Generate Synthetic Data**: Create two pandas DataFrames:
    *   `products_df`: 100-150 rows. Columns: `product_id` (unique int), `category` (e.g., 'Electronics', 'Books', 'Home Goods', 'Clothing' with varying proportions), `price` (random floats between 50.0 and 500.0).
    *   `reviews_df`: 800-1200 rows. Columns: `review_id` (unique int), `product_id` (randomly sampled from `products_df` IDs, ensuring multiple reviews per product), `review_text` (short text containing keywords like 'good', 'great', 'excellent', 'bad', 'terrible', 'slow', and '!' randomly, mixed with generic words), `rating` (random integers 1-5, biased towards 4-5).

2. **Load into SQLite & SQL Analytics**: Create an in-memory SQLite database using `sqlite3`. Load `products_df` into a table named `products` and `reviews_df` into a table named `reviews`. Write a single SQL query that performs the following:
    *   **Joins** `reviews` and `products` tables on `product_id`.
    *   **Calculates `avg_product_rating`**: For each `review_id`, include the average `rating` for its corresponding `product_id` across *all* reviews for that product (using a window function `AVG(rating) OVER (PARTITION BY product_id)`).
    *   The query should return `review_id`, `review_text`, `rating`, `category`, `price`, and `avg_product_rating`.

3. **Pandas Feature Engineering & Target Creation**: Fetch the SQL query results into a pandas DataFrame. Create the following new features:
    *   **Target Variable `sentiment`**: Create a binary target column `sentiment` where 1 if `rating >= 4` (positive) and 0 if `rating < 4` (negative). Drop the original `rating` column.
    *   **Text Features from `review_text`**:
        *   `review_length`: The length of the `review_text` string.
        *   `has_exclamation`: Binary (1 if '!' is found in `review_text`, 0 otherwise).
        *   `has_positive_word`: Binary (1 if 'good', 'great', or 'excellent' (case-insensitive) is found, 0 otherwise).
        *   `has_negative_word`: Binary (1 if 'bad', 'terrible', or 'slow' (case-insensitive) is found, 0 otherwise).

4. **Data Visualization**: Create two visualizations:
    *   A box plot showing the distribution of `price` for each `sentiment` (0 and 1).
    *   A bar plot (or count plot) showing the distribution of `sentiment` across different `category` values.
    Ensure plots have appropriate labels and titles.

5. **ML Pipeline & Evaluation**: 
    *   Define features `X` (all numerical, categorical, and binary text features created) and target `y` (`sentiment`) from the engineered DataFrame.
    *   Split the data into training and testing sets (e.g., 70/30 split) using `sklearn.model_selection.train_test_split` (set `random_state=42`).
    *   Create an `sklearn.pipeline.Pipeline` with a `ColumnTransformer` for preprocessing:
        *   For numerical features (`price`, `avg_product_rating`, `review_length`, `has_exclamation`): Apply `sklearn.preprocessing.StandardScaler`.
        *   For the categorical feature (`category`): Apply `sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')`.
        *   For binary features (`has_positive_word`, `has_negative_word`): Use `Passthrough`.
    *   The final estimator in the pipeline should be `sklearn.linear_model.LogisticRegression` (set `random_state=42`, `solver='liblinear'` for reproducibility).
    *   Train the pipeline on the training data (`X_train`, `y_train`). Predict `sentiment` for the test set (`X_test`).
    *   Calculate and print the `sklearn.metrics.accuracy_score` and `sklearn.metrics.classification_report` for the test set predictions.

## Focus
SQL analytics, pandas feature engineering (text-based & derived), ML pipelines, classification, data visualization

## Dataset
Synthetic product and review data generated with pandas/numpy.

## Hint
Pay close attention to using the `AVG() OVER (PARTITION BY ...)` window function in SQL. For pandas text features, use `.str.contains()` with `case=False` for case-insensitive matching.
