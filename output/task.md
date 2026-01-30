# AI Daily Lab — 2026-01-30

## Task
1. **Generate Synthetic Data (Pandas/Numpy)**: Create two pandas DataFrames:
    *   `products_df`: 100-150 rows. Columns: `product_id` (unique int), `category` (e.g., 'Electronics', 'Books', 'Home & Kitchen', 'Clothing'), `price` (random floats between 50.0 and 500.0).
    *   `reviews_df`: 800-1200 rows. Columns: `review_id` (unique int), `product_id` (randomly sampled from `products_df` IDs), `submission_date` (random dates over the last 3 years), `rating` (random integers 1-5, with a bias towards 3-5).
    *   **Crucially**: Synthetically generate `review_text` (short text strings) that **reflects the `rating`**. For example, reviews with `rating` 5-4 should contain positive words ('excellent', 'great', 'loved it'), `rating` 3 should contain neutral words ('ok', 'fine', 'average'), and `rating` 2-1 should contain negative words ('bad', 'terrible', 'broken'). Mix these with generic words and some exclamation marks.

2. **Load into SQLite & SQL Analytics**: Create an in-memory SQLite database using `sqlite3`. Load `products_df` into a table named `products` and `reviews_df` into a table named `reviews`. Write a single SQL query that performs the following:
    *   **Joins** `reviews` and `products` tables on `product_id`.
    *   **Calculates `avg_price_in_category`**: For each `review_id`, include the average `price` for its corresponding `category` across *all* products (using a window function `AVG(price) OVER (PARTITION BY category)`).
    *   The query should return `review_id`, `review_text`, `submission_date`, `category`, `price`, `rating`, and `avg_price_in_category`.

3. **Pandas Feature Engineering & Multi-Class Target Creation**: Fetch the SQL query results into a pandas DataFrame.
    *   **Create Multi-Class Target `sentiment_class`**: Based on the `rating` column:
        *   'Positive' if `rating` >= 4
        *   'Neutral' if `rating` == 3
        *   'Negative' if `rating` <= 2
    *   Drop the original `rating` column.
    *   **Text Features from `review_text`**:
        *   `review_length`: The length of the `review_text` string.
        *   `num_exclamation_marks`: Count the occurrences of '!' in `review_text`.
    *   **Date Feature**: Determine an `analysis_date` (e.g., `max(submission_date)` + 30 days from your generated data). Calculate `days_since_review`: The number of days between `submission_date` and `analysis_date`.

4. **Chronological Data Split**: Define features `X` (all numerical, categorical, and text features created) and target `y` (`sentiment_class`). Convert `submission_date` to datetime and sort the DataFrame by this column. Split the data chronologically, using the last 20% of the data for the test set. Ensure `X_train`, `X_test`, `y_train`, `y_test` are correctly defined.

5. **Data Visualization**: Create two separate plots to visually inspect relationships with `sentiment_class`:
    *   A stacked bar plot showing the distribution of `sentiment_class` across different `category` values.
    *   A violin plot (or box plot) showing the distribution of `days_since_review` for each `sentiment_class`.
    Ensure plots have appropriate labels and titles.

6. **ML Pipeline & Evaluation (Multi-Class with Text Features and Basic Neural Network)**: 
    *   Create an `sklearn.pipeline.Pipeline` with a `sklearn.compose.ColumnTransformer` for preprocessing:
        *   For numerical features (`price`, `avg_price_in_category`, `review_length`, `num_exclamation_marks`, `days_since_review`): Apply `sklearn.preprocessing.StandardScaler`.
        *   For the categorical feature (`category`): Apply `sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')`.
        *   For the text feature (`review_text`): Apply `sklearn.feature_extraction.text.TfidfVectorizer(max_features=1000, stop_words='english')`.
    *   The final estimator in the pipeline should be `sklearn.neural_network.MLPClassifier` (set `random_state=42`, `hidden_layer_sizes=(100, 50)`, `max_iter=300`, `early_stopping=True`, `verbose=False`).
    *   Train the pipeline on `X_train`, `y_train`. Predict `sentiment_class` for `X_test`.
    *   Calculate and print the `sklearn.metrics.accuracy_score` and `sklearn.metrics.classification_report` for the test set predictions.

## Focus
Multi-class sentiment classification, text feature engineering (TF-IDF), SQL window functions, chronological data split, and basic neural networks.

## Dataset
Synthetic product review data with associated product details.

## Hint
When generating `review_text`, create a list of positive, neutral, and negative keywords. Use `np.random.choice` to pick words based on the generated `rating` to ensure consistency. Remember to convert date columns to datetime objects for accurate calculations and chronological splitting.
