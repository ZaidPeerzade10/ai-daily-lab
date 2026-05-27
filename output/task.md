# AI Daily Lab — 2026-05-27

## Task
Develop a machine learning pipeline to predict the **sentiment category** ('Negative', 'Neutral', 'Positive') of a customer product review, based on the review text, product attributes, and the customer's historical rating behavior.

## Focus
Multi-class Classification, Text Feature Engineering (TF-IDF), SQL Aggregations (All-time), ML Pipeline Integration.

## Dataset
1.  **Synthetic Data Generation (Pandas/Numpy)**: Create two pandas DataFrames:
    *   `customers_df`: With 1000-1500 rows. Columns: `customer_id` (unique integers), `signup_date` (random dates over the last 3-5 years), `loyalty_status` (e.g., 'Bronze', 'Silver', 'Gold').
    *   `products_df`: With 200-300 rows. Columns: `product_id` (unique integers), `product_name` (unique strings), `category` (e.g., 'Electronics', 'Books', 'Clothing', 'Home Goods'), `price_usd` (random floats 10-1000).
    *   `reviews_df`: With 15000-25000 rows. Columns: `review_id` (unique integers), `customer_id` (randomly sampled from `customers_df` IDs), `product_id` (randomly sampled from `products_df` IDs), `review_date` (random datetimes over the last 2 years), `rating` (random integers 1-5), `review_text` (text strings).
    *   **Simulate realistic patterns**: Ensure `review_date` is always after `signup_date`. Generate `review_text` that correlates with `rating`: for `rating` 1-2, include negative keywords (e.g., 'bad', 'disappointing', 'terrible'); for `rating` 3, include neutral keywords (e.g., 'okay', 'decent', 'average'); for `rating` 4-5, include positive keywords (e.g., 'great', 'excellent', 'love it'). Mix with generic words. 'Gold' customers might give slightly higher average ratings. Certain `category` products might have more extreme ratings.
    *   Sort `reviews_df` by `customer_id` then `review_date`.

2.  **Load into SQLite & SQL Feature Engineering (All-time Aggregations)**: Create an in-memory SQLite database using `sqlite3`. Load `customers_df`, `products_df`, and `reviews_df` into tables named `customers`, `products`, and `reviews` respectively.
    *   Write a single SQL query that performs the following for *each review*:
        *   Joins `reviews` with `customers` and `products`.
        *   Aggregates *all-time historical features* for the respective `customer_id` and `product_id` *up to (but not including) the current review's `review_date`*. If no prior reviews, use default values:
            *   `customer_avg_rating_prev`: Average `rating` for this `customer_id` from their *previous reviews*.
            *   `customer_num_reviews_prev`: Count of *previous reviews* for this `customer_id`.
            *   `product_avg_rating_all_time`: Average `rating` for this `product_id` from *all its reviews*.
            *   `product_num_reviews_all_time`: Count of *all reviews* for this `product_id`.
        *   Includes static attributes: `review_id`, `review_text`, `rating` (target for now), `loyalty_status`, `category`, `price_usd`.
    *   **Ensures** all reviews are included (using `LEFT JOIN`). Handle `NULL`s for historical aggregates (e.g., 0.0 for averages, 0 for counts if no prior activity).
    *   The query should return all mentioned fields.
    *   **Hint**: Use CTEs for pre-calculating customer/product historical aggregates. For 'previous reviews' for a customer, you'll need a window function (`AVG(...) OVER (PARTITION BY customer_id ORDER BY review_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)`) or a self-join with date comparison. For this 45-min task, simplify 'previous reviews' to 'all prior reviews up to the current review_date', and `product_avg_rating_all_time` to just the *overall* average rating for that product (pre-calculated once).

3.  **Pandas Feature Engineering & Multi-class Target Creation**: Fetch the SQL query results into a pandas DataFrame (`review_features_df`).
    *   Convert `review_date` to datetime objects.
    *   Handle `NaN` values: Fill numerical historical aggregates (e.g., averages, counts) with 0.0 or 0 as appropriate.
    *   Calculate `review_text_length`: Length of the `review_text`.
    *   **Create the Multi-class Target `sentiment_category`**: Based on the original `rating`:
        *   'Negative': `rating` <= 2
        *   'Neutral': `rating` == 3
        *   'Positive': `rating` >= 4
    *   Define features `X` (numerical: `price_usd`, `customer_avg_rating_prev`, `customer_num_reviews_prev`, `product_avg_rating_all_time`, `product_num_reviews_all_time`, `review_text_length`; categorical: `loyalty_status`, `category`; text: `review_text`) and target `y` (`sentiment_category`).
    *   Split into training and testing sets (e.g., 70/30 split) using `sklearn.model_selection.train_test_split` (set `random_state=42`, `stratify` on `y` for class balance).

4.  **Data Visualization (Matplotlib/Seaborn)**: Create two separate plots to visually inspect relationships with `sentiment_category`:
    *   A violin plot (or box plot) showing the distribution of `review_text_length` for each `sentiment_category`. Ensure appropriate labels and titles.
    *   A stacked bar chart showing the proportion of `sentiment_category` across different `category` values. Ensure appropriate labels and titles.

5.  **ML Pipeline & Evaluation (Multi-class Classification with Text Features)**:
    *   Create an `sklearn.pipeline.Pipeline` with a `sklearn.compose.ColumnTransformer` for preprocessing:
        *   For numerical features: Apply `sklearn.preprocessing.SimpleImputer(strategy='mean')` followed by `sklearn.preprocessing.StandardScaler`.
        *   For categorical features: Apply `sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')`.
        *   **For the text feature (`review_text`): Apply `sklearn.feature_extraction.text.TfidfVectorizer(max_features=1000)` (adjust `max_features` if needed). Use a `FunctionTransformer` to select the text column for the `TfidfVectorizer` within the `ColumnTransformer`.**
    *   The final estimator in the pipeline should be `sklearn.ensemble.HistGradientBoostingClassifier` (set `random_state=42`).
    *   Train the pipeline on `X_train`, `y_train`. Predict the `sentiment_category` on the test set (`X_test`).
    *   Calculate and print a `sklearn.metrics.classification_report` for the test set predictions.

## Hint
When performing SQL aggregations for 'previous reviews' for a customer, it is simpler to calculate *all-time* aggregates per customer and product first, then join them to the reviews table. For `TfidfVectorizer` within `ColumnTransformer`, remember that `ColumnTransformer` expects array-like inputs, so `FunctionTransformer(lambda x: x.squeeze(), accept_sparse=True)` or `FunctionTransformer(lambda x: x.values.astype(str))` can be used to correctly pass the single text column to the vectorizer. Ensure `TfidfVectorizer` gets string input.
