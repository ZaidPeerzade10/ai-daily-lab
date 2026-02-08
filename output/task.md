# AI Daily Lab — 2026-02-08

## Task
1. **Generate Synthetic Data (Pandas/Numpy)**: Create two pandas DataFrames:
    *   `products_df`: With 100-150 rows. Columns: `product_id` (unique integers), `category` (e.g., 'Electronics', 'Books', 'Clothing', 'HomeGoods'), `price` (random floats between 50.0 and 500.0), `release_date` (random dates over the last 3 years).
    *   `reviews_df`: With 800-1200 rows. Columns: `review_id` (unique integers), `product_id` (randomly sampled from `products_df` IDs), `user_id` (random integers to simulate unique users, 1-200), `review_date` (random dates occurring *after* `release_date`), `rating` (random integers 1-5, biased towards 3-5).
    *   **Crucially**: Synthetically generate `review_text` (short text strings) that **reflects the `rating`**. For example, reviews with `rating` 5-4 should contain positive words ('excellent', 'great', 'loved it', 'high quality'), `rating` 3 should contain neutral words ('ok', 'fine', 'average', 'decent'), and `rating` 2-1 should contain negative words ('bad', 'terrible', 'broken', 'disappointing'). Mix these with generic words.

2. **Load into SQLite & SQL Feature Engineering**: Create an in-memory SQLite database using `sqlite3`. Load `products_df` into a table named `products` and `reviews_df` into a table named `reviews`. Determine a `global_analysis_date` (e.g., `max(review_date)` from `reviews_df` + 30 days, using pandas).
    Write a single SQL query that performs the following for *each product*:
    *   **Joins** `products` and `reviews` tables.
    *   **Aggregates product-level features** from reviews (up to `global_analysis_date`):
        *   `avg_rating` (average of `rating`)
        *   `num_reviews` (count of reviews)
        *   `days_since_last_review`: Number of days between `global_analysis_date` and `MAX(review_date)` for the product.
    *   **Ensures** all products are included (using a `LEFT JOIN`), showing 0 for counts and `NULL` for `avg_rating`, `days_since_last_review` if no reviews.
    *   The query should return `product_id`, `category`, `price`, `release_date`, `avg_rating`, `num_reviews`, `days_since_last_review`, and `GROUP_CONCAT(review_text, ' ') AS concatenated_reviews_text` (to extract text features in pandas).

3. **Pandas Feature Engineering & Binary Target Creation**: Fetch the SQL query results into a pandas DataFrame (`product_features_df`).
    *   Handle `NaN` values: Fill `num_reviews` with 0. For `avg_rating`, fill with a neutral value (e.g., 3.0) for products with no reviews. For `days_since_last_review`, fill with a large sentinel value (e.g., `365 * 5` or 1825 days). Fill `concatenated_reviews_text` with an empty string for products with no reviews.
    *   Convert `release_date` to datetime objects. Calculate `product_age_at_analysis_days`: Days between `release_date` and the `global_analysis_date` (from step 2). Handle division by zero for `product_age_at_analysis_days` if a product was released on the analysis date (or simply use 1 to avoid it).
    *   **Extract Text Features**: From `concatenated_reviews_text`, calculate:
        *   `positive_word_count`: Number of occurrences of pre-defined positive keywords (e.g., 'great', 'excellent', 'loved', 'high quality')
        *   `negative_word_count`: Number of occurrences of pre-defined negative keywords (e.g., 'bad', 'terrible', 'broken', 'disappointing')
    *   Calculate `review_density`: `num_reviews` / (`product_age_at_analysis_days` + 1). Use `+1` to prevent division by zero for newly released products.
    *   **Create the Binary Target `is_successful_product`**: A product is 'successful' (1) if its `avg_rating` is greater than or equal to 4.0 *AND* its `num_reviews` is above the 70th percentile among products with at least one review. Otherwise, 0.
    *   Define features `X` (`category`, `price`, `product_age_at_analysis_days`, `avg_rating`, `num_reviews`, `days_since_last_review`, `positive_word_count`, `negative_word_count`, `review_density`) and target `y` (`is_successful_product`). Split into training and testing sets (e.g., 70/30 split) using `sklearn.model_selection.train_test_split` (set `random_state=42`, `stratify` on `y` for class balance).

4. **Data Visualization**: Create two separate plots to visually inspect relationships with `is_successful_product`:
    *   A violin plot (or box plot) showing the distribution of `avg_rating` for 'successful' vs. 'unsuccessful' products.
    *   A stacked bar chart showing the proportion of `is_successful_product` (0 or 1) across different `category` values.
    Ensure plots have appropriate labels and titles.

5. **ML Pipeline & Evaluation (Binary Classification)**: 
    *   Create an `sklearn.pipeline.Pipeline` with a `sklearn.compose.ColumnTransformer` for preprocessing:
        *   For numerical features (e.g., `price`, `product_age_at_analysis_days`, `avg_rating`, `num_reviews`, `days_since_last_review`, `positive_word_count`, `negative_word_count`, `review_density`): Apply `sklearn.preprocessing.SimpleImputer(strategy='mean')` followed by `sklearn.preprocessing.StandardScaler`.
        *   For the categorical feature (`category`): Apply `sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')`.
    *   The final estimator in the pipeline should be `sklearn.ensemble.GradientBoostingClassifier` (set `random_state=42`, `n_estimators=100`, `learning_rate=0.1`).
    *   Train the pipeline on `X_train`, `y_train`. Predict probabilities for the positive class (class 1) on `X_test`.
    *   Calculate and print the `sklearn.metrics.roc_auc_score` and a `sklearn.metrics.classification_report` for the test set predictions.

## Focus
Product success prediction using aggregated review data and product attributes, featuring custom text feature extraction, temporal analytics, and binary classification.

## Dataset
Synthetic Product and Review data.

## Hint
When generating `review_text`, create lists of positive, neutral, and negative words. Sample from these lists based on the `rating` to ensure realistic correlation. For SQL, use `GROUP_CONCAT` to get all review text for a product, then process keywords in pandas.
