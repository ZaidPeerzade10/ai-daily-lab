# AI Daily Lab — 2026-03-14

## Task
1. **Generate Synthetic Data (Pandas/Numpy)**: Create three pandas DataFrames:
    *   `users_df`: With 500-700 rows. Columns: `user_id` (unique integers), `signup_date` (random dates over the last 5 years), `reputation_score` (random integers 0-100), `account_tier` (e.g., 'Bronze', 'Silver', 'Gold').
    *   `products_df`: With 100-200 rows. Columns: `product_id` (unique integers), `product_category` (e.g., 'Electronics', 'Books', 'Home&Kitchen', 'Apparel'), `price` (random floats 10-1000), `release_date` (random dates over the last 3 years).
    *   `reviews_df`: With 8000-12000 rows. Columns: `review_id` (unique integers), `user_id` (randomly sampled from `users_df` IDs), `product_id` (randomly sampled from `products_df` IDs), `review_date` (random dates occurring *after* `signup_date` and *after* `release_date`), `rating` (random integers 1-5), `review_text` (simulated text strings of varying lengths and sentiments).
    *   **Simulate realistic helpfulness patterns**: Define `is_helpful` (binary, 0 or 1) for each `reviews_df` row, with an overall 10-15% helpfulness rate. Bias `is_helpful` such that:
        *   Users with higher `reputation_score` or 'Gold' `account_tier` are more likely to write helpful reviews.
        *   Longer `review_text` (more descriptive) tends to be more helpful.
        *   Reviews with extreme `rating` (1 or 5) might be less frequently helpful unless combined with long text.
        *   Reviews written closer to the `release_date` of the product tend to be more helpful.
    *   Sort `reviews_df` by `user_id` then `review_date` for easier sequential processing.

2. **Load into SQLite & SQL Feature Engineering (Review-Level Context)**: Create an in-memory SQLite database using `sqlite3`. Load `users_df`, `products_df`, and `reviews_df` into tables named `users`, `products`, and `reviews` respectively.
    Write a single SQL query that performs the following for *each review event* in `reviews`:
    *   **Joins** `users`, `products`, and `reviews` tables.
    *   **Calculates sequential features based on the user's *prior reviews* and the product's *prior reviews* (excluding the current one), relative to the current `review_date`**:
        *   `user_prior_reviews_count`: Count of all *previous* reviews for the same user.
        *   `user_avg_prior_rating`: Average `rating` of *previous* reviews for the same user (0.0 if no prior reviews).
        *   `days_since_last_user_review`: Number of days between the current `review_date` and the user's *most recent prior* `review_date`. If no prior review, use the number of days between `signup_date` and the current `review_date`.
        *   `product_prior_reviews_count`: Count of all *previous* reviews for the same product (across all users).
        *   `product_avg_prior_rating`: Average `rating` of *previous* reviews for the same product (0.0 if no prior reviews).
        *   `days_since_product_first_review`: Number of days between `release_date` and `MIN(review_date)` for the product, if any prior reviews exist. Otherwise `NULL`.
    *   **Includes static user, product, and current review attributes**: `review_id`, `user_id`, `product_id`, `review_date`, `rating`, `review_text`, `is_helpful` (the target), `reputation_score`, `account_tier`, `product_category`, `price`, `signup_date`, `release_date`.
    *   The query should return all these attributes and engineered features. Missing values for prior aggregates/dates should be `NULL`.
    *   **Hint**: Use window functions with `LAG` and `AVG(...) OVER (...)` / `COUNT(...) OVER (...)` over `PARTITION BY user_id ORDER BY review_date` and `PARTITION BY product_id ORDER BY review_date`. Use `julianday()` for date differences.

3. **Pandas Feature Engineering & Binary Target Creation**: Fetch the SQL query results into a pandas DataFrame (`review_features_df`).
    *   Handle `NaN` values: Fill `user_prior_reviews_count`, `product_prior_reviews_count` with 0. Fill `user_avg_prior_rating`, `product_avg_prior_rating` with 0.0. Ensure `days_since_last_user_review` is handled appropriately (SQL should mostly do this; if `NaN`s remain for first reviews, fill with `user_account_age_at_review_days`). For `days_since_product_first_review`, fill with `days_since_product_release_at_review` if `product_prior_reviews_count` is 0, or a large sentinel (e.g., 9999).
    *   Convert all date columns (`signup_date`, `release_date`, `review_date`) to datetime objects.
    *   Calculate `user_account_age_at_review_days`: Days between `signup_date` and `review_date`.
    *   Calculate `days_since_product_release_at_review`: Days between `release_date` and `review_date`.
    *   Extract `review_length_chars`: Length of the `review_text` string.
    *   Calculate `rating_deviation_from_product_mean`: `rating` - `product_avg_prior_rating` (or `rating` - global mean rating if `product_avg_prior_rating` is 0).
    *   Define features `X` (all numerical: `rating`, `price`, `reputation_score`, `user_prior_reviews_count`, `user_avg_prior_rating`, `days_since_last_user_review`, `product_prior_reviews_count`, `product_avg_prior_rating`, `days_since_product_first_review`, `user_account_age_at_review_days`, `days_since_product_release_at_review`, `review_length_chars`, `rating_deviation_from_product_mean`; categorical: `account_tier`, `product_category`) and target `y` (`is_helpful`). Split into training and testing sets (e.g., 70/30 split) using `sklearn.model_selection.train_test_split` (set `random_state=42`, `stratify` on `y` for class balance due to helpfulness rate).

4. **Data Visualization**: Create two separate plots to visually inspect relationships with `is_helpful`:
    *   A violin plot (or box plot) showing the distribution of `review_length_chars` for `is_helpful=0` vs. `is_helpful=1`. Ensure appropriate labels and titles.
    *   A stacked bar chart showing the proportion of `is_helpful` (0 or 1) across different `product_category` values. Ensure appropriate labels and titles.

5. **ML Pipeline & Evaluation (Binary Classification)**: 
    *   Create an `sklearn.pipeline.Pipeline` with a `sklearn.compose.ColumnTransformer` for preprocessing:
        *   For numerical features: Apply `sklearn.preprocessing.SimpleImputer(strategy='mean')` followed by `sklearn.preprocessing.StandardScaler`.
        *   For categorical features: Apply `sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')`.
    *   The final estimator in the pipeline should be `sklearn.ensemble.HistGradientBoostingClassifier` (set `random_state=42`).
    *   Train the pipeline on `X_train`, `y_train`. Predict probabilities for the positive class (class 1) on the test set (`X_test`).
    *   Calculate and print the `sklearn.metrics.roc_auc_score` and a `sklearn.metrics.classification_report` for the test set predictions.

## Focus
Predicting the helpfulness of product reviews using sequential user/product review history and simple text-derived features.

## Dataset
Simulated user profiles, product details, and product reviews including review text.

## Hint
When simulating `review_text`, create a mix of positive, neutral, and negative sentence fragments, and vary the number of sentences to get different `review_length_chars`. For the `is_helpful` target, apply conditions such as: `is_helpful=1` is more likely if `rating` is 4-5 and `review_length_chars` is high, or if `reputation_score` is high. SQLite date differences for `days_since...` features can be calculated using `julianday(current_date) - julianday(previous_date)` or `strftime('%J', current_date) - strftime('%J', previous_date)`.
