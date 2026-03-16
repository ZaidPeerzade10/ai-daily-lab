# AI Daily Lab — 2026-03-16

## Task
1. **Generate Synthetic Data (Pandas/Numpy)**: Create three pandas DataFrames:
    *   `users_df`: With 500-700 rows. Columns: `user_id` (unique integers), `signup_date` (random dates over the last 5 years), `acquisition_channel` (e.g., 'Organic', 'Paid_Social', 'Referral', 'Direct'), `device_type` (e.g., 'Mobile', 'Desktop', 'Tablet').
    *   `sessions_df`: With 8000-12000 rows. Columns: `session_id` (unique integers), `user_id` (randomly sampled from `users_df` IDs), `session_start_time` (random timestamps occurring *after* their respective `signup_date`), `session_duration_seconds` (random integers 30-1800), `num_page_views` (random integers 1-50).
    *   `page_views_df`: With 25000-40000 rows. Columns: `page_view_id` (unique integers), `session_id` (randomly sampled from `sessions_df` IDs), `page_url_category` (e.g., 'Homepage', 'Product_Page', 'Category_Page', 'Cart_Page', 'Checkout_Page', 'Help_Page', 'Blog'), `view_time_seconds` (random integers 5-300).
    *   **Simulate realistic retention patterns**: Ensure `session_start_time` is after `signup_date`. Bias the data such that users who eventually get 'retained' (see step 3 for target definition) generally exhibit:
        *   Higher `session_duration_seconds` and `num_page_views` in their initial sessions.
        *   More frequent viewing of 'Product_Page' or 'Cart_Page' in their initial sessions.
        *   Fewer 'Help_Page' visits for longer durations, or fewer 'Homepage' visits with no further navigation in their initial sessions.
        *   Certain `acquisition_channel`s (e.g., 'Referral') or `device_type`s (e.g., 'Desktop') might have higher initial engagement leading to retention.
    *   Sort `sessions_df` by `user_id` then `session_start_time` and `page_views_df` by `session_id` then `view_time_seconds` for easier sequential processing.

2. **Load into SQLite & SQL Feature Engineering (User Initial Web Behavior)**: Create an in-memory SQLite database using `sqlite3`. Load `users_df`, `sessions_df`, and `page_views_df` into tables named `users`, `sessions`, and `page_views` respectively. For each user, define their `initial_behavior_cutoff_date` as `signup_date + 7 days`.
    Write a single SQL query that performs the following for *each user*, aggregating their session and page view behavior *within their first 7 days post-signup* (i.e., `session_start_time` before or on `initial_behavior_cutoff_date`):
    *   **Joins** `users` with aggregated subqueries for `sessions` and `page_views` (via `sessions`).
    *   **Aggregates features based on activities *within the first 7 days* post-signup**:
        *   `num_sessions_first_7d` (count of `session_id`s)
        *   `total_session_duration_first_7d` (sum of `session_duration_seconds`)
        *   `avg_session_duration_first_7d` (average `session_duration_seconds`)
        *   `total_page_views_first_7d` (sum of `num_page_views` from `sessions`, or total count of `page_view_id`s from `page_views`)
        *   `num_product_page_views_first_7d` (count of `page_url_category = 'Product_Page'`)
        *   `num_cart_page_views_first_7d` (count of `page_url_category = 'Cart_Page'`)
        *   `days_with_activity_first_7d` (count of distinct dates from `session_start_time`)
        *   `avg_page_views_per_session_first_7d` (average `num_page_views` from `sessions`)
    *   **Includes static user attributes**: `user_id`, `signup_date`, `acquisition_channel`, `device_type`.
    *   **Ensures** all users are included (using `LEFT JOIN`s to aggregated subqueries), showing 0 for counts/sums, and 0.0 for averages if no activity in the first 7 days.
    *   The query should return `user_id`, `signup_date`, `acquisition_channel`, `device_type`, and all the aggregated features.
    *   **Hint**: Use `julianday()` for date differences for filtering. For `days_with_activity_first_7d`, extract the date part of `session_start_time`.

3. **Pandas Feature Engineering & Binary Target Creation (User Retention)**: Fetch the SQL query results into a pandas DataFrame (`user_initial_features_df`).
    *   Handle `NaN` values: Fill `num_sessions_first_7d`, `total_session_duration_first_7d`, `total_page_views_first_7d`, `num_product_page_views_first_7d`, `num_cart_page_views_first_7d`, `days_with_activity_first_7d` with 0. Fill `avg_session_duration_first_7d` and `avg_page_views_per_session_first_7d` with 0.0.
    *   Convert `signup_date` to datetime objects. Calculate `account_age_at_cutoff_days`: The number of days between `signup_date` and the `initial_behavior_cutoff_date` (which is always 7 days).
    *   Calculate `engagement_ratio_first_7d`: (`num_product_page_views_first_7d` + `num_cart_page_views_first_7d`) / (`total_page_views_first_7d` + 1). Use `+1` to prevent division by zero.
    *   Calculate `session_frequency_first_7d`: `num_sessions_first_7d` / 7.0.
    *   **Create the Binary Target `is_retained_after_30_days`**: For each user, define their `retention_start_date` as `signup_date + 30 days` and `retention_end_date` as `signup_date + 90 days`. Check the *original* `sessions_df` for any session where `session_start_time` falls *between* `retention_start_date` and `retention_end_date`. Assign `1` if at least one such session exists, otherwise `0`. Perform a left merge, filling `NaN`s from the merge with 0.
    *   Define features `X` (all numerical: `num_sessions_first_7d`, `total_session_duration_first_7d`, `avg_session_duration_first_7d`, `total_page_views_first_7d`, `num_product_page_views_first_7d`, `num_cart_page_views_first_7d`, `days_with_activity_first_7d`, `avg_page_views_per_session_first_7d`, `account_age_at_cutoff_days`, `engagement_ratio_first_7d`, `session_frequency_first_7d`; categorical: `acquisition_channel`, `device_type`) and target `y` (`is_retained_after_30_days`). Split into training and testing sets (e.g., 70/30 split) using `sklearn.model_selection.train_test_split` (set `random_state=42`, `stratify` on `y` for class balance due to potential retention rate).

4. **Data Visualization**: Create two separate plots to visually inspect relationships with `is_retained_after_30_days`:
    *   A violin plot (or box plot) showing the distribution of `total_session_duration_first_7d` for non-retained (0) vs. retained (1) users. Ensure appropriate labels and titles.
    *   A stacked bar chart showing the proportion of `is_retained_after_30_days` (0 or 1) across different `acquisition_channel` values. Ensure appropriate labels and titles.

5. **ML Pipeline & Evaluation (Binary Classification)**: 
    *   Create an `sklearn.pipeline.Pipeline` with a `sklearn.compose.ColumnTransformer` for preprocessing:
        *   For numerical features: Apply `sklearn.preprocessing.SimpleImputer(strategy='mean')` followed by `sklearn.preprocessing.StandardScaler`.
        *   For categorical features: Apply `sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')`.
    *   The final estimator in the pipeline should be `sklearn.ensemble.HistGradientBoostingClassifier` (set `random_state=42`).
    *   Train the pipeline on `X_train`, `y_train`. Predict probabilities for the positive class (class 1) on the test set (`X_test`).
    *   Calculate and print the `sklearn.metrics.roc_auc_score` and a `sklearn.metrics.classification_report` for the test set predictions.

## Focus
Predicting user retention based on early website session and page view behavior using a time-windowed feature engineering approach.

## Dataset
Synthetic user profiles, website sessions, and granular page view logs.

## Hint
For SQL aggregates, remember to filter both the `sessions` and `page_views` subqueries based on the `initial_behavior_cutoff_date` for each user. When creating the `is_retained_after_30_days` target in Pandas, ensure you correctly join/merge the aggregated `sessions_df` information for the retention window back to your `user_initial_features_df`.
