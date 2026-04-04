# AI Daily Lab — 2026-04-04

## Task
Develop a machine learning pipeline to predict the likelihood of a customer renewing their subscription, based on their subscription plan, region, and early usage patterns within the first 30 days.

## Focus
Early engagement feature engineering, binary classification, and pipeline development.

## Dataset
1. **Generate Synthetic Data (Pandas/Numpy)**: Create two pandas DataFrames:
    *   `subscribers_df`: With 500-700 rows. Columns: `subscriber_id` (unique integers), `signup_date` (random dates over the last 2 years), `plan_type` (e.g., 'Basic', 'Standard', 'Premium'), `region` (e.g., 'North', 'South', 'East', 'West').
    *   `usage_df`: With 15000-25000 rows. Columns: `usage_id` (unique integers), `subscriber_id` (randomly sampled from `subscribers_df` IDs), `event_timestamp` (random timestamps occurring *after* their respective `signup_date`), `activity_type` (e.g., 'stream_content', 'download_item', 'support_chat', 'settings_change'), `duration_minutes` (random integers 1-240, primarily for 'stream_content'/'download_item'; 0 for others).
    *   **Simulate realistic renewal patterns**: Define the target `is_renewed` (binary, 0 or 1) for each *subscriber*. A subscriber `is_renewed=1` if their *early engagement* is high. Overall renewal rate should be 40-60%.
        *   Bias `is_renewed` such that:
            *   'Premium' `plan_type` users are more likely to renew.
            *   Users from certain `region`s might have higher renewal rates.
            *   Users who show higher `total_stream_duration` or `num_downloads` (simulated by having more usage events in general) within their first 30 days are more likely to renew.
            *   Users with fewer `support_chat` events are more likely to renew.
    *   Sort `usage_df` by `subscriber_id` then `event_timestamp` for easier sequential processing.

2. **Load into SQLite & SQL Feature Engineering (Early Subscriber Behavior)**: Create an in-memory SQLite database using `sqlite3`. Load `subscribers_df` and `usage_df` into tables named `subscribers` and `usage` respectively. For each subscriber, define their `first_month_cutoff_date` as `signup_date + 30 days`.
    Write a single SQL query that performs the following for *each subscriber*, aggregating their usage behavior *within their first 30 days post-signup* (i.e., `event_timestamp` before or on `first_month_cutoff_date`):
    *   **Joins** `subscribers` with an aggregated subquery for `usage`.
    *   **Aggregates features based on activities *within the first 30 days* post-signup**:
        *   `num_activities_first_30d` (count of `usage_id`s)
        *   `total_stream_duration_first_30d` (sum of `duration_minutes` where `activity_type = 'stream_content'`)
        *   `num_downloads_first_30d` (count of `activity_type = 'download_item'`)
        *   `num_support_chats_first_30d` (count of `activity_type = 'support_chat'`)
        *   `days_with_activity_first_30d` (count of distinct dates from `event_timestamp`)
    *   **Includes static subscriber attributes**: `subscriber_id`, `signup_date`, `plan_type`, `region`, `is_renewed` (the target).
    *   **Ensures** all subscribers are included (using `LEFT JOIN` to the aggregated subquery), showing 0 for counts/sums and 0 for binary flags if no activity in the first 30 days.
    *   The query should return `subscriber_id`, `signup_date`, `plan_type`, `region`, `is_renewed`, and all the aggregated features.
    *   **Hint**: Use `julianday()` for date comparisons. Aggregate event types using `SUM(CASE WHEN activity_type = '...' THEN duration_minutes ELSE 0 END)`. Use `strftime('%Y-%m-%d', event_timestamp)` for distinct dates, and `DATE(s.signup_date, '+30 days')` for the cutoff.

3. **Pandas Feature Engineering & Binary Target Creation**: Fetch the SQL query results into a pandas DataFrame (`subscriber_early_features_df`).
    *   Handle `NaN` values: Fill `num_activities_first_30d`, `total_stream_duration_first_30d`, `num_downloads_first_30d`, `num_support_chats_first_30d`, `days_with_activity_first_30d` with 0 or 0.0 as appropriate.
    *   Convert `signup_date` to datetime objects.
    *   Calculate `activity_frequency_first_30d`: `num_activities_first_30d` / 30.0. Fill any `NaN`s with 0.
    *   Calculate `engagement_score_composite`: (`total_stream_duration_first_30d` * 0.5) + (`num_downloads_first_30d` * 10) - (`num_support_chats_first_30d` * 20).
    *   Define features `X` (all numerical: `num_activities_first_30d`, `total_stream_duration_first_30d`, `num_downloads_first_30d`, `num_support_chats_first_30d`, `days_with_activity_first_30d`, `activity_frequency_first_30d`, `engagement_score_composite`; categorical: `plan_type`, `region`) and target `y` (`is_renewed`). Split into training and testing sets (e.g., 70/30 split) using `sklearn.model_selection.train_test_split` (set `random_state=42`, `stratify` on `y` for class balance).

4. **Data Visualization**: Create two separate plots to visually inspect relationships with `is_renewed`:
    *   A violin plot (or box plot) showing the distribution of `total_stream_duration_first_30d` for non-renewed (0) vs. renewed (1) subscribers. Ensure appropriate labels and titles.
    *   A stacked bar chart showing the proportion of `is_renewed` (0 or 1) across different `plan_type` values. Ensure appropriate labels and titles.

5. **ML Pipeline & Evaluation (Binary Classification)**: 
    *   Create an `sklearn.pipeline.Pipeline` with a `sklearn.compose.ColumnTransformer` for preprocessing:
        *   For numerical features: Apply `sklearn.preprocessing.SimpleImputer(strategy='mean')` followed by `sklearn.preprocessing.StandardScaler`.
        *   For categorical features: Apply `sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')`.
    *   The final estimator in the pipeline should be `sklearn.ensemble.HistGradientBoostingClassifier` (set `random_state=42`).
    *   Train the pipeline on `X_train`, `y_train`. Predict probabilities for the positive class (class 1) on the test set (`X_test`).
    *   Calculate and print the `sklearn.metrics.roc_auc_score` and a `sklearn.metrics.classification_report` for the test set predictions.

## Hint
When generating synthetic data for `is_renewed`, consider creating a temporary engagement score or using `plan_type` directly to assign `is_renewed` to ensure realistic correlations before generating usage data. In SQL, `SUM(CASE WHEN activity_type = 'stream_content' THEN duration_minutes ELSE 0 END)` is useful for conditional sums. Remember to use `stratify` in `train_test_split` for binary classification tasks with imbalanced classes.
