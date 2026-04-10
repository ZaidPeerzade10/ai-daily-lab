# AI Daily Lab — 2026-04-10

## Task
Develop a machine learning pipeline to predict user churn (inactivity) within the next 30 days, based on their profile and recent activity patterns.

## Focus
Predictive modeling for user churn, leveraging time-series aggregation in SQL and Pandas, followed by a binary classification ML pipeline.

## Dataset
1. **Generate Synthetic Data (Pandas/Numpy)**: Create two pandas DataFrames:
    *   `users_df`: With 500-700 rows. Columns: `user_id` (unique integers), `signup_date` (random dates over the last 3 years), `plan_type` (e.g., 'Free', 'Basic', 'Pro'), `country` (e.g., 'US', 'UK', 'DE', 'FR').
    *   `events_df`: With 15000-25000 rows. Columns: `event_id` (unique integers), `user_id` (randomly sampled from `users_df` IDs), `event_timestamp` (random timestamps occurring *after* their respective `signup_date`), `event_type` (e.g., 'login', 'view_profile', 'create_post', 'like_post', 'settings_update').
    *   **Simulate realistic activity and churn patterns**: Ensure `event_timestamp` is always after `signup_date`. For a subset of users (10-20%), ensure their `event_timestamp`s stop before a certain point (e.g., they have no events in the last 60-90 days of the dataset's overall time range, or their last event is X days after signup). Premium plans ('Basic', 'Pro') should generally have more events and be less likely to churn. Sort `events_df` by `user_id` then `event_timestamp`.

2. **Load into SQLite & SQL Feature Engineering (Recent User Activity)**: Create an in-memory SQLite database using `sqlite3`. Load `users_df` and `events_df` into tables named `users` and `events` respectively. For each user, define their `snapshot_date` as `signup_date + 60 days` (ensure `snapshot_date` is within the range of `events_df` timestamps for analysis; if not, adjust or filter users).
    Write a single SQL query that performs the following for *each user*, aggregating their event behavior *within the 30 days leading up to their snapshot_date* (i.e., `event_timestamp` between `snapshot_date - 30 days` and `snapshot_date`):
    *   **Joins** `users` with an aggregated subquery for `events`.
    *   **Aggregates features based on activities *in the 30 days before snapshot_date***:
        *   `num_events_last_30d` (count of `event_id`s)
        *   `num_distinct_event_types_last_30d` (count of distinct `event_type`s)
        *   `days_with_activity_last_30d` (count of distinct dates from `event_timestamp`)
        *   `num_logins_last_30d` (count of `event_type = 'login'`)
        *   `num_create_posts_last_30d` (count of `event_type = 'create_post'`)
    *   **Includes static user attributes**: `user_id`, `signup_date`, `plan_type`, `country`.
    *   **Ensures** all users are included (using `LEFT JOIN` to the aggregated subquery), showing 0 for counts/sums if no activity in the 30 days before `snapshot_date`.
    *   The query should return `user_id`, `signup_date`, `plan_type`, `country`, and all the aggregated features.
    *   **Hint**: Use `julianday()` for date comparisons. Aggregate event types using `SUM(CASE WHEN event_type = '...' THEN 1 ELSE 0 END)`. Use `strftime('%Y-%m-%d', event_timestamp)` for distinct dates, and `DATE(u.signup_date, '+60 days')` for the `snapshot_date`.

3. **Pandas Feature Engineering & Binary Target Creation (Churn)**: Fetch the SQL query results into a pandas DataFrame (`user_activity_features_df`).
    *   Handle `NaN` values: Fill all aggregated numerical features (e.g., `num_events_last_30d`, `num_distinct_event_types_last_30d`) with 0 or 0.0 as appropriate.
    *   Convert `signup_date` and derived `snapshot_date` (recalculate or retrieve from SQL) to datetime objects. Calculate `days_since_signup_at_snapshot`: Days between `signup_date` and `snapshot_date`.
    *   Calculate `event_frequency_last_30d`: `num_events_last_30d` / 30.0. Fill any `NaN`s with 0.
    *   **Create the Binary Target `is_churned_next_30d`**: For *each user*, determine if they had *any* event in the *original* `events_df` between their `snapshot_date` and `snapshot_date + 30 days`. If there are no events in this future 30-day window, `is_churned_next_30d = 1` (churned), else `0` (not churned). Merge this target with `user_activity_features_df` (left join), filling `NaN`s with 1 for users who might not have any future events due to data simulation limits or actual churn. Aim for a churn rate of 10-20% by design.
    *   Define features `X` (numerical: `num_events_last_30d`, `num_distinct_event_types_last_30d`, `days_with_activity_last_30d`, `num_logins_last_30d`, `num_create_posts_last_30d`, `days_since_signup_at_snapshot`, `event_frequency_last_30d`; categorical: `plan_type`, `country`) and target `y` (`is_churned_next_30d`). Split into training and testing sets (e.g., 70/30 split) using `sklearn.model_selection.train_test_split` (set `random_state=42`, `stratify` on `y` for class balance).

4. **Data Visualization**: Create two separate plots to visually inspect relationships with `is_churned_next_30d`:
    *   A violin plot (or box plot) showing the distribution of `event_frequency_last_30d` for churned (1) vs. non-churned (0) users. Ensure appropriate labels and titles.
    *   A stacked bar chart showing the proportion of `is_churned_next_30d` (0 or 1) across different `plan_type` values. Ensure appropriate labels and titles.

5. **ML Pipeline & Evaluation (Binary Classification)**: 
    *   Create an `sklearn.pipeline.Pipeline` with a `sklearn.compose.ColumnTransformer` for preprocessing:
        *   For numerical features: Apply `sklearn.preprocessing.SimpleImputer(strategy='mean')` followed by `sklearn.preprocessing.StandardScaler`.
        *   For categorical features: Apply `sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')`.
    *   The final estimator in the pipeline should be `sklearn.ensemble.HistGradientBoostingClassifier` (set `random_state=42`).
    *   Train the pipeline on `X_train`, `y_train`. Predict probabilities for the positive class (class 1) on the test set (`X_test`).
    *   Calculate and print the `sklearn.metrics.roc_auc_score` and a `sklearn.metrics.classification_report` for the test set predictions.

## Hint
When simulating churn for `events_df`, for a chosen subset of `user_id`s, determine a 'last_active_date' (e.g., random date after signup but before the dataset's end). Then, filter out any events for those users that would occur after their 'last_active_date'. For the `is_churned_next_30d` target, remember to calculate `snapshot_date + 30 days` and use this range for filtering events from the *original* `events_df`.
