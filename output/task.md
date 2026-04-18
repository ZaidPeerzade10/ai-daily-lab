# AI Daily Lab — 2026-04-18

## Task
Develop a machine learning pipeline to predict if a new user will become a 'High Engagement' user in their next 30 days, based on their profile and first 7 days of app session activity.

## Focus
Early User Value Prediction (Binary Classification), Feature Engineering from Session Data, ML Pipeline.

## Dataset
1. **Generate Synthetic Data (Pandas/Numpy)**: Create two pandas DataFrames:
    *   `users_df`: With 700-1000 rows. Columns: `user_id` (unique integers), `signup_date` (random dates over the last 2 years), `device_type` (e.g., 'iOS', 'Android', 'Web'), `region` (e.g., 'North', 'South', 'East', 'West').
    *   `sessions_df`: With 20000-30000 rows. Columns: `session_id` (unique integers), `user_id` (randomly sampled from `users_df` IDs), `session_start_time` (random timestamps occurring *after* their respective `signup_date`), `session_duration_minutes` (random integers 1-60), `num_features_interacted` (random integers 0-10).
    *   **Simulate realistic engagement patterns**: Ensure `session_start_time` is always after `signup_date`. Bias data such that users with 'iOS' devices or from certain `region`s might have slightly longer sessions. Users who eventually become 'High Engagement' (see step 3 for target definition) should show higher `session_duration_minutes` and `num_features_interacted` in their initial sessions.
    *   Sort `sessions_df` by `user_id` then `session_start_time`.

2. **Load into SQLite & SQL Feature Engineering (First 7 Days)**: Create an in-memory SQLite database using `sqlite3`. Load `users_df` and `sessions_df` into tables named `users` and `sessions` respectively. For each user, define their `early_behavior_cutoff_date` as `signup_date + 7 days`.
    Write a single SQL query that performs the following for *each user*, aggregating their session behavior *within their first 7 days post-signup* (i.e., `session_start_time` before or on `early_behavior_cutoff_date`):
    *   **Joins** `users` with an aggregated subquery for `sessions`.
    *   **Aggregates features based on activities *within the first 7 days* post-signup**:
        *   `num_sessions_first_7d` (count of `session_id`s)
        *   `total_duration_first_7d` (sum of `session_duration_minutes`)
        *   `avg_session_duration_first_7d` (average of `session_duration_minutes`)
        *   `num_unique_days_active_first_7d` (count of distinct dates from `session_start_time`)
        *   `avg_features_per_session_first_7d` (average of `num_features_interacted` per session).
    *   **Includes static user attributes**: `user_id`, `signup_date`, `device_type`, `region`.
    *   **Ensures** all users are included (using `LEFT JOIN`), showing 0 for counts/sums, and 0.0 for averages if no activity in the first 7 days.
    *   The query should return `user_id`, `signup_date`, `device_type`, `region`, and all the aggregated features.

3. **Pandas Feature Engineering & Binary Target Creation (Future Engagement)**: Fetch the SQL query results into a pandas DataFrame (`user_early_features_df`).
    *   Handle `NaN` values: Fill `num_sessions_first_7d`, `total_duration_first_7d`, `num_unique_days_active_first_7d` with 0. Fill `avg_session_duration_first_7d`, `avg_features_per_session_first_7d` with 0.0.
    *   Convert `signup_date` to datetime objects.
    *   Calculate `session_frequency_first_7d`: `num_sessions_first_7d` / 7.0. Fill any `NaN`s with 0.
    *   **Create the Binary Target `is_high_engagement_next_30d`**: Calculate `total_duration_next_30d` (sum of `session_duration_minutes`) for each user from the *original* `sessions_df` for sessions occurring *after* their `signup_date + 7 days` AND *before* their `signup_date + 37 days` (i.e., the 30-day period immediately following the early behavior window). Merge this aggregate with `user_early_features_df` (left join), filling `NaN`s with 0 for users with no future sessions.
        *   Define 'High Engagement' users as those whose `total_duration_next_30d` is greater than the 75th percentile of *non-zero* `total_duration_next_30d` values. Create `is_high_engagement_next_30d` (1 if High Engagement, 0 otherwise).
    *   Define features `X` (numerical: `num_sessions_first_7d`, `total_duration_first_7d`, `avg_session_duration_first_7d`, `num_unique_days_active_first_7d`, `avg_features_per_session_first_7d`, `session_frequency_first_7d`; categorical: `device_type`, `region`) and target `y` (`is_high_engagement_next_30d`). Split into training and testing sets (e.g., 70/30 split) using `sklearn.model_selection.train_test_split` (set `random_state=42`, `stratify` on `y` for class balance).

4. **Data Visualization**: Create two separate plots to visually inspect relationships with `is_high_engagement_next_30d`:
    *   A violin plot (or box plot) showing the distribution of `total_duration_first_7d` for non-high-engagement (0) vs. high-engagement (1) users. Ensure appropriate labels and titles.
    *   A stacked bar chart showing the proportion of `is_high_engagement_next_30d` (0 or 1) across different `device_type` values. Ensure appropriate labels and titles.

5. **ML Pipeline & Evaluation (Binary Classification)**: 
    *   Create an `sklearn.pipeline.Pipeline` with a `sklearn.compose.ColumnTransformer` for preprocessing:
        *   For numerical features: Apply `sklearn.preprocessing.SimpleImputer(strategy='mean')` followed by `sklearn.preprocessing.StandardScaler`.
        *   For categorical features: Apply `sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')`.
    *   The final estimator in the pipeline should be `sklearn.ensemble.HistGradientBoostingClassifier` (set `random_state=42`).
    *   Train the pipeline on `X_train`, `y_train`. Predict probabilities for the positive class (class 1) on the test set (`X_test`).
    *   Calculate and print the `sklearn.metrics.roc_auc_score` and a `sklearn.metrics.classification_report` for the test set predictions.

## Hint
For SQL date arithmetic in SQLite: `DATE(u.signup_date, '+7 days')` for cutoff. Use `strftime('%Y-%m-%d', session_start_time)` for distinct dates. For Pandas target creation, ensure to calculate percentiles only on non-zero future engagement to distinguish genuinely low engagement from users who simply stopped using the app.
