# AI Daily Lab — 2026-03-30

## Task
1. **Generate Synthetic Data (Pandas/Numpy)**: Create two pandas DataFrames:
    *   `users_df`: With 500-700 rows. Columns: `user_id` (unique integers), `signup_date` (random dates over the last 5 years), `country` (e.g., 'US', 'UK', 'DE', 'FR'), `subscription_plan` (e.g., 'Free', 'Premium_Monthly', 'Premium_Annual').
    *   `events_df`: With 15000-25000 rows. Columns: `event_id` (unique integers), `user_id` (randomly sampled from `users_df` IDs), `event_timestamp` (random timestamps occurring *after* their respective `signup_date`), `event_type` (e.g., 'app_open', 'post_view', 'post_like', 'comment', 'share', 'profile_update', 'settings_change'), `duration_seconds` (random integers 0-600, primarily for 'app_open' events). Ensure `duration_seconds` is > 0 for 'app_open' events and generally 0 for other event types (or a very small, short duration for 'post_view').
    *   **Simulate realistic engagement patterns**: Ensure `event_timestamp` is always after `signup_date`. Bias data such that:
        *   Users with 'Premium_Monthly' or 'Premium_Annual' plans tend to have higher overall event counts, especially 'comment' and 'share' events.
        *   Users who eventually become 'High_Engagement' (see step 3 for target definition) show higher initial counts of 'post_like', 'comment', 'share', and 'profile_update' events.
        *   Users with 'Free' plans might have more 'app_open' events but fewer deep engagement events initially.
    *   Sort `events_df` by `user_id` then `event_timestamp` for easier sequential processing.

2. **Load into SQLite & SQL Feature Engineering (Early User Engagement)**: Create an in-memory SQLite database using `sqlite3`. Load `users_df` and `events_df` into tables named `users` and `events` respectively. For each user, define their `early_behavior_cutoff_date` as `signup_date + 14 days`.
    Write a single SQL query that performs the following for *each user*, aggregating their event behavior *within their first 14 days post-signup* (i.e., `event_timestamp` before or on `early_behavior_cutoff_date`):
    *   **Joins** `users` with an aggregated subquery for `events`.
    *   **Aggregates features based on activities *within the first 14 days* post-signup**:
        *   `num_events_first_14d` (count of `event_id`s)
        *   `total_app_open_duration_first_14d` (sum of `duration_seconds` where `event_type = 'app_open'`)
        *   `num_likes_first_14d` (count of `event_type = 'post_like'`)
        *   `num_comments_first_14d` (count of `event_type = 'comment'`)
        *   `num_shares_first_14d` (count of `event_type = 'share'`)
        *   `days_with_activity_first_14d` (count of distinct dates from `event_timestamp`)
        *   `has_profile_update_first_14d` (binary: 1 if `event_type = 'profile_update'` exists, else 0)
    *   **Includes static user attributes**: `user_id`, `signup_date`, `country`, `subscription_plan`.
    *   **Ensures** all users are included (using `LEFT JOIN` to the aggregated subquery), showing 0 for counts/sums and 0 for binary flags if no activity in the first 14 days.
    *   The query should return `user_id`, `signup_date`, `country`, `subscription_plan`, and all the aggregated features.
    *   **Hint**: Use `julianday()` for date comparisons. Aggregate event types using `SUM(CASE WHEN event_type = '...' THEN 1 ELSE 0 END)`. Use `strftime('%Y-%m-%d', event_timestamp)` for distinct dates, and `DATE(u.signup_date, '+14 days')` for the cutoff.

3. **Pandas Feature Engineering & Multi-Class Target Creation (Future Engagement Tier)**: Fetch the SQL query results into a pandas DataFrame (`user_early_features_df`).
    *   Handle `NaN` values: Fill `num_events_first_14d`, `total_app_open_duration_first_14d`, `num_likes_first_14d`, `num_comments_first_14d`, `num_shares_first_14d`, `days_with_activity_first_14d`, `has_profile_update_first_14d` with 0.
    *   Convert `signup_date` to datetime objects. Calculate `account_age_at_cutoff_days`: The number of days between `signup_date` and `early_behavior_cutoff_date` (which is always 14 days).
    *   Calculate `event_frequency_first_14d`: `num_events_first_14d` / 14.0. Fill any `NaN`s with 0.
    *   Calculate `engagement_action_ratio_first_14d`: (`num_likes_first_14d` + `num_comments_first_14d` + `num_shares_first_14d`) / (`num_events_first_14d` + 1). Use `+1` to prevent division by zero. Fill any `NaN`s with 0.
    *   **Create the Multi-Class Target `future_engagement_tier`**: Calculate `total_events_after_14d` (count of `event_id`s) for each user from the *original* `events_df` for events occurring *after* their `signup_date + 14 days`. Merge this aggregate with `user_early_features_df` (left join), filling `NaN`s with 0.
        *   Calculate the 25th, 50th, and 75th percentiles for *non-zero* `total_events_after_14d`.
        *   Define segments:
            *   'Inactive': `total_events_after_14d` == 0.
            *   'Low_Engagement': `total_events_after_14d` > 0 AND `total_events_after_14d` <= 25th percentile.
            *   'Medium_Engagement': `total_events_after_14d` > 25th percentile AND `total_events_after_14d` <= 50th percentile.
            *   'High_Engagement': `total_events_after_14d` > 50th percentile AND `total_events_after_14d` <= 75th percentile.
            *   'Very_High_Engagement': `total_events_after_14d` > 75th percentile.
    *   Define features `X` (all numerical: `num_events_first_14d`, `total_app_open_duration_first_14d`, `num_likes_first_14d`, `num_comments_first_14d`, `num_shares_first_14d`, `days_with_activity_first_14d`, `account_age_at_cutoff_days`, `event_frequency_first_14d`, `engagement_action_ratio_first_14d`; categorical: `country`, `subscription_plan`, `has_profile_update_first_14d`) and target `y` (`future_engagement_tier`). Split into training and testing sets (e.g., 70/30 split) using `sklearn.model_selection.train_test_split` (set `random_state=42`, `stratify` on `y` for class balance).

4. **Data Visualization**: Create two separate plots to visually inspect relationships with `future_engagement_tier`:
    *   A violin plot (or box plot) showing the distribution of `num_likes_first_14d` for each `future_engagement_tier`. Ensure appropriate labels and titles.
    *   A stacked bar chart showing the proportion of `future_engagement_tier` across different `subscription_plan` values. Ensure appropriate labels and titles.

5. **ML Pipeline & Evaluation (Multi-Class)**: 
    *   Create an `sklearn.pipeline.Pipeline` with a `sklearn.compose.ColumnTransformer` for preprocessing:
        *   For numerical features: Apply `sklearn.preprocessing.SimpleImputer(strategy='mean')` followed by `sklearn.preprocessing.StandardScaler`.
        *   For categorical features: Apply `sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')`.
    *   The final estimator in the pipeline should be `sklearn.ensemble.HistGradientBoostingClassifier` (set `random_state=42`).
    *   Train the pipeline on `X_train`, `y_train`. Predict `future_engagement_tier` for `X_test`.
    *   Calculate and print the `sklearn.metrics.accuracy_score` and a `sklearn.metrics.classification_report` for the test set predictions.

## Focus
Predicting multi-class future user engagement tier based on early event-level behavioral features and user profile.

## Dataset
Synthetic user profiles and detailed in-app event logs.

## Hint
Pay attention to correct date filtering in SQL for the early behavior window and for calculating the future engagement target. When generating `duration_seconds` for `events_df`, ensure it's non-zero primarily for 'app_open' events and close to zero for others. The multi-class target definition with 5 tiers based on percentiles requires careful `pd.cut` or a custom function application, ensuring the 'Inactive' tier (zero future events) is handled separately before percentile calculation.
