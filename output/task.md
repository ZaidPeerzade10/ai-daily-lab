# AI Daily Lab — 2026-03-06

## Task
1. **Generate Synthetic Data (Pandas/Numpy)**: Create three pandas DataFrames:
    *   `users_df`: With 500-700 rows. Columns: `user_id` (unique integers), `signup_date` (random dates over the last 5 years), `initial_subscription_plan` (e.g., 'Starter', 'Pro', 'Enterprise'), `region` (e.g., 'North', 'South', 'East', 'West'), `has_opted_for_annual_billing` (binary, 0 or 1).
    *   `usage_logs_df`: With 5000-8000 rows. Columns: `log_id` (unique integers), `user_id` (randomly sampled from `users_df` IDs), `log_date` (random dates occurring *after* their respective `signup_date`), `feature_accessed` (e.g., 'Dashboard_View', 'Report_Gen', 'Data_Export', 'API_Access', 'Support_Chat'), `session_duration_minutes` (random floats 1.0-120.0).
    *   `subscription_events_df`: With 1500-2500 rows. Columns: `event_id` (unique integers), `user_id` (randomly sampled from `users_df` IDs), `event_date` (random dates occurring *after* their respective `signup_date`), `event_type` (e.g., 'Subscription_Start', 'Renewal_Success', 'Renewal_Failed_Payment', 'Cancellation', 'Upgrade', 'Downgrade'), `plan_effected` (e.g., 'Pro'), `amount_charged` (random floats 10.0-500.0, mostly NaN for non-charge events).
    *   **Simulate realistic patterns**: Ensure `log_date` and `event_date` are always after `signup_date`. Bias `subscription_events_df` such that users with `has_opted_for_annual_billing=1` have a higher proportion of `Renewal_Success` events. Users experiencing `Renewal_Failed_Payment` or `Cancellation` should show declining `session_duration_minutes` or fewer usage events of 'Report_Gen', 'Data_Export', 'API_Access' in the *30-60 days leading up to their event_date*. 'Enterprise' plan users should show higher usage of 'Data_Export' and 'API_Access'. A user's `initial_subscription_plan` should be reflected in their first `Subscription_Start` event.

2. **Load into SQLite & SQL Feature Engineering (User Pre-Renewal Behavior)**: Create an in-memory SQLite database using `sqlite3`. Load `users_df`, `usage_logs_df`, and `subscription_events_df` into tables named `users`, `usage_logs`, and `subscription_events` respectively. Determine a `global_analysis_date` (e.g., `max(log_date)` from `usage_logs_df` + 90 days, using pandas) and a `feature_cutoff_date` (`global_analysis_date` - 90 days).
    Write a single SQL query that performs the following for *each user*, aggregating their feature usage and subscription events *before* the `feature_cutoff_date`:
    *   **Joins** `users` with aggregated subqueries for `usage_logs` and `subscription_events`.
    *   **Aggregates features based on activity *before* `feature_cutoff_date`**: 
        *   `current_plan_at_cutoff`: The `plan_effected` from the *latest* `subscription_event` for the user *before or on* `feature_cutoff_date`. If no events, use `initial_subscription_plan` from `users_df`.
        *   `total_usage_duration_pre_cutoff` (sum of `session_duration_minutes`)
        *   `num_usage_events_pre_cutoff` (count of `log_id`s)
        *   `avg_session_duration_pre_cutoff` (average `session_duration_minutes`)
        *   `num_api_access_events_pre_cutoff` (count of `feature_accessed = 'API_Access'`)
        *   `num_renewal_failures_pre_cutoff` (count of `event_type = 'Renewal_Failed_Payment'`)
        *   `days_since_last_usage_pre_cutoff`: Number of days between `feature_cutoff_date` and `MAX(log_date)`.
        *   `days_since_last_subscription_event_pre_cutoff`: Number of days between `feature_cutoff_date` and `MAX(event_date)` (for any `subscription_events`).
    *   **Includes static user attributes**: `user_id`, `signup_date`, `region`, `has_opted_for_annual_billing`.
    *   **Ensures** all users are included (using `LEFT JOIN` to aggregated subqueries), showing 0 for counts/sums, 0.0 for averages, and `NULL` for `days_since_last_usage_pre_cutoff`/`days_since_last_subscription_event_pre_cutoff` if no relevant activity before cutoff.
    *   The query should return `user_id`, `signup_date`, `region`, `has_opted_for_annual_billing`, and all the aggregated features.

3. **Pandas Feature Engineering & Binary Target Creation**: Fetch the SQL query results into a pandas DataFrame (`user_renewal_features_df`).
    *   Handle `NaN` values: Fill `total_usage_duration_pre_cutoff`, `num_usage_events_pre_cutoff`, `num_api_access_events_pre_cutoff`, `num_renewal_failures_pre_cutoff` with 0. Fill `avg_session_duration_pre_cutoff` with 0.0. For `days_since_last_usage_pre_cutoff` and `days_since_last_subscription_event_pre_cutoff` (for users with no activities before cutoff), fill with a large sentinel value (e.g., 9999 days).
    *   Convert `signup_date` to datetime objects. Calculate `account_age_at_cutoff_days`: The number of days between `signup_date` and the `feature_cutoff_date`.
    *   Calculate `usage_frequency_pre_cutoff`: `num_usage_events_pre_cutoff` / (`account_age_at_cutoff_days` + 1). Use `+1` to prevent division by zero for very new accounts.
    *   Calculate `avg_daily_duration_pre_cutoff`: `total_usage_duration_pre_cutoff` / (`account_age_at_cutoff_days` + 1).
    *   **Create the Binary Target `will_renew_in_next_90_days`**: For each user, use the `subscription_events_df`. First, determine if the user had an 'Active' subscription *at* `feature_cutoff_date` (i.e., their last event before/on cutoff was not 'Cancellation' or `Renewal_Failed_Payment`, or it was a 'Subscription_Start' or 'Renewal_Success'). Then, if active, assign `1` if there is a `Renewal_Success` event *between* `feature_cutoff_date` and `feature_cutoff_date + timedelta(days=90)`. Otherwise, assign `0`. This requires careful merging and filtering based on the sequence of `subscription_events`. Perform a left merge, filling `NaN`s from the merge with 0.
    *   Define features `X` (all numerical and categorical features engineered) and target `y` (`will_renew_in_next_90_days`). Split into training and testing sets (e.g., 70/30 split) using `sklearn.model_selection.train_test_split` (set `random_state=42`, `stratify` on `y` for class balance).

4. **Data Visualization**: Create two separate plots to visually inspect relationships with `will_renew_in_next_90_days`:
    *   A violin plot (or box plot) showing the distribution of `avg_session_duration_pre_cutoff` for non-renewing (0) vs. renewing (1) users.
    *   A stacked bar chart showing the proportion of `will_renew_in_next_90_days` (0 or 1) across different `current_plan_at_cutoff` values.
    Ensure plots have appropriate labels and titles.

5. **ML Pipeline & Evaluation (Binary Classification)**: 
    *   Create an `sklearn.pipeline.Pipeline` with a `sklearn.compose.ColumnTransformer` for preprocessing:
        *   For numerical features (e.g., `account_age_at_cutoff_days`, `total_usage_duration_pre_cutoff`, `num_usage_events_pre_cutoff`, `avg_session_duration_pre_cutoff`, `num_api_access_events_pre_cutoff`, `num_renewal_failures_pre_cutoff`, `days_since_last_usage_pre_cutoff`, `days_since_last_subscription_event_pre_cutoff`, `usage_frequency_pre_cutoff`, `avg_daily_duration_pre_cutoff`): Apply `sklearn.preprocessing.SimpleImputer(strategy='mean')` followed by `sklearn.preprocessing.StandardScaler`.
        *   For categorical features (`region`, `initial_subscription_plan`, `has_opted_for_annual_billing`, `current_plan_at_cutoff`): Apply `sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')`.
    *   The final estimator in the pipeline should be `sklearn.ensemble.HistGradientBoostingClassifier` (set `random_state=42`).
    *   Train the pipeline on `X_train`, `y_train`. Predict probabilities for the positive class (class 1) on the test set (`X_test`).
    *   Calculate and print the `sklearn.metrics.roc_auc_score` and a `sklearn.metrics.classification_report` for the test set predictions.

## Focus
Predicting user subscription renewal based on historical usage and billing events, with a time-based feature engineering approach.

## Dataset
Synthetic user profiles, detailed feature usage logs, and subscription lifecycle events (start, renewal, failure, cancellation).

## Hint
In SQL, use `strftime('%J', ...)` for Julian day differences to calculate days. For `current_plan_at_cutoff`, consider using a subquery with `ROW_NUMBER()` or `MAX()`/`MIN()` filtered by date. Pandas will be crucial for calculating the `will_renew_in_next_90_days` target by carefully interpreting the sequence of `subscription_events` relative to the `feature_cutoff_date`.
