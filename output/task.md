# AI Daily Lab — 2026-02-26

## Task
Develop a machine learning pipeline to predict long-term user retention based on their initial onboarding behavior in an A/B test setting.

## Focus
SQL aggregation of time-windowed event data, Pandas for complex target creation and feature engineering, A/B test context, binary classification.

## Dataset
1. **Generate Synthetic Data (Pandas/Numpy)**: Create three pandas DataFrames:
    *   `users_df`: With 500-700 rows. Columns: `user_id` (unique integers), `signup_date` (random dates over the last 2-3 years), `assigned_onboarding_variant` (e.g., 'Control', 'Variant_A', 'Variant_B'), `referral_source` (e.g., 'Organic', 'Paid_Ad', 'Referral').
    *   `onboarding_events_df`: With 3000-5000 rows. Columns: `event_id` (unique integers), `user_id` (randomly sampled from `users_df` IDs), `event_date` (random dates occurring *within 7 days* after their respective `signup_date`), `event_type` (e.g., 'step_1_completed', 'profile_filled', 'tutorial_viewed', 'payment_info_added', 'email_verified'), `duration_seconds` (random integers 10-300).
    *   `future_activity_df`: With 2000-3000 rows. Columns: `activity_id` (unique integers), `user_id` (randomly sampled from `users_df` IDs), `activity_date` (random dates occurring *between 30 and 90 days* after their respective `signup_date`), `activity_type` (e.g., 'login', 'feature_use_A', 'feature_use_B', 'content_view').
    *   **Simulate realistic patterns**: Ensure `event_date` and `activity_date` are chronologically consistent with `signup_date`. Simulate that different `assigned_onboarding_variant`s impact onboarding event completion rates and `duration_seconds`. Users who complete more critical onboarding steps (e.g., 'profile_filled', 'payment_info_added') or spend more time on 'tutorial_viewed' should have a higher probability of having records in `future_activity_df`.

2. **Load into SQLite & SQL Feature Engineering (Onboarding Behavior)**: Create an in-memory SQLite database using `sqlite3`. Load `users_df` and `onboarding_events_df` into tables named `users` and `onboarding_events` respectively.
    Write a single SQL query that performs the following for *each user*, aggregating their onboarding event behavior *within the first 7 days* of their `signup_date`:
    *   **Joins** `users` and `onboarding_events` tables.
    *   **Aggregates features based on early onboarding activities**:
        *   `num_onboarding_events`: Count of all onboarding `event_id`s within the 7-day window.
        *   `total_onboarding_duration`: Sum of `duration_seconds` for all onboarding events.
        *   `avg_step_duration`: Average `duration_seconds` per onboarding event.
        *   `num_critical_steps_completed`: Count of `event_type`s 'profile_filled' OR 'payment_info_added' within the window.
        *   `days_to_first_onboarding_event`: Number of days between `signup_date` and `MIN(event_date)` (only for events within the 7-day window). `NULL` if no events.
        *   `onboarding_completion_rate`: `num_critical_steps_completed` / 2.0 (assuming 2 critical steps total, 0.0 if `num_critical_steps_completed` is 0).
    *   **Includes static user attributes**: `user_id`, `signup_date`, `assigned_onboarding_variant`, `referral_source`.
    *   **Ensures** all users from `users_df` are included (using a `LEFT JOIN` to the aggregated subquery), showing 0 for counts/sums, 0.0 for averages/rates, and `NULL` for `days_to_first_onboarding_event` if no onboarding events occurred within the window.
    *   The query should return `user_id`, `signup_date`, `assigned_onboarding_variant`, `referral_source`, and all the aggregated features.
    *   **Hint**: Use `strftime('%J', ...)` for Julian day differences to calculate days in SQLite, then convert to integer days for date differences.

3. **Pandas Feature Engineering & Binary Target Creation**: Fetch the SQL query results into a pandas DataFrame (`user_onboarding_features_df`).
    *   Handle `NaN` values: Fill `num_onboarding_events`, `total_onboarding_duration`, `num_critical_steps_completed` with 0. Fill `avg_step_duration`, `onboarding_completion_rate` with 0.0. For `days_to_first_onboarding_event` (for users with no onboarding activity), fill with a large sentinel value (e.g., 9999 days).
    *   Convert `signup_date` to datetime objects.
    *   **Create the Binary Target `is_retained_90_days`**: A user is considered `is_retained_90_days` (1) if they have *any* entry in the original `future_activity_df` where `activity_date` falls *between `signup_date + 30 days` and `signup_date + 90 days`*. Otherwise, 0. This requires processing `future_activity_df` separately and merging the result.
    *   Define features `X` (numerical: `num_onboarding_events`, `total_onboarding_duration`, `avg_step_duration`, `num_critical_steps_completed`, `days_to_first_onboarding_event`, `onboarding_completion_rate`; categorical: `assigned_onboarding_variant`, `referral_source`) and target `y` (`is_retained_90_days`). Split into training and testing sets (e.g., 70/30 split) using `sklearn.model_selection.train_test_split` (set `random_state=42`, `stratify` on `y` for class balance).

4. **Data Visualization**: Create two separate plots to visually inspect relationships with `is_retained_90_days`:
    *   A stacked bar chart showing the proportion of `is_retained_90_days` (0 or 1) across different `assigned_onboarding_variant` values. Include a clear title like '90-Day Retention Rate by Onboarding Variant'.
    *   A violin plot (or box plot) showing the distribution of `total_onboarding_duration` for users with `is_retained_90_days=0` vs. `is_retained_90_days=1`. Ensure appropriate labels and titles.

5. **ML Pipeline & Evaluation (Binary Classification)**: 
    *   Create an `sklearn.pipeline.Pipeline` with a `sklearn.compose.ColumnTransformer` for preprocessing:
        *   For numerical features (`num_onboarding_events`, `total_onboarding_duration`, `avg_step_duration`, `num_critical_steps_completed`, `days_to_first_onboarding_event`, `onboarding_completion_rate`): Apply `sklearn.preprocessing.SimpleImputer(strategy='mean')` followed by `sklearn.preprocessing.StandardScaler`.
        *   For categorical features (`assigned_onboarding_variant`, `referral_source`): Apply `sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')`.
    *   The final estimator in the pipeline should be `sklearn.linear_model.LogisticRegression` (set `random_state=42`, `solver='liblinear'`, `class_weight='balanced'` for potential class imbalance).
    *   Train the pipeline on `X_train`, `y_train`. Predict probabilities for the positive class (class 1) on the test set (`X_test`).
    *   Calculate and print the `sklearn.metrics.roc_auc_score` and a `sklearn.metrics.classification_report` for the test set predictions.

## Hint
Pay close attention to date comparisons in SQL for `event_date` filtering within the 7-day onboarding window and for `days_to_first_onboarding_event`. For the Pandas target creation, ensure you correctly define the future time window for activity. Remember to handle potential division by zero when calculating `onboarding_completion_rate` if `num_critical_steps_completed` is 0.
