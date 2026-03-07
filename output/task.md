# AI Daily Lab — 2026-03-07

## Task
1. **Generate Synthetic Data (Pandas/Numpy)**: Create two pandas DataFrames:
    *   `users_df`: With 500-700 rows. Columns: `user_id` (unique integers), `signup_date` (random dates over the last 5 years), `referral_source` (e.g., 'Organic', 'Paid_Ad', 'Referral'), `initial_plan` (e.g., 'Free_Tier', 'Trial_Pro'), `signup_device` (e.g., 'Mobile', 'Desktop').
    *   `onboarding_activities_df`: With 5000-8000 rows. Columns: `activity_id` (unique integers), `user_id` (randomly sampled from `users_df` IDs), `activity_date` (random dates occurring *after* `signup_date` but *within the first 45 days* of the user's signup), `activity_type` (e.g., 'Login', 'Profile_Setup', 'Feature_A_Used', 'Feature_B_Used', 'Help_Accessed'), `duration_seconds` (random floats 10-600, mostly for 'Feature_A_Used', 'Feature_B_Used', and significantly lower for 'Login', 'Help_Accessed').
    *   **Simulate realistic patterns**: Ensure `activity_date` is always after `signup_date` and within the 45-day window. Bias `onboarding_activities_df` such that users with `initial_plan='Trial_Pro'` tend to have more `Feature_A_Used` and `Feature_B_Used` events with longer `duration_seconds`. Users from `referral_source='Paid_Ad'` might have more `Login` events but fewer valuable feature usages. Some `activity_type`s (e.g., 'Feature_A_Used', 'Feature_B_Used') should contribute more to "power user" status. Sort `onboarding_activities_df` by `user_id` then `activity_date`.

2. **Load into SQLite & SQL Feature Engineering (User Early Onboarding Metrics)**: Create an in-memory SQLite database using `sqlite3`. Load `users_df` into a table named `users` and `onboarding_activities_df` into a table named `onboarding_activities`.
    Write a single SQL query that performs the following for *each user*, aggregating their activity *within the first 7 days* of their `signup_date`:
    *   **Joins** `users` with aggregated subqueries for `onboarding_activities`.
    *   **Aggregates features based on activity *within the first 7 days post-signup***: 
        *   `num_logins_first_7d` (count of 'Login' `activity_type`)
        *   `num_feature_a_uses_first_7d` (count of 'Feature_A_Used' `activity_type`)
        *   `total_duration_first_7d` (sum of `duration_seconds` for all activities)
        *   `days_with_activity_first_7d` (count of distinct `activity_date`s)
        *   `avg_activity_duration_first_7d` (average `duration_seconds` for *all* activities)
        *   `has_completed_profile_first_7d` (binary: 1 if 'Profile_Setup' `activity_type` exists, else 0)
    *   **Includes static user attributes**: `user_id`, `signup_date`, `referral_source`, `initial_plan`, `signup_device`.
    *   **Ensures** all users are included (using `LEFT JOIN`), showing 0 for counts/sums/binary flags and 0.0 for averages if no activity in the first 7 days.
    *   The query should return `user_id`, `signup_date`, `referral_source`, `initial_plan`, `signup_device`, and all the aggregated features.
    *   **Hint**: Use `julianday()` for date differences. Filter activities based on `julianday(a.activity_date) - julianday(u.signup_date) <= 7` and `julianday(a.activity_date) - julianday(u.signup_date) >= 0` to ensure activities are post-signup and within the 7-day window.

3. **Pandas Feature Engineering & Binary Target Creation (Early Power User)**: Fetch the SQL query results into a pandas DataFrame (`user_onboarding_features_df`).
    *   Handle `NaN` values: Fill `num_logins_first_7d`, `num_feature_a_uses_first_7d`, `total_duration_first_7d`, `days_with_activity_first_7d`, `has_completed_profile_first_7d` with 0. Fill `avg_activity_duration_first_7d` with 0.0.
    *   Convert `signup_date` to datetime objects.
    *   Calculate `onboarding_activity_frequency_first_7d`: `days_with_activity_first_7d` / 7.0 (proportion of active days in the first week). Fill any `NaN`s with 0.
    *   **Create the Binary Target `is_early_power_user`**: For each user, calculate `total_duration_in_first_30d` (sum of `duration_seconds` from the *original* `onboarding_activities_df` for events occurring *within 30 days* of `signup_date`). Merge this aggregate with `user_onboarding_features_df`, ensuring users with no activity in 30 days get `0` for this new column. Calculate the 75th percentile of *non-zero* `total_duration_in_first_30d`. A user is an `is_early_power_user=1` if their `total_duration_in_first_30d` is greater than this percentile, otherwise `0`. Fill any remaining `NaN`s in the target column with 0.
    *   Define features `X` (all numerical: `num_logins_first_7d`, `num_feature_a_uses_first_7d`, `total_duration_first_7d`, `days_with_activity_first_7d`, `avg_activity_duration_first_7d`, `onboarding_activity_frequency_first_7d`; categorical: `referral_source`, `initial_plan`, `signup_device`, `has_completed_profile_first_7d`) and target `y` (`is_early_power_user`). Split into training and testing sets (e.g., 70/30 split) using `sklearn.model_selection.train_test_split` (set `random_state=42`, `stratify` on `y` for class balance).

4. **Data Visualization**: Create two separate plots to visually inspect relationships with `is_early_power_user`:
    *   A violin plot (or box plot) showing the distribution of `total_duration_first_7d` for non-power users (0) vs. power users (1). Ensure appropriate labels and titles.
    *   A stacked bar chart showing the proportion of `is_early_power_user` (0 or 1) across different `initial_plan` values. Ensure appropriate labels and titles.

5. **ML Pipeline & Evaluation (Binary Classification)**: 
    *   Create an `sklearn.pipeline.Pipeline` with a `sklearn.compose.ColumnTransformer` for preprocessing:
        *   For numerical features: Apply `sklearn.preprocessing.SimpleImputer(strategy='mean')` followed by `sklearn.preprocessing.StandardScaler`.
        *   For categorical features: Apply `sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')`.
    *   The final estimator in the pipeline should be `sklearn.ensemble.HistGradientBoostingClassifier` (set `random_state=42`).
    *   Train the pipeline on `X_train`, `y_train`. Predict probabilities for the positive class (class 1) on the test set (`X_test`).
    *   Calculate and print the `sklearn.metrics.roc_auc_score` and a `sklearn.metrics.classification_report` for the test set predictions.

## Focus
Predicting a user's 'power user' status based on early onboarding behavior, combining static user attributes with aggregated initial activity metrics. Emphasizes time-windowed feature engineering from sequential data and binary classification.

## Dataset
Synthetic user signup and initial platform activity data.

## Hint
When generating `onboarding_activities_df`, ensure `activity_date` is within `signup_date + timedelta(days=45)`. For SQL, filter activities using `julianday(a.activity_date) - julianday(u.signup_date) BETWEEN 0 AND 7` for the 7-day window. For the target, calculate each user's total `duration_seconds` within their first 30 days post-signup (from the original `onboarding_activities_df`), then determine the 75th percentile of *non-zero* durations for classification. Ensure `NaN`s are handled carefully in both SQL and Pandas.
