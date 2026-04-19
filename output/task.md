# AI Daily Lab — 2026-04-19

## Task
1. **Generate Synthetic Data (Pandas/Numpy)**: Create two pandas DataFrames:
    *   `users_df`: With 800-1200 rows. Columns: `user_id` (unique integers), `signup_date` (random dates over the last 3 years), `segment` (e.g., 'New User', 'Explorer', 'Power User'), `device_type` (e.g., 'Mobile', 'Desktop', 'Tablet'), `age` (random integers 18-65), `previous_feature_engagement_score` (random floats 0.0-10.0, representing general engagement with existing features).
    *   `feature_events_df`: With 20000-30000 rows. Columns: `event_id` (unique integers), `user_id` (randomly sampled from `users_df` IDs), `event_timestamp` (random timestamps occurring *after* their respective `signup_date`), `feature_name` (e.g., 'Search', 'Upload', 'Share', 'Settings', 'New_Analytics_Dashboard'), `duration_seconds` (random integers 0-600, primarily for active usage events).
    *   **Simulate realistic patterns**: Ensure `event_timestamp` is always after `signup_date`. Define `NEW_FEATURE_LAUNCH_DATE = pd.to_datetime('2023-01-15')`. Events for `feature_name = 'New_Analytics_Dashboard'` must only occur *on or after* `NEW_FEATURE_LAUNCH_DATE`.
        *   Bias `previous_feature_engagement_score` higher for 'Power User' segment.
        *   Users with higher `previous_feature_engagement_score` and 'Desktop' `device_type` should have more events, especially for 'New_Analytics_Dashboard' after its launch. 'Mobile' users might have more 'Search' events.
    *   Sort `feature_events_df` by `user_id` then `event_timestamp`.

2. **Load into SQLite & SQL Feature Engineering (Early Behavior of Existing Features)**: Create an in-memory SQLite database using `sqlite3`. Load `users_df` and `feature_events_df` into tables named `users` and `feature_events` respectively. Define `early_behavior_cutoff_date` as `signup_date + 14 days`.
    Write a single SQL query that performs the following for *each user*, aggregating their event behavior *within their first 14 days post-signup* (i.e., `event_timestamp` before or on `early_behavior_cutoff_date`) and *excluding the new feature* (`New_Analytics_Dashboard`):
    *   **Joins** `users` with an aggregated subquery for `feature_events`.
    *   **Aggregates features based on activities *within the first 14 days* post-signup (for existing features only)**:
        *   `num_events_first_14d` (count of `event_id`s)
        *   `total_duration_first_14d` (sum of `duration_seconds`)
        *   `num_unique_features_first_14d` (count of distinct `feature_name`s)
    *   **Includes static user attributes**: `user_id`, `signup_date`, `segment`, `device_type`, `age`, `previous_feature_engagement_score`.
    *   **Ensures** all users are included (using `LEFT JOIN`), showing 0 for counts/sums if no activity in the first 14 days.
    *   The query should return `user_id`, `signup_date`, `segment`, `device_type`, `age`, `previous_feature_engagement_score`, and all the aggregated features.
    *   **Hint**: Use `julianday()` for date comparisons. Exclude the new feature using `feature_name != 'New_Analytics_Dashboard'` in the WHERE clause.

3. **Pandas Feature Engineering & Binary Target Creation (New Feature Adoption)**: Fetch the SQL query results into a pandas DataFrame (`user_early_features_df`).
    *   Handle `NaN` values: Fill `num_events_first_14d`, `total_duration_first_14d`, `num_unique_features_first_14d` with 0.
    *   Convert `signup_date` to datetime objects. Set `NEW_FEATURE_LAUNCH_DATE = pd.to_datetime('2023-01-15')` consistently.
    *   Calculate `days_from_signup_to_feature_launch`: Number of days between `signup_date` and `NEW_FEATURE_LAUNCH_DATE`. (Ensure `users` who signed up after launch date get appropriate values, or filter if only pre-launch users are relevant).
    *   Calculate `engagement_per_event_first_14d`: `total_duration_first_14d` / (`num_events_first_14d` + 1). Fill `NaN` or `inf` with 0.
    *   **Create the Binary Target `will_adopt_new_feature`**: Define `adoption_window_days = 60`. A user is considered to `adopt` (1) the `New_Analytics_Dashboard` if they have *any* event for this feature (`feature_name = 'New_Analytics_Dashboard'`) in `feature_events_df` between `NEW_FEATURE_LAUNCH_DATE` and `NEW_FEATURE_LAUNCH_DATE + pd.Timedelta(adoption_window_days, 'days')`. Otherwise, `will_adopt_new_feature` is 0.
        *   Merge this aggregate (max of 0/1 indicator) with `user_early_features_df` (left join), filling `NaN`s with 0 for users who did not adopt.
    *   Define features `X` (numerical: `num_events_first_14d`, `total_duration_first_14d`, `num_unique_features_first_14d`, `age`, `previous_feature_engagement_score`, `days_from_signup_to_feature_launch`, `engagement_per_event_first_14d`; categorical: `segment`, `device_type`) and target `y` (`will_adopt_new_feature`). Split into training and testing sets (e.g., 70/30 split) using `sklearn.model_selection.train_test_split` (set `random_state=42`, `stratify` on `y` for class balance).

4. **Data Visualization**: Create two separate plots to visually inspect relationships with `will_adopt_new_feature`:
    *   A violin plot (or box plot) showing the distribution of `previous_feature_engagement_score` for non-adopters (0) vs. adopters (1). Ensure appropriate labels and titles.
    *   A stacked bar chart showing the proportion of `will_adopt_new_feature` (0 or 1) across different `segment` values. Ensure appropriate labels and titles.

5. **ML Pipeline & Evaluation (Binary Classification)**: 
    *   Create an `sklearn.pipeline.Pipeline` with a `sklearn.compose.ColumnTransformer` for preprocessing:
        *   For numerical features: Apply `sklearn.preprocessing.SimpleImputer(strategy='mean')` followed by `sklearn.preprocessing.StandardScaler`.
        *   For categorical features: Apply `sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')`.
    *   The final estimator in the pipeline should be `sklearn.ensemble.HistGradientBoostingClassifier` (set `random_state=42`).
    *   Train the pipeline on `X_train`, `y_train`. Predict probabilities for the positive class (class 1) on the test set (`X_test`).
    *   Calculate and print the `sklearn.metrics.roc_auc_score` and a `sklearn.metrics.classification_report` for the test set predictions.

## Focus
Predicting New Feature Adoption using early user behavior and profile attributes.

## Dataset
Synthetic user profiles and feature usage logs.

## Hint
When calculating `days_from_signup_to_feature_launch`, consider how to handle users who signed up *after* the new feature's launch date. For the `will_adopt_new_feature` target, remember to filter `feature_events_df` for the specific new feature and the defined adoption window before aggregating.
