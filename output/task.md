# AI Daily Lab — 2026-05-16

## Task
Develop a machine learning pipeline to predict the **future engagement level category** ('Low', 'Medium', 'High') for a website user in the next 7 days, based on their demographic profile and historical website activity up to a specific cutoff date.

1.  **Synthetic Data Generation (Pandas/Numpy)**: Create two pandas DataFrames:
    *   `users_df`: With 1000-1500 rows. Columns: `user_id` (unique integers), `signup_date` (random dates over the last 3-5 years), `country` (e.g., 'USA', 'CAN', 'MEX'), `device_type` (e.g., 'Mobile', 'Desktop'), `age_group` (e.g., '18-24', '25-44', '45+').
    *   `activity_df`: With 20000-30000 rows. Columns: `activity_id` (unique integers), `user_id` (randomly sampled from `users_df` IDs), `activity_date` (random datetimes occurring *after* their respective `signup_date` and up to a `current_max_date` like `pd.Timestamp.now() - pd.Timedelta(weeks=1)`), `activity_type` (e.g., 'Page View', 'Login', 'Add to Cart', 'Search', 'Comment').
    *   **Simulate realistic patterns**: Ensure `activity_date` is always after `signup_date`. Simulate varying activity levels: some users are very active, others less so. Mobile users might have more `Page View` activities. Older users might be less active. A small percentage of users could have very high activity. Ensure activity drops off for users who haven't been active recently.
    *   Sort `activity_df` by `user_id` then `activity_date`.

2.  **Load into SQLite & SQL Feature Engineering (Activity Aggregations)**: Create an in-memory SQLite database using `sqlite3`. Load `users_df` and `activity_df` into tables named `users` and `activity` respectively.
    *   Define `GLOBAL_PREDICTION_CUTOFF_DATE` as 7 days prior to the latest `activity_date` in your generated `activity_df` (e.g., `activity_df['activity_date'].max() - pd.Timedelta(days=7)`).
    *   Write a single SQL query that performs the following for *each user*, aggregating their website activity *within the 30 days immediately preceding `GLOBAL_PREDICTION_CUTOFF_DATE`*:
        *   `current_cutoff_date` (the `GLOBAL_PREDICTION_CUTOFF_DATE` itself, for consistency).
        *   `num_logins_prev_30d` (count of `activity_type = 'Login'`).
        *   `num_page_views_prev_30d` (count of `activity_type = 'Page View'`).
        *   `total_activity_count_prev_30d` (total count of all activities).
        *   `days_since_last_activity_at_cutoff`: Number of days between `GLOBAL_PREDICTION_CUTOFF_DATE` and the most recent `activity_date` *before or on* the cutoff. Return a large number (e.g., 9999) if no activities before cutoff.
        *   `num_unique_activity_types_prev_30d` (count of distinct `activity_type`).
    *   **Includes static user attributes**: `user_id`, `signup_date`, `country`, `device_type`, `age_group`.
    *   **Ensures** all users are included (using `LEFT JOIN`), showing 0 for counts/sums if no activity in the 30-day window. Handle `NULL`s appropriately.
    *   The query should return `user_id`, `signup_date`, `country`, `device_type`, `age_group`, `current_cutoff_date`, and all aggregated features.

3.  **Pandas Feature Engineering & Multi-class Target Creation**: Fetch the SQL query results into a pandas DataFrame (`user_features_df`).
    *   Convert `signup_date` and `current_cutoff_date` to datetime objects.
    *   Handle `NaN` values: Fill numerical aggregated features with 0 or 0.0 as appropriate. Fill `days_since_last_activity_at_cutoff` with 9999.
    *   Calculate `user_tenure_at_cutoff_days`: Number of days between `signup_date` and `current_cutoff_date`.
    *   **Create the Multi-class Target `next_7d_engagement_category`**: For *each user*, sum their activities (`activity_df` `activity_type` counts) for all activities that occurred *after* `current_cutoff_date` and *before or on* `current_cutoff_date + pd.Timedelta(days=7)`. Let this sum be `next_7d_activity_count`.
    *   Merge this `next_7d_activity_count` into `user_features_df`, filling `NaN`s with 0 for users with no activity in the target window.
    *   Categorize `next_7d_activity_count` into:
        *   'Low': if `next_7d_activity_count` <= 5
        *   'Medium': if 5 < `next_7d_activity_count` <= 20
        *   'High': if `next_7d_activity_count` > 20
        (Adjust these thresholds based on data distribution if needed).
    *   Define features `X` (numerical: `num_logins_prev_30d`, `num_page_views_prev_30d`, `total_activity_count_prev_30d`, `days_since_last_activity_at_cutoff`, `num_unique_activity_types_prev_30d`, `user_tenure_at_cutoff_days`; categorical: `country`, `device_type`, `age_group`) and target `y` (`next_7d_engagement_category`).
    *   Split into training and testing sets (e.g., 70/30 split) using `sklearn.model_selection.train_test_split` (set `random_state=42`, `stratify` on `y` for class balance).

4.  **Data Visualization (Matplotlib/Seaborn)**: Create two separate plots to visually inspect relationships with `next_7d_engagement_category`:
    *   A violin plot (or box plot) showing the distribution of `total_activity_count_prev_30d` for each `next_7d_engagement_category`. Ensure appropriate labels and titles.
    *   A stacked bar chart showing the proportion of `next_7d_engagement_category` (across 'Low', 'Medium', 'High') for different `device_type` values. Ensure appropriate labels and titles.

5.  **ML Pipeline & Evaluation (Multi-class Classification)**: 
    *   Create an `sklearn.pipeline.Pipeline` with a `sklearn.compose.ColumnTransformer` for preprocessing:
        *   For numerical features: Apply `sklearn.preprocessing.SimpleImputer(strategy='mean')` followed by `sklearn.preprocessing.StandardScaler`.
        *   For categorical features: Apply `sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')`.
    *   The final estimator in the pipeline should be `sklearn.ensemble.HistGradientBoostingClassifier` (set `random_state=42`).
    *   Train the pipeline on `X_train`, `y_train`. Predict the `next_7d_engagement_category` on the test set (`X_test`).
    *   Calculate and print a `sklearn.metrics.classification_report` for the test set predictions.

## Focus
User engagement prediction (Multi-class Classification)

## Dataset
Synthetic website user profiles and activity data

## Hint
When creating the `next_7d_engagement_category` target, use `pd.cut` or custom logic with `np.select` to define 'Low', 'Medium', 'High' based on the `next_7d_activity_count` distribution. Adjust the category thresholds (e.g., 5, 20) after inspecting the distribution of `next_7d_activity_count` to ensure a reasonable balance across classes.
