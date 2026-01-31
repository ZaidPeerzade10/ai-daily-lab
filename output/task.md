# AI Daily Lab — 2026-01-31

## Task
1. **Generate Synthetic Data (Pandas/Numpy)**: Create three pandas DataFrames:
    *   `users_df`: With 500-700 rows. Columns: `user_id` (unique integers), `signup_date` (random dates over the last 3-5 years), `region` (e.g., 'North', 'South', 'East', 'West' with random distribution), `device_type` (e.g., 'Mobile', 'Desktop').
    *   `sessions_df`: With 3000-5000 rows. Columns: `session_id` (unique integers), `user_id` (randomly sampled from `users_df` IDs), `start_time` (random dates *after* respective `signup_date`), `end_time` (randomly *after* `start_time`). Calculate `session_duration_minutes` from `start_time` and `end_time`.
    *   `pageviews_df`: With 8000-12000 rows. Columns: `pageview_id` (unique integers), `session_id` (randomly sampled from `sessions_df` IDs), `page_url` (e.g., '/home', '/product/123', '/checkout', '/help/faq', '/about'), `view_time` (randomly *during* its respective `session_id`'s `start_time` and `end_time`).
    *   Ensure data generation reflects varying user activity levels.

2. **Load into SQLite & SQL Feature Engineering**: Create an in-memory SQLite database using `sqlite3`. Load `users_df`, `sessions_df`, and `pageviews_df` into tables. Determine an `analysis_date` (e.g., `max(view_time)` from `pageviews_df` + 7 days, using pandas).
    Write a single SQL query that performs the following for *each user*:
    *   **Joins** `users`, `sessions`, and `pageviews` tables.
    *   **Aggregates** user-level features for activity occurring within the **last 60 days** relative to the `analysis_date`:
        *   `num_sessions_last_60d` (count of sessions)
        *   `total_duration_last_60d` (sum of `session_duration_minutes`)
        *   `num_product_views_last_60d` (count of `page_url`s containing '/product/')
        *   `num_checkout_views_last_60d` (count of `page_url`s exactly '/checkout')
    *   **Ensures** all users are included, showing 0 for counts/sums if no activity in the last 60 days.
    *   The query should return `user_id`, `region`, `device_type`, `signup_date`, and the aggregated features.

3. **Pandas Feature Engineering & Multi-Class Target Creation**: Fetch the SQL query results into a pandas DataFrame.
    *   Merge this aggregated DataFrame with the original `users_df` on `user_id` (using a left join to ensure all users are present).
    *   Handle `NaN` values: Fill `num_sessions_last_60d`, `total_duration_last_60d`, `num_product_views_last_60d`, `num_checkout_views_last_60d` with 0.
    *   Calculate `account_age_days`: Days between `signup_date` and the `analysis_date` (from step 2).
    *   **Create the multi-class target `engagement_level`**: Based on `total_duration_last_60d` and `num_sessions_last_60d`:
        *   'High_Engaged': `total_duration_last_60d` is in the top 33% *and* `num_sessions_last_60d` > 0.
        *   'Low_Engaged': `num_sessions_last_60d` == 0 *or* `total_duration_last_60d` is in the bottom 33%.
        *   'Medium_Engaged': All remaining users.
    *   Define features `X` (all numerical: `account_age_days`, `num_sessions_last_60d`, `total_duration_last_60d`, `num_product_views_last_60d`, `num_checkout_views_last_60d`; categorical: `region`, `device_type`) and target `y` (`engagement_level`). Split into training and testing sets (e.g., 70/30 split) using `sklearn.model_selection.train_test_split` (set `random_state=42`).

4. **Data Visualization**: Create two separate plots to visually inspect relationships with `engagement_level`:
    *   A violin plot (or box plot) showing the distribution of `total_duration_last_60d` for each `engagement_level`.
    *   A stacked bar chart showing the distribution of `engagement_level` across different `device_type`s.
    Ensure plots have appropriate labels and titles.

5. **ML Pipeline & Evaluation (Multi-Class)**: 
    *   Create an `sklearn.pipeline.Pipeline` with a `sklearn.compose.ColumnTransformer` for preprocessing:
        *   For numerical features: Apply `sklearn.preprocessing.StandardScaler`.
        *   For categorical features: Apply `sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')`.
    *   The final estimator in the pipeline should be `sklearn.ensemble.HistGradientBoostingClassifier` (set `random_state=42`).
    *   Train the pipeline on `X_train`, `y_train`. Predict `engagement_level` for `X_test`.
    *   Calculate and print the `sklearn.metrics.accuracy_score` and `sklearn.metrics.classification_report` for the test set predictions.

## Focus
Predicting multi-class user engagement tiers using time-windowed SQL aggregations, Pandas feature engineering, and a robust ML pipeline.

## Dataset
Synthetic dataframes simulating user profiles, session activity with start/end times, and page-level interactions with URLs and view times.

## Hint
When performing time-windowed aggregations in SQLite, remember to convert date strings to a comparable format (e.g., `JULIANDAY()` or `STRFTIME()`) and use date arithmetic (e.g., `JULIANDAY(analysis_date) - JULIANDAY(start_time) <= 60`) in your `WHERE` or `ON` clause to filter activity within the last 60 days.
