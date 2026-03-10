# AI Daily Lab — 2026-03-10

## Task
1. **Generate Synthetic Data (Pandas/Numpy)**: Create three pandas DataFrames:
    *   `users_df`: With 500-700 rows. Columns: `user_id` (unique integers), `signup_date` (random dates over the last 5 years), `age_group` (e.g., '18-24', '25-40', '41-60', '60+'), `preferred_genre` (e.g., 'Tech', 'Science', 'History', 'Lifestyle', 'News').
    *   `content_df`: With 200-300 rows. Columns: `content_id` (unique integers), `genre` (e.g., 'Tech', 'Science', 'History', 'Lifestyle', 'News'), `difficulty` (e.g., 'Beginner', 'Intermediate', 'Advanced'), `avg_read_time_minutes` (random floats 5-60), `upload_date` (random dates over the last 4 years).
    *   `recommendations_df`: With 10000-15000 rows. Columns: `rec_id` (unique integers), `user_id` (randomly sampled from `users_df` IDs), `content_id` (randomly sampled from `content_df` IDs), `rec_date` (random dates occurring *after* `signup_date` of the user, and *after* `upload_date` of the content), `was_clicked` (binary, 0 or 1, representing engagement).
    *   **Simulate realistic engagement patterns**: Ensure `rec_date` is always after `signup_date` and `upload_date`. Bias `was_clicked` (overall 5-10% click rate) such that:
        *   Users are more likely to click on content matching their `preferred_genre`.
        *   Content with `difficulty='Beginner'` might have higher initial click rates for all users.
        *   More recently uploaded content generally gets more clicks.
        *   Some `age_group`s might prefer certain `genre`s or `difficulty` levels.
    *   Sort `recommendations_df` by `user_id` then `rec_date` for easier sequential processing.

2. **Load into SQLite & SQL Feature Engineering (User's Initial Recommendation Behavior)**: Create an in-memory SQLite database using `sqlite3`. Load `users_df`, `content_df`, and `recommendations_df` into tables named `users`, `content`, and `recommendations` respectively. For each user, define their `initial_window_cutoff_date` as `signup_date + 30 days`.
    Write a single SQL query that performs the following for *each user*, aggregating their recommendation behavior *within their first 30 days post-signup* (i.e., before or on `initial_window_cutoff_date`):
    *   **Joins** `users` with `recommendations` and `content` tables.
    *   **Aggregates features based on activities *within the first 30 days* post-signup**:
        *   `num_recs_first_30d` (count of `rec_id`s)
        *   `num_clicks_first_30d` (count of `was_clicked=1`)
        *   `avg_clicked_read_time_first_30d` (average `avg_read_time_minutes` for clicked content)
        *   `num_unique_genres_clicked_first_30d` (count of distinct `genre` for clicked content)
        *   `avg_difficulty_score_first_30d`: Assign 'Beginner'=1, 'Intermediate'=2, 'Advanced'=3, then average this score for clicked content.
        *   `days_since_first_rec_first_30d`: Number of days between `signup_date` and `MIN(rec_date)` for the user's first recommendation (if within the 30-day window).
    *   **Includes static user attributes**: `user_id`, `signup_date`, `age_group`, `preferred_genre`.
    *   **Ensures** all users are included (using `LEFT JOIN`s to aggregated subqueries), showing 0 for counts/sums, 0.0 for averages, and `NULL` for `days_since_first_rec_first_30d` if no recommendations in the first 30 days.
    *   The query should return `user_id`, `signup_date`, `age_group`, `preferred_genre`, and all the aggregated features.
    *   **Hint**: Use `julianday()` for date differences. Filter recommendations based on `r.rec_date BETWEEN u.signup_date AND DATE(u.signup_date, '+30 days')`.

3. **Pandas Feature Engineering & Multi-Class Target Creation (Future Engagement Tier)**: Fetch the SQL query results into a pandas DataFrame (`user_initial_features_df`).
    *   Handle `NaN` values: Fill `num_recs_first_30d`, `num_clicks_first_30d`, `num_unique_genres_clicked_first_30d` with 0. Fill `avg_clicked_read_time_first_30d`, `avg_difficulty_score_first_30d` with 0.0. For `days_since_first_rec_first_30d` (for users with no recommendations in the first 30 days), fill with 30 (representing activity started on day 30, or no activity).
    *   Convert `signup_date` to datetime objects. Calculate `user_account_age_at_analysis_days`: The number of days between `signup_date` and the `global_analysis_date` (e.g., `max(rec_date)` from `recommendations_df` + 60 days, using pandas).
    *   Calculate `click_rate_first_30d`: `num_clicks_first_30d` / (`num_recs_first_30d` if `num_recs_first_30d` > 0 else 1.0).
    *   **Create the Multi-Class Target `future_engagement_tier`**: Calculate `total_future_clicks` (sum of `was_clicked`) for each user from the *original* `recommendations_df` for events occurring *after* their `signup_date + 30 days` and up to `global_analysis_date`. Merge this aggregate with `user_initial_features_df` (left join), filling `NaN`s with 0.
        *   Calculate the 33rd and 66th percentiles for *non-zero* `total_future_clicks`.
        *   Define segments:
            *   'Low_Engagement': `total_future_clicks` == 0.
            *   'Medium_Engagement': `total_future_clicks` > 0 AND `total_future_clicks` <= 33rd percentile.
            *   'High_Engagement': `total_future_clicks` > 33rd percentile AND `total_future_clicks` <= 66th percentile.
            *   'Very_High_Engagement': `total_future_clicks` > 66th percentile.
    *   Define features `X` (all numerical: `user_account_age_at_analysis_days`, `num_recs_first_30d`, `num_clicks_first_30d`, `avg_clicked_read_time_first_30d`, `num_unique_genres_clicked_first_30d`, `avg_difficulty_score_first_30d`, `days_since_first_rec_first_30d`, `click_rate_first_30d`; categorical: `age_group`, `preferred_genre`) and target `y` (`future_engagement_tier`). Split into training and testing sets (e.g., 70/30 split) using `sklearn.model_selection.train_test_split` (set `random_state=42`, `stratify` on `y` for class balance).

4. **Data Visualization**: Create two separate plots to visually inspect relationships with `future_engagement_tier`:
    *   A violin plot (or box plot) showing the distribution of `click_rate_first_30d` for each `future_engagement_tier`.
    *   A stacked bar chart showing the proportion of `future_engagement_tier` across different `age_group` values.
    Ensure plots have appropriate labels and titles.

5. **ML Pipeline & Evaluation (Multi-Class)**: 
    *   Create an `sklearn.pipeline.Pipeline` with a `sklearn.compose.ColumnTransformer` for preprocessing:
        *   For numerical features: Apply `sklearn.preprocessing.SimpleImputer(strategy='mean')` followed by `sklearn.preprocessing.StandardScaler`.
        *   For categorical features: Apply `sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')`.
    *   The final estimator in the pipeline should be `sklearn.ensemble.HistGradientBoostingClassifier` (set `random_state=42`).
    *   Train the pipeline on `X_train`, `y_train`. Predict `future_engagement_tier` for `X_test`.
    *   Calculate and print the `sklearn.metrics.accuracy_score` and a `sklearn.metrics.classification_report` for the test set predictions.

## Focus
Predicting multi-class user-level future engagement tier with content recommendations based on initial 30-day activity, comprehensive feature engineering using SQL and Pandas, and end-to-end ML pipeline.

## Dataset
Synthetic DataFrames: `users_df` (user profiles), `content_df` (content attributes), `recommendations_df` (user-content interaction logs including clicks).

## Hint
Carefully manage date windows for feature aggregation (first 30 days post-signup) and target creation (future clicks after the 30-day window). Remember to convert numerical categorical features (like 'difficulty') into actual numerical values before averaging. Ensure stratification during train-test split for balanced target classes.
