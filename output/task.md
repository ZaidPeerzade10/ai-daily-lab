# AI Daily Lab — 2026-02-14

## Task
1. **Generate Synthetic Data (Pandas/Numpy)**: Create three pandas DataFrames:
    *   `users_df`: With 500-700 rows. Columns: `user_id` (unique integers), `signup_date` (random dates over the last 5 years), `age_group` (e.g., '18-25', '26-40', '41-60', '60+'), `country` (e.g., 'USA', 'Germany', 'Japan', 'Brazil'), `device_preference` (e.g., 'Mobile', 'Desktop', 'Tablet').
    *   `content_df`: With 100-150 rows. Columns: `content_id` (unique integers), `content_type` (e.g., 'Article', 'Video', 'Podcast', 'E-book'), `topic` (e.g., 'Tech', 'Finance', 'Health', 'Lifestyle', 'Gaming'), `duration_minutes` (random floats between 5.0 and 120.0).
    *   `interactions_df`: With 5000-8000 rows. Columns: `interaction_id` (unique integers), `user_id` (randomly sampled from `users_df` IDs), `content_id` (randomly sampled from `content_df` IDs), `interaction_date` (random dates occurring *after* their respective `signup_date` and content's hypothetical `release_date`), `interaction_type` (e.g., 'view', 'like', 'share', 'comment'), `time_spent_seconds` (random integers 10-7200, typically less than or equal to `duration_minutes` * 60).
    *   **Simulate realistic patterns**: Ensure `interaction_date` is always after `signup_date`. Generate data such that users in certain `age_group`s or `country`s show a preference for specific `content_type`s or `topic`s. Some users should have many interactions, others few or none. `time_spent_seconds` should correlate with `duration_minutes` for 'view' interactions.

2. **Load into SQLite & SQL Feature Engineering (Early Engagement)**: Create an in-memory SQLite database using `sqlite3`. Load `users_df`, `content_df`, and `interactions_df` into tables named `users`, `content`, and `interactions` respectively. Determine a `global_analysis_date` (e.g., `max(interaction_date)` from `interactions_df` + 60 days, using pandas) and define an `initial_engagement_period_days` (e.g., 60 days).
    Write a single SQL query that performs the following for *each user*, aggregating their activity within their **first `initial_engagement_period_days`** from their `signup_date`:
    *   **Joins** `users`, `content`, and `interactions` tables.
    *   **Aggregates features based on early interactions**: 
        *   `initial_total_interactions`: Count of all interactions.
        *   `initial_total_time_spent`: Sum of `time_spent_seconds`.
        *   `initial_avg_interaction_duration`: Average `time_spent_seconds` per interaction.
        *   `initial_num_articles`, `initial_num_videos`, `initial_num_podcasts`, `initial_num_ebooks`: Count of interactions for each respective `content_type`.
        *   `initial_engagement_score`: A weighted sum, e.g., (`initial_total_interactions` * 0.5) + (`initial_total_time_spent` / 60 * 0.3) + (`SUM(CASE WHEN i.interaction_type = 'comment' THEN 1 ELSE 0 END)` * 2).
    *   **Ensures** all users are included (using a `LEFT JOIN`), showing 0 for counts/sums and `NULL` for averages if no early interactions.
    *   The query should return `user_id`, `age_group`, `country`, `device_preference`, `signup_date`, and all the aggregated initial features.

3. **Pandas Feature Engineering & Multi-Class Target Creation**: Fetch the SQL query results into a pandas DataFrame (`user_early_features_df`).
    *   Handle `NaN` values: Fill `initial_total_interactions`, `initial_total_time_spent`, `initial_num_articles`, `initial_num_videos`, `initial_num_podcasts`, `initial_num_ebooks`, `initial_engagement_score` with 0. Fill `initial_avg_interaction_duration` with 0.0.
    *   Convert `signup_date` to datetime objects. Calculate `account_age_at_analysis_days`: The number of days between `signup_date` and the `global_analysis_date` (from step 2).
    *   **Create the Multi-Class Target `overall_preferred_content_type`**: For each user, determine the `content_type` from `content_df` that they spent the *most total time* (sum of `time_spent_seconds` from `interactions_df`) on *across their entire history up to `global_analysis_date`*. If a user has *no total interaction time*, assign them a special category, e.g., 'No_Preference'. You'll need to aggregate total time spent per user and content type from the original `interactions_df` and `content_df` separately and then determine the max.
    *   Define features `X` (all numerical and categorical features engineered from early engagement and static user info) and target `y` (`overall_preferred_content_type`). Split into training and testing sets (e.g., 70/30 split) using `sklearn.model_selection.train_test_split` (set `random_state=42`, `stratify` on `y` for class balance, as some content types might be less popular).

4. **Data Visualization**: Create two separate plots to visually inspect relationships with `overall_preferred_content_type`:
    *   A violin plot (or box plot) showing the distribution of `initial_total_time_spent` for each `overall_preferred_content_type` group.
    *   A stacked bar chart showing the distribution of `overall_preferred_content_type` across different `age_group` values.
    Ensure plots have appropriate labels and titles.

5. **ML Pipeline & Evaluation (Multi-Class)**: 
    *   Create an `sklearn.pipeline.Pipeline` with a `sklearn.compose.ColumnTransformer` for preprocessing:
        *   For numerical features (e.g., `account_age_at_analysis_days`, `initial_total_interactions`, `initial_total_time_spent`, `initial_avg_interaction_duration`, category-specific counts, `initial_engagement_score`): Apply `sklearn.preprocessing.SimpleImputer(strategy='mean')` followed by `sklearn.preprocessing.StandardScaler`.
        *   For categorical features (`age_group`, `country`, `device_preference`): Apply `sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')`.
    *   The final estimator in the pipeline should be `sklearn.ensemble.RandomForestClassifier` (set `random_state=42`, `n_estimators=100`, `class_weight='balanced'` to handle potential class imbalance).
    *   Train the pipeline on `X_train`, `y_train`. Predict `overall_preferred_content_type` for `X_test`.
    *   Calculate and print the `sklearn.metrics.accuracy_score` and a `sklearn.metrics.classification_report` for the test set predictions.

## Focus
Predicting long-term user content preferences from initial engagement signals using SQL for feature engineering and a multi-class classification pipeline.

## Dataset
Three DataFrames: `users_df` (user demographics), `content_df` (content metadata), `interactions_df` (user interactions with content). Data simulates diverse user preferences and activity levels.

## Hint
When generating `interactions_df`, ensure `time_spent_seconds` is consistent with `duration_minutes` for 'view' interactions. For the SQL query, use `WHERE i.interaction_date BETWEEN u.signup_date AND DATE(u.signup_date, '+' || {initial_engagement_period_days} || ' days')` for early engagement. For the target, aggregate all-time `time_spent_seconds` by user and `content_type`, then use `idxmax()` to find the preferred type, carefully handling users with no interactions. Remember to add the `num_comments` calculation in the SQL query for the `initial_engagement_score` if you choose to include it.
