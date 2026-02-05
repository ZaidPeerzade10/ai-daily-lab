# AI Daily Lab — 2026-02-05

## Task
1. **Generate Synthetic Data (Pandas/Numpy)**: Create three pandas DataFrames:
    *   `users_df`: With 500-700 rows. Columns: `user_id` (unique integers), `signup_date` (random dates over the last 5 years), `age` (random integers 18-70), `country` (e.g., 'USA', 'Canada', 'UK', 'Germany', 'France' with random distribution).
    *   `content_df`: With 100-150 rows. Columns: `content_id` (unique integers), `content_type` (e.g., 'Article', 'Video', 'Podcast' with varying proportions), `topic` (e.g., 'Tech', 'Finance', 'Health', 'Lifestyle', 'Entertainment', 'Science'), `difficulty_level` (random integers 1-5, representing increasing difficulty).
    *   `interactions_df`: With 5000-8000 rows. Columns: `interaction_id` (unique integers), `user_id` (randomly sampled from `users_df` IDs), `content_id` (randomly sampled from `content_df` IDs), `interaction_date` (random dates occurring *after* their respective `signup_date`), `interaction_type` (e.g., 'view', 'like', 'bookmark', 'share', with 'view' being most frequent, and 'bookmark'/'share' being rarer).
    *   **Simulate realistic patterns**: Ensure `interaction_date` is always after `signup_date`. Generate data such that users generally 'view' more than 'like', and 'like' more than 'bookmark' or 'share'. Introduce a subtle correlation where content with higher `difficulty_level` or specific `topic` (e.g., 'Tech', 'Science') has a slightly higher chance of being 'bookmarked' or 'shared' by users.

2. **Load into SQLite & SQL Feature Engineering**: Create an in-memory SQLite database using `sqlite3`. Load `users_df`, `content_df`, and `interactions_df` into tables named `users`, `content`, and `interactions` respectively. Determine a `global_analysis_date` (e.g., `max(interaction_date)` from `interactions_df` + 60 days, using pandas) and a `feature_cutoff_date` (`global_analysis_date` - 90 days).
    Write a single SQL query that performs the following for *each unique (user_id, content_id) pair* that had at least one 'view' interaction *before* the `feature_cutoff_date`:
    *   **Joins** `users`, `content`, and `interactions` tables.
    *   **Aggregates features based on interactions *before* `feature_cutoff_date`**: 
        *   `num_views_on_content_pre_cutoff`: Count of 'view' interactions for *this specific user and this specific content*.
        *   `user_total_likes_pre_cutoff`: Total 'like' interactions by *this user across all content*.
        *   `content_total_views_pre_cutoff`: Total 'view' interactions for *this specific content across all users*.
        *   `days_since_first_view_pre_cutoff`: Number of days between `feature_cutoff_date` and the `MIN(interaction_date)` where `interaction_type = 'view'` for *this user and this content*.
    *   **Includes static user and content attributes**: `age`, `country`, `signup_date`, `content_type`, `topic`, `difficulty_level`.
    *   The query should return `user_id`, `content_id`, `age`, `country`, `signup_date`, `content_type`, `topic`, `difficulty_level`, and all the aggregated features.
    *   **Hint**: Use CTEs for pre-aggregations before joining to the main `UserContentViews` and `users`/`content` tables.

3. **Pandas Feature Engineering & Binary Target Creation**: Fetch the SQL query results into a pandas DataFrame (let's call it `user_content_features_df`).
    *   Convert `signup_date` to datetime objects.
    *   Handle `NaN` values: Ensure all aggregated count/sum features are filled with 0. For `days_since_first_view_pre_cutoff` (if any are NaN, which shouldn't happen if the base is `UserContentViews` but as a safeguard), fill with a large sentinel value, e.g., 9999.
    *   Calculate `account_age_at_cutoff_days`: The number of days between `signup_date` and the `feature_cutoff_date`.
    *   **Create the Binary Target `is_bookmarked_or_shared_future`**: For each `(user_id, content_id)` pair in `user_content_features_df`, determine if there was *any* 'bookmark' or 'share' interaction for that specific pair *between `feature_cutoff_date` and `global_analysis_date`*. Merge this information back into `user_content_features_df`, filling `NaN`s with 0 and converting to integer type.
    *   Define features `X` (all numerical and categorical features engineered) and target `y` (`is_bookmarked_or_shared_future`). Split into training and testing sets (e.g., 70/30 split) using `sklearn.model_selection.train_test_split` (set `random_state=42`, `stratify` on `y` to handle potential class imbalance).

4. **Data Visualization**: Create two separate plots to visually inspect relationships with `is_bookmarked_or_shared_future`:
    *   A violin plot (or box plot) showing the distribution of `difficulty_level` for users who `is_bookmarked_or_shared_future=0` vs. `is_bookmarked_or_shared_future=1`.
    *   A stacked bar chart showing the proportion of `is_bookmarked_or_shared_future` (0 or 1) across different `content_type`s.
    Ensure plots have appropriate labels and titles.

5. **ML Pipeline & Evaluation (Basic AI Experimentation)**: 
    *   Create an `sklearn.pipeline.Pipeline` with a `sklearn.compose.ColumnTransformer` for preprocessing:
        *   For numerical features (e.g., `age`, `account_age_at_cutoff_days`, `num_views_on_content_pre_cutoff`, `user_total_likes_pre_cutoff`, `content_total_views_pre_cutoff`, `days_since_first_view_pre_cutoff`): Apply `sklearn.preprocessing.StandardScaler`.
        *   For categorical features (`country`, `content_type`, `topic`): Apply `sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')`.
        *   For `difficulty_level` (which is numeric but represents categories 1-5): Treat as numerical for scaling if it implies order, or categorical for one-hot encoding if it's nominal. Let's apply `StandardScaler` assuming it's ordinal.
    *   The final estimator in the pipeline should be `sklearn.linear_model.LogisticRegression` (set `random_state=42`, `solver='liblinear'` for reproducibility).
    *   Train the pipeline on `X_train`, `y_train`. Predict probabilities for the positive class (class 1) on the test set (`X_test`).
    *   Calculate and print the `sklearn.metrics.roc_auc_score` and a `sklearn.metrics.classification_report` for the test set predictions.

## Focus
Predicting secondary content engagement (bookmark/share) at a user-content pair level, leveraging prior interactions and content/user attributes, with chronological feature/target split.

## Dataset
Synthetic users, content, and interaction logs.

## Hint
When constructing the SQL query, filter interactions based on `interaction_date < '{feature_cutoff_date_str}'` within CTEs for pre-cutoff aggregations. The base for your main `SELECT` statement should be the `UserContentViews` CTE (or equivalent), then LEFT JOIN `users`, `content`, and other aggregated feature CTEs. For the target creation in Pandas, filter `interactions_df` for `interaction_type` in ['bookmark', 'share'] and `interaction_date` between `feature_cutoff_date` and `global_analysis_date` to identify future engagements.
