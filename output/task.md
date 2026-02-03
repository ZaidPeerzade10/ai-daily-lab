# AI Daily Lab — 2026-02-03

## Task
1. **Generate Synthetic Data (Pandas/Numpy)**: Create three pandas DataFrames:
    *   `users_df`: With 500-700 rows. Columns: `user_id` (unique integers), `signup_date` (random dates over the last 3 years), `device_type` (e.g., 'Mobile', 'Desktop', 'Tablet' with random distribution), `country` (e.g., 'USA', 'Canada', 'UK', 'Germany' with random distribution).
    *   `content_df`: With 100-150 rows. Columns: `content_id` (unique integers), `content_type` (e.g., 'Article', 'Video', 'Podcast' with varying proportions), `topic` (e.g., 'Tech', 'Finance', 'Health', 'Lifestyle', 'Entertainment'), `duration_minutes` (random floats between 2.0 and 60.0).
    *   `interactions_df`: With 3000-5000 rows. Columns: `interaction_id` (unique integers), `user_id` (randomly sampled from `users_df` IDs, ensuring some users have many interactions and a few have none), `content_id` (randomly sampled from `content_df` IDs), `interaction_date` (random dates occurring *after* their respective `signup_date`), `interaction_type` (e.g., 'view', 'like', 'share', 'comment', 'bookmark' with varying frequencies).
    *   **Simulate varied engagement**: Ensure the data generation reflects that some users primarily 'view', others 'like' or 'comment' more, and that activity levels vary.

2. **Load into SQLite & SQL Feature Engineering**: Create an in-memory SQLite database using `sqlite3`. Load `users_df`, `content_df`, and `interactions_df` into tables named `users`, `content`, and `interactions` respectively. Determine an `analysis_date` (e.g., `max(interaction_date)` from `interactions_df` + 30 days, using pandas).
    Write a single SQL query that performs the following for *each user*:
    *   **Joins** `users`, `content`, and `interactions` tables.
    *   **Aggregates** user-level features:
        *   `total_interactions` (count of all interactions)
        *   `num_views` (count of 'view' interactions)
        *   `num_likes` (count of 'like' interactions)
        *   `num_comments` (count of 'comment' interactions)
        *   `total_duration_spent` (sum of `content.duration_minutes` for all interactions a user had).
        *   `days_since_last_interaction` (number of days between the `analysis_date` and the user's `MAX(interaction_date)`).
    *   **Ensures** all users are included (using a `LEFT JOIN`), showing 0 for counts/sums and `NULL` for `days_since_last_interaction` if no interactions.
    *   The query should return `user_id`, `country`, `device_type`, `signup_date`, `total_interactions`, `num_views`, `num_likes`, `num_comments`, `total_duration_spent`, `days_since_last_interaction`.

3. **Pandas Feature Engineering & Multi-Class Target Creation**: Fetch the SQL query results into a pandas DataFrame.
    *   Handle `NaN` values: Fill `total_interactions`, `num_views`, `num_likes`, `num_comments`, `total_duration_spent` with 0. For `days_since_last_interaction` (for users with no activities), fill with a large sentinel value (e.g., `365 * 5` or 1825 days).
    *   Convert `signup_date` to datetime. Calculate `account_age_days`: The number of days between `signup_date` and the `analysis_date` (from step 2).
    *   **Create the multi-class target `engagement_segment`**: Based on `total_interactions`, `num_comments`, and `num_likes`. First, calculate the 25th percentile for non-zero `num_comments` and `num_likes`, and the 50th percentile for non-zero `total_interactions`. Then, define segments:
        *   'Heavy_Contributor': If `num_comments` is above the 25th percentile OR `num_likes` is above the 25th percentile.
        *   'Frequent_Viewer': If `total_interactions` is above the 50th percentile AND NOT classified as 'Heavy_Contributor'.
        *   'Casual_Browser': All remaining users.
    *   Define features `X` (`country`, `device_type`, `account_age_days`, `total_interactions`, `num_views`, `num_likes`, `num_comments`, `total_duration_spent`, `days_since_last_interaction`) and target `y` (`engagement_segment`). Split into training and testing sets (e.g., 70/30 split) using `sklearn.model_selection.train_test_split` (set `random_state=42`, `stratify` on `y` for class balance).

4. **Data Visualization**: Create two separate plots to visually inspect relationships with `engagement_segment`:
    *   A violin plot (or box plot) showing the distribution of `total_duration_spent` for each `engagement_segment`.
    *   A stacked bar chart showing the distribution of `engagement_segment` across different `device_type`s.
    Ensure plots have appropriate labels and titles.

5. **ML Pipeline & Evaluation (Multi-Class)**: 
    *   Create an `sklearn.pipeline.Pipeline` with a `sklearn.compose.ColumnTransformer` for preprocessing:
        *   For numerical features (e.g., `account_age_days`, `total_interactions`, `num_views`, `num_likes`, `num_comments`, `total_duration_spent`, `days_since_last_interaction`): Apply `sklearn.preprocessing.SimpleImputer(strategy='mean')` followed by `sklearn.preprocessing.StandardScaler`.
        *   For categorical features (`country`, `device_type`): Apply `sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')`.
    *   The final estimator in the pipeline should be `sklearn.ensemble.RandomForestClassifier` (set `random_state=42`, `n_estimators=100`, `class_weight='balanced'` for potential class imbalance).
    *   Train the pipeline on `X_train`, `y_train`. Predict `engagement_segment` for `X_test`.
    *   Calculate and print the `sklearn.metrics.accuracy_score` and a `sklearn.metrics.classification_report` for the test set predictions.

## Focus
User Content Engagement Segmentation

## Dataset
Synthetic User Content Interaction Data

## Hint
When creating the `engagement_segment` target, remember to calculate quantiles only on non-zero values for relevant features (e.g., `df['col'][df['col'] > 0].quantile(0.25)`) to ensure meaningful thresholds for active users. For the SQL query, use `SUM(CASE WHEN interactions.interaction_type = 'view' THEN 1 ELSE 0 END)` for specific interaction counts.
