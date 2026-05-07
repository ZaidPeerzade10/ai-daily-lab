# AI Daily Lab — 2026-05-07

## Task
Develop a machine learning pipeline to predict if a new social media post will 'go viral' within 7 days of its creation, based on initial engagement metrics and post attributes.

## Focus
Predictive modeling of content popularity, time-series feature engineering, binary classification.

## Dataset
1. **Generate Synthetic Data (Pandas/Numpy)**: Create two pandas DataFrames:
    *   `posts_df`: With 1000-1500 rows. Columns: `post_id` (unique integers), `user_id` (random integers 1-200, simulating different creators), `post_date` (random dates over the last 1-2 years), `category` (e.g., 'News', 'Humor', 'DIY', 'Tech', 'Fashion'), `num_hashtags` (random integers 0-10), `sentiment_score` (random floats -1.0 to 1.0), `user_follower_count` (random integers 100-100000, for the `user_id` at `post_date`).
    *   `interactions_df`: With 30000-50000 rows. Columns: `interaction_id` (unique integers), `post_id` (randomly sampled from `posts_df` IDs), `interaction_timestamp` (random timestamps *after* their respective `post_date`), `interaction_type` (e.g., 'like', 'comment', 'share', 'view').
    *   **Simulate realistic patterns**: Ensure `interaction_timestamp` is always after `post_date`.
        *   For 5-10% of posts, simulate 'viral' behavior: significantly higher interaction counts (especially 'share' and 'comment') within the first 7 days, fading thereafter.
        *   Posts by users with higher `user_follower_count` should have a higher baseline number of interactions.
        *   Posts with higher `sentiment_score` might attract more 'like' interactions.
    *   Sort `interactions_df` by `post_id` then `interaction_timestamp`.

2. **Load into SQLite & SQL Feature Engineering (Early Engagement)**: Create an in-memory SQLite database using `sqlite3`. Load `posts_df` and `interactions_df` into tables named `posts` and `interactions` respectively.
    *   Define `early_engagement_window_hours` as 24 hours.
    *   Write a single SQL query that performs the following for *each post*, aggregating its interaction behavior *within the first 24 hours post-creation* (i.e., `interaction_timestamp` before `post_date + 24 hours`):
        *   `num_likes_first_24h` (count of `interaction_id`s where `interaction_type` is 'like')
        *   `num_comments_first_24h` (count of `interaction_id`s where `interaction_type` is 'comment')
        *   `num_shares_first_24h` (count of `interaction_id`s where `interaction_type` is 'share')
        *   `total_interactions_first_24h` (total count of `interaction_id`s for all types)
        *   `unique_users_first_24h` (count of distinct `user_id`s who interacted with the post).
    *   **Includes static post attributes**: `post_id`, `post_date`, `category`, `num_hashtags`, `sentiment_score`, `user_follower_count`.
    *   **Ensures** all posts are included (using `LEFT JOIN`), showing 0 for counts if no activity in the 24-hour window.
    *   The query should return `post_id`, `post_date`, `category`, `num_hashtags`, `sentiment_score`, `user_follower_count`, and all aggregated features.
    *   **Hint**: Use `julianday()` for date/time comparisons. Aggregate features using `COALESCE` to handle `NULL`s from `LEFT JOIN`s.

3. **Pandas Feature Engineering & Binary Target Creation (Viral Status)**: Fetch the SQL query results into a pandas DataFrame (`post_early_features_df`).
    *   Handle `NaN` values: Fill `num_likes_first_24h`, `num_comments_first_24h`, `num_shares_first_24h`, `total_interactions_first_24h`, `unique_users_first_24h` with 0.
    *   Convert `post_date` to datetime objects.
    *   Calculate `engagement_rate_first_24h`: `total_interactions_first_24h` / (`user_follower_count` + 1). Fill `NaN` or `inf` with 0.
    *   Calculate `share_comment_ratio_first_24h`: `num_shares_first_24h` / (`num_comments_first_24h` + 1). Fill `NaN` or `inf` with 0.
    *   **Create the Binary Target `will_go_viral`**: Define `viral_window_days = 7`. A post is considered `viral` (1) if its total interactions (e.g., sum of `num_likes`, `num_comments`, `num_shares` from `interactions_df` within `post_date` and `post_date + pd.Timedelta(viral_window_days, 'days')`) for all `interaction_types` except 'view' (to focus on active engagement) exceeds the 90th percentile of all posts' total interactions in this window. Otherwise, `will_go_viral` is 0.
        *   Merge this aggregate (max of 0/1 indicator) with `post_early_features_df` (left join), ensuring all posts have a target.
    *   Define features `X` (numerical: `num_hashtags`, `sentiment_score`, `user_follower_count`, `num_likes_first_24h`, `num_comments_first_24h`, `num_shares_first_24h`, `total_interactions_first_24h`, `unique_users_first_24h`, `engagement_rate_first_24h`, `share_comment_ratio_first_24h`; categorical: `category`) and target `y` (`will_go_viral`). Split into training and testing sets (e.g., 70/30 split) using `sklearn.model_selection.train_test_split` (set `random_state=42`, `stratify` on `y` for class balance).

4. **Data Visualization**: Create two separate plots to visually inspect relationships with `will_go_viral`:
    *   A violin plot (or box plot) showing the distribution of `engagement_rate_first_24h` for non-viral (0) vs. viral (1) posts. Ensure appropriate labels and titles.
    *   A stacked bar chart showing the proportion of `will_go_viral` (0 or 1) across different `category` values. Ensure appropriate labels and titles.

5. **ML Pipeline & Evaluation (Binary Classification)**: 
    *   Create an `sklearn.pipeline.Pipeline` with a `sklearn.compose.ColumnTransformer` for preprocessing:
        *   For numerical features: Apply `sklearn.preprocessing.SimpleImputer(strategy='mean')` followed by `sklearn.preprocessing.StandardScaler`.
        *   For categorical features: Apply `sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')`.
    *   The final estimator in the pipeline should be `sklearn.ensemble.HistGradientBoostingClassifier` (set `random_state=42`).
    *   Train the pipeline on `X_train`, `y_train`. Predict probabilities for the positive class (class 1) on the test set (`X_test`).
    *   Calculate and print the `sklearn.metrics.roc_auc_score` and a `sklearn.metrics.classification_report` for the test set predictions.

## Hint
When creating the `will_go_viral` target, make sure to aggregate *all* interactions within the 7-day window for *each post* first, then calculate the 90th percentile threshold from this aggregated data, and finally apply it to determine the binary target for each post. Remember that 'view' interactions are often passive; focusing on 'like', 'comment', 'share' might better capture 'virality'.
