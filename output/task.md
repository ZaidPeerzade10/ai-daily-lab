# AI Daily Lab — 2026-04-02

## Task
Develop a machine learning pipeline to predict the category of a user's next content interaction based on their prior behavior and profile.

## Focus
Predicting the next content category (multi-class classification) using sequential interaction features and user/content attributes.

## Dataset
1. **Generate Synthetic Data (Pandas/Numpy)**: Create three pandas DataFrames:
    *   `users_df`: With 500-700 rows. Columns: `user_id` (unique integers), `signup_date` (random dates over the last 5 years), `age_group` (e.g., '18-24', '25-34', '35-49', '50+'), `premium_status` (e.g., 'Free', 'Basic', 'Premium').
    *   `content_items_df`: With 100-150 rows. Columns: `content_id` (unique integers), `category` (e.g., 'Video', 'Article', 'Quiz', 'Forum', 'Ebook'), `difficulty` (e.g., 'Beginner', 'Intermediate', 'Advanced'), `avg_rating` (random floats 2.0-5.0).
    *   `interactions_df`: With 10000-15000 rows. Columns: `interaction_id` (unique integers), `user_id` (randomly sampled from `users_df` IDs), `content_id` (randomly sampled from `content_items_df` IDs), `timestamp` (random timestamps occurring *after* their respective `signup_date`), `interaction_type` (e.g., 'view', 'like', 'share', 'comment', 'complete').
    *   **Simulate realistic patterns**: Ensure `timestamp` is always after `signup_date`. Bias the data such that 'Premium' users interact with more diverse content and potentially more 'Advanced' difficulty items. Certain `interaction_type`s (e.g., 'comment', 'complete') are rarer. Users often interact with multiple items of the same `category` in a session before moving to another. Ensure that for most interactions, there is at least one subsequent interaction for the same user to define a target.
    *   Sort `interactions_df` by `user_id` then `timestamp` for easier sequential processing.

2. **Load into SQLite & SQL Feature Engineering (Sequential Interaction Prediction)**: Create an in-memory SQLite database using `sqlite3`. Load `users_df`, `content_items_df`, and `interactions_df` into tables named `users`, `content_items`, and `interactions` respectively.
    Write a single SQL query that performs the following for *each interaction* in `interactions` (excluding the very last interaction for each user, as it won't have a 'next' interaction):
    *   **Joins** `users`, `content_items`, and `interactions` tables.
    *   **Calculates sequential features based on the user's *prior interactions* (excluding the current one), relative to the current `timestamp`**:
        *   `user_prior_num_interactions`: Count of all *previous* interactions for the same user.
        *   `days_since_last_user_interaction`: Number of days between the current `timestamp` and the user's *most recent prior* `timestamp`. If it's the user's first interaction, use the number of days between `signup_date` and the current `timestamp`.
        *   `user_prior_num_unique_content_categories`: Count of distinct `category` from the user's *prior* interactions.
        *   `user_prior_num_video_views`: Count of 'view' interactions for `category = 'Video'` from *prior* interactions.
        *   `user_prior_num_article_views`: Count of 'view' interactions for `category = 'Article'` from *prior* interactions.
    *   **Includes static user, current content, and current interaction attributes**: `interaction_id`, `user_id`, `timestamp`, `interaction_type` (current), `content_id`, `category` (current content category), `difficulty`, `avg_rating`, `signup_date`, `age_group`, `premium_status`.
    *   **Creates the Multi-Class Target `next_content_category`**: This should be the `category` of the *immediately subsequent* interaction made by the same user. Use a window function for this. Filter out rows where `next_content_category` is `NULL`.
    *   The query should return all these attributes and engineered features. Missing values for prior aggregates/dates should be `NULL`.
    *   **Hint**: Use window functions with `LAG` and `LEAD` over `PARTITION BY user_id ORDER BY timestamp`. For strictly *prior* aggregates, use `ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING` within `SUM(...) OVER (...)` or `COUNT(...) OVER (...)`. For `days_since_last_user_interaction`, use `julianday()` and `LAG(i.timestamp, 1, u.signup_date) OVER (PARTITION BY u.user_id ORDER BY i.timestamp)` to handle first interactions. Filter out rows where `next_content_category` is `NULL`.

3. **Pandas Feature Engineering & Multi-Class Target Creation**: Fetch the SQL query results into a pandas DataFrame (`interaction_features_df`).
    *   Handle `NaN` values: Fill `user_prior_num_interactions`, `user_prior_num_unique_content_categories`, `user_prior_num_video_views`, `user_prior_num_article_views` with 0. For `days_since_last_user_interaction` (for a user's first interaction), SQL should handle, but if any `NaN`s remain, fill with `days_since_signup_at_interaction`.
    *   Convert `signup_date` and `timestamp` to datetime objects.
    *   Calculate `days_since_signup_at_interaction`: Days between `signup_date` and `timestamp`.
    *   Calculate `interaction_frequency_prior`: `user_prior_num_interactions` / (`days_since_signup_at_interaction` + 1). Fill any `NaN` or `inf` with 0.
    *   Define features `X` (all numerical: `avg_rating`, `user_prior_num_interactions`, `days_since_last_user_interaction`, `user_prior_num_unique_content_categories`, `user_prior_num_video_views`, `user_prior_num_article_views`, `days_since_signup_at_interaction`, `interaction_frequency_prior`; categorical: `interaction_type`, `category` (current), `difficulty`, `age_group`, `premium_status`) and target `y` (`next_content_category`). Split into training and testing sets (e.g., 70/30 split) using `sklearn.model_selection.train_test_split` (set `random_state=42`, `stratify` on `y` for class balance).

4. **Data Visualization**: Create two separate plots to visually inspect relationships with `next_content_category`:
    *   A violin plot (or box plot) showing the distribution of `days_since_last_user_interaction` for each of the top 5 `next_content_category` values. Ensure appropriate labels and titles.
    *   A stacked bar chart showing the proportion of `next_content_category` values across different `premium_status` values. Ensure appropriate labels and titles.

5. **ML Pipeline & Evaluation (Multi-Class)**: 
    *   Create an `sklearn.pipeline.Pipeline` with a `sklearn.compose.ColumnTransformer` for preprocessing:
        *   For numerical features: Apply `sklearn.preprocessing.SimpleImputer(strategy='mean')` followed by `sklearn.preprocessing.StandardScaler`.
        *   For categorical features: Apply `sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')`.
    *   The final estimator in the pipeline should be `sklearn.ensemble.HistGradientBoostingClassifier` (set `random_state=42`).
    *   Train the pipeline on `X_train`, `y_train`. Predict `next_content_category` for `X_test`.
    *   Calculate and print the `sklearn.metrics.accuracy_score` and a `sklearn.metrics.classification_report` for the test set predictions.

## Hint
Use window functions with `LAG` and `LEAD` over `PARTITION BY user_id ORDER BY timestamp`. For strictly *prior* aggregates (e.g., counts of previous interactions), consider using `ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING` within `SUM(...) OVER (...)` or `COUNT(...) OVER (...)`. For `days_since_last_user_interaction`, combine `julianday()` with `LAG(i.timestamp, 1, u.signup_date) OVER (...)` to correctly handle a user's very first interaction.
