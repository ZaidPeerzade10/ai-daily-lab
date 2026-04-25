# AI Daily Lab — 2026-04-25

## Task
Develop a machine learning pipeline to predict a user's *next preferred content genre* based on their historical interaction patterns and profile attributes.

## Focus
Multi-class classification for user content preferences, involving date-based historical aggregation, and target creation from future behavior.

## Dataset
1. **Generate Synthetic Data (Pandas/Numpy)**: Create three pandas DataFrames:
    *   `users_df`: With 500-800 rows. Columns: `user_id` (unique integers), `signup_date` (random dates over the last 3 years), `region` (e.g., 'North', 'South', 'East', 'West'), `subscription_tier` (categorical: 'Free', 'Basic', 'Premium'), `device_type` (categorical: 'Mobile', 'Desktop', 'Tablet'), `age` (random integers 18-65).
    *   `content_df`: With 100-150 rows. Columns: `content_id` (unique integers), `genre` (e.g., 'Action', 'Comedy', 'Drama', 'Sci-Fi', 'Documentary', 'Fantasy'), `avg_rating` (random floats 1.0-5.0), `production_year` (random integers over the last 15 years).
    *   `interactions_df`: With 10000-15000 rows. Columns: `interaction_id` (unique integers), `user_id` (randomly sampled from `users_df` IDs), `content_id` (randomly sampled from `content_df` IDs), `interaction_date` (random dates occurring *after* their respective `signup_date`, up to a `current_prediction_date`), `interaction_type` (e.g., 'view', 'like', 'share'), `duration_minutes` (random integers 1-120).
    *   **Simulate realistic patterns**: Ensure `interaction_date` is always after `signup_date`. `Premium` users should have a higher average `duration_minutes` and more `like` interactions. `Mobile` users might favor 'Comedy' or 'Shorts' (you can adjust duration), `Desktop` users 'Sci-Fi' or 'Drama'. Content with higher `avg_rating` should generally have more `view` interactions. Interactions should span several months for users.
    *   Define a `current_prediction_date = pd.to_datetime('2024-03-01')`. Ensure `interaction_date` values do not exceed this date.
    *   Sort `interactions_df` by `user_id` then `interaction_date`.

2. **Load into SQLite & SQL Feature Engineering (Historical Interaction Patterns)**: Create an in-memory SQLite database using `sqlite3`. Load `users_df`, `content_df`, and `interactions_df` into tables named `users`, `content`, and `interactions` respectively.
    Write a single SQL query that performs the following for *each user*, aggregating their interaction behavior *within the 60 days ending at `current_prediction_date - 30 days`* (let's call this `history_cutoff_date`).
    *   **Joins** `users` with an aggregated subquery for `interactions` (and `content` to get `genre` and `avg_rating`).
    *   **Aggregates features based on activities *within the 60-day historical window* (i.e., `interaction_date` between `history_cutoff_date - 60 days` and `history_cutoff_date`)**:
        *   `num_interactions_prev_60d` (count of `interaction_id`s)
        *   `total_duration_prev_60d` (sum of `duration_minutes`)
        *   `num_unique_genres_prev_60d` (count of distinct `genre`s)
        *   `avg_content_rating_prev_60d` (average of `content.avg_rating` for interacted content).
    *   **Includes static user attributes**: `user_id`, `signup_date`, `region`, `subscription_tier`, `device_type`, `age`.
    *   **Ensures** all users are included (using `LEFT JOIN`), showing 0 for counts/sums and 0.0 for averages if no activity in the 60-day window.
    *   The query should return `user_id`, `signup_date`, `region`, `subscription_tier`, `device_type`, `age`, and all the aggregated features.
    *   **Hint**: Use `julianday()` for date comparisons. Define `history_cutoff_date` in your SQL query as `DATE('2024-03-01', '-30 days')` and filter interactions accordingly.

3. **Pandas Feature Engineering & Multi-Class Target Creation (Next Preferred Genre)**: Fetch the SQL query results into a pandas DataFrame (`user_history_features_df`).
    *   Handle `NaN` values: Fill `num_interactions_prev_60d`, `total_duration_prev_60d`, `num_unique_genres_prev_60d` with 0. Fill `avg_content_rating_prev_60d` with 0.0.
    *   Convert `signup_date` to datetime objects. Set `current_prediction_date = pd.to_datetime('2024-03-01')` and `history_cutoff_date = current_prediction_date - pd.Timedelta(30, 'days')`.
    *   Calculate `interaction_frequency_prev_60d`: `num_interactions_prev_60d` / 60.0. Fill any `NaN`s with 0.
    *   **Create the Multi-Class Target `next_preferred_genre`**: For *each user*, identify the `genre` with the *highest total `duration_minutes`* from all their interactions (from the original `interactions_df` and `content_df`) that occur *between* `history_cutoff_date` AND `history_cutoff_date + 30 days`. Merge this preferred genre with `user_history_features_df` (left join).
        *   If a user has *no interactions* in this 30-day future window, assign them the target class 'No Future Preference'.
    *   Define features `X` (numerical: `num_interactions_prev_60d`, `total_duration_prev_60d`, `num_unique_genres_prev_60d`, `avg_content_rating_prev_60d`, `age`, `interaction_frequency_prev_60d`; categorical: `region`, `subscription_tier`, `device_type`) and target `y` (`next_preferred_genre`). Split into training and testing sets (e.g., 70/30 split) using `sklearn.model_selection.train_test_split` (set `random_state=42`, `stratify` on `y` for class balance).

4. **Data Visualization**: Create two separate plots to visually inspect relationships with `next_preferred_genre`:
    *   A violin plot (or box plot) showing the distribution of `avg_content_rating_prev_60d` for each `next_preferred_genre` tier. Ensure appropriate labels and titles.
    *   A stacked bar chart showing the proportion of `next_preferred_genre` across different `subscription_tier` values. Ensure appropriate labels and titles.

5. **ML Pipeline & Evaluation (Multi-Class)**: 
    *   Create an `sklearn.pipeline.Pipeline` with a `sklearn.compose.ColumnTransformer` for preprocessing:
        *   For numerical features: Apply `sklearn.preprocessing.SimpleImputer(strategy='mean')` followed by `sklearn.preprocessing.StandardScaler`.
        *   For categorical features: Apply `sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')`.
    *   The final estimator in the pipeline should be `sklearn.ensemble.HistGradientBoostingClassifier` (set `random_state=42`).
    *   Train the pipeline on `X_train`, `y_train`. Predict `next_preferred_genre` for `X_test`.
    *   Calculate and print the `sklearn.metrics.accuracy_score` and a `sklearn.metrics.classification_report` for the test set predictions.

## Hint
When creating the `next_preferred_genre` target in Pandas, filter `interactions_df` for the target window, join with `content_df`, group by `user_id` and `genre`, sum `duration_minutes`, then use `idxmax()` on the grouped sums to find the preferred genre for each user. Remember to handle users with no interactions in the target window by assigning 'No Future Preference'.
