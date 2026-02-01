# AI Daily Lab — 2026-02-01

## Task
1. **Generate Synthetic Data (Pandas/Numpy)**: Create three pandas DataFrames:
    *   `users_df`: 500-700 rows. Columns: `user_id` (unique int), `signup_date` (random dates over last 3 years), `source` (e.g., 'Organic', 'Social', 'Referral').
    *   `articles_df`: 100-150 rows. Columns: `article_id` (unique int), `category` (e.g., 'Tech', 'Finance', 'Health', 'Sports'), `reading_time_minutes` (random floats 2-15), `publish_date` (random dates over last 2 years).
    *   `reads_df`: 3000-5000 rows. Columns: `read_id` (unique int), `user_id` (randomly sampled from `users_df` IDs), `article_id` (randomly sampled from `articles_df` IDs), `read_date` (random dates *after* user's `signup_date` and article's `publish_date`), `time_spent_seconds` (random ints 10-600, typically less than `reading_time_minutes` * 60). Ensure some users have many reads, others few or none, and that read dates are chronologically consistent.

2. **Load into SQLite & SQL Feature Engineering (Initial Engagement)**: Create an in-memory SQLite database using `sqlite3`. Load `users_df`, `articles_df`, `reads_df` into tables named `users`, `articles`, and `reads` respectively. Determine an `analysis_date` (e.g., `max(read_date)` from `reads_df` + 30 days, using pandas). Define an `initial_period_days` (e.g., 60 days).
    Write a single SQL query that performs the following for *each user* based on their activity within the `initial_period_days` *from their `signup_date`*:
    *   **Joins** `users`, `articles`, and `reads` tables.
    *   **Aggregates** `initial_num_reads` (count of reads), `initial_total_time_spent` (sum of `time_spent_seconds`), `initial_avg_reading_time_per_article` (average of `time_spent_seconds`).
    *   **Aggregates category-specific read counts**: `initial_reads_tech`, `initial_reads_finance`, `initial_reads_health`, `initial_reads_sports` (count of reads for each respective `category` within the initial period).
    *   **Ensures** all users are included, showing 0 for counts/sums and `NULL` for averages if no reads in the initial period.
    *   The query should return `user_id`, `signup_date`, `source`, `initial_num_reads`, `initial_total_time_spent`, `initial_avg_reading_time_per_article`, `initial_reads_tech`, `initial_reads_finance`, `initial_reads_health`, `initial_reads_sports`.

3. **Pandas Feature Engineering & Target Creation (Regular Reader)**: Fetch the SQL query results into a pandas DataFrame (`user_initial_features_df`).
    *   **Calculate All-Time Read Metrics**: From the original `reads_df`, calculate `total_reads_all_time` (count of reads) and `last_read_date_all_time` (maximum `read_date`) for each user. Merge these aggregates with `user_initial_features_df` (left join).
    *   Merge the result with `users_df` (left join on `user_id`) to ensure all users are present and have their `source` and `signup_date`.
    *   Handle `NaN` values: Fill `initial_num_reads`, `initial_total_time_spent`, `initial_avg_reading_time_per_article`, `initial_reads_tech`, `initial_reads_finance`, `initial_reads_health`, `initial_reads_sports`, `total_reads_all_time` with 0 for users with no activity. For `last_read_date_all_time` (for users with no reads ever), fill with `signup_date`.
    *   Convert all relevant date columns to datetime objects. Calculate `account_age_days`: Days between `signup_date` and the `analysis_date` (from step 2).
    *   Calculate `days_since_last_read`: Days between `last_read_date_all_time` and `analysis_date`. For users with no activity, fill with a large sentinel value (e.g., `account_age_days` + 30).
    *   **Create `favorite_category_initial`**: Based on `initial_reads_tech`, `initial_reads_finance`, etc. (the category with the highest count among the *initial period* features). If all are 0, assign 'None'. (Hint: Use `idxmax()` on selected columns and `.fillna('None')`).
    *   **Create Binary Target `is_regular_reader`**: A user is a 'regular reader' (1) if their `total_reads_all_time` is in the top 30th percentile (among non-zero `total_reads_all_time`) *and* their `days_since_last_read` is less than 60 days. Otherwise, 0.
    *   Define features `X` (`source`, `account_age_days`, `initial_num_reads`, `initial_total_time_spent`, `initial_avg_reading_time_per_article`, `initial_reads_tech`, `initial_reads_finance`, `initial_reads_health`, `initial_reads_sports`, `favorite_category_initial`, `days_since_last_read`) and target `y` (`is_regular_reader`). Split into training and testing sets (e.g., 70/30 split) using `sklearn.model_selection.train_test_split` (set `random_state=42`, `stratify` on `y` for class balance).

4. **Data Visualization**: Create two separate plots to visually inspect relationships with `is_regular_reader`:
    *   A violin plot (or box plot) showing the distribution of `initial_num_reads` for each `is_regular_reader` group.
    *   A stacked bar chart showing the distribution of `is_regular_reader` across different `source` values.
    Ensure plots have appropriate labels and titles.

5. **ML Pipeline & Evaluation**: 
    *   Create an `sklearn.pipeline.Pipeline` with a `sklearn.compose.ColumnTransformer` for preprocessing:
        *   For numerical features (`account_age_days`, `initial_num_reads`, `initial_total_time_spent`, `initial_avg_reading_time_per_article`, `initial_reads_tech`, `initial_reads_finance`, `initial_reads_health`, `initial_reads_sports`, `days_since_last_read`): Apply `sklearn.preprocessing.SimpleImputer(strategy='mean')` followed by `sklearn.preprocessing.StandardScaler`.
        *   For categorical features (`source`, `favorite_category_initial`): Apply `sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')`.
    *   The final estimator in the pipeline should be `sklearn.ensemble.GradientBoostingClassifier` (set `random_state=42`, `n_estimators=100`, `learning_rate=0.1`).
    *   Train the pipeline on `X_train`, `y_train`. Predict probabilities for the positive class (class 1) on `X_test`.
    *   Calculate and print the `sklearn.metrics.roc_auc_score` and a `sklearn.metrics.classification_report` for the test set predictions.

## Focus
Predicting User Engagement (Regular Reader) based on Initial Activity and All-Time Recency with SQL Aggregation and ML Pipeline.

## Dataset
Synthetic data: Users, Articles, and Read Activities.

## Hint
When generating `read_date` ensure it's always after both `signup_date` and `publish_date`. For SQL category-specific counts, use `SUM(CASE WHEN a.category = 'Tech' THEN 1 ELSE 0 END)` within your `GROUP BY user_id` clause. For `favorite_category_initial` in pandas, consider using `df[['initial_reads_tech', 'initial_reads_finance', ...]].idxmax(axis=1)` and then mapping the resulting column names to actual category strings. Remember to handle cases where all initial category counts are zero.
