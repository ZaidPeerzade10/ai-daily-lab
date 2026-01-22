# AI Daily Lab — 2026-01-22

## Task
1. **Generate Synthetic Data (Pandas/Numpy)**: Create two pandas DataFrames:
    *   `users_df`: With 500 rows. Columns: `user_id` (unique integers), `signup_date` (random dates over the last 3 years), `country` (e.g., 'USA', 'Canada', 'UK', 'Australia' with random distribution), `device_type` (e.g., 'Mobile', 'Web', 'Desktop'), `engagement_level` (binary target, 0 or 1, with an approximate 70/30 split for class 0/1).
    *   `activities_df`: With 3000-5000 rows. Columns: `activity_id` (unique integers), `user_id` (randomly sampled from `users_df` IDs, ensuring some users have many activities and a few have no activities), `activity_date` (random dates occurring *after* their respective `signup_date`), `activity_type` (e.g., 'login', 'logout', 'view_profile', 'send_message', 'post_content', 'like_content' with varying frequencies).
2. **Load into SQLite**: Create an in-memory SQLite database using `sqlite3` and load `users_df` into a table named `users` and `activities_df` into `activities`.
3. **SQL Feature Engineering (User Activity Aggregation)**: First, determine an `analysis_date` by finding the maximum `activity_date` in your `activities_df` (using pandas) and adding 30 days to it. Then, write a single SQL query that performs the following for *each user*:
    *   **Joins** `users` and `activities` tables.
    *   **Aggregates** user-level features: `total_activities` (count of all activities), `num_logins` (count of 'login' events), `num_messages_sent` (count of 'send_message' events), `num_content_posts` (count of 'post_content' events).
    *   Calculates `first_activity_date` and `last_activity_date` for each user.
    *   **Ensures** that all users are included, even those with no activities, showing appropriate default values (e.g., 0 for counts, `NULL` for dates).
    *   The query should return `user_id`, `country`, `device_type`, `signup_date`, `total_activities`, `num_logins`, `num_messages_sent`, `num_content_posts`, `first_activity_date`, `last_activity_date`.
4. **Retrieve, Merge, and Pandas Feature Engineering**: 
    *   Fetch the SQL query results into a pandas DataFrame.
    *   Merge this aggregated DataFrame with the original `users_df` on `user_id`.
    *   Handle `NaN` values resulting from the SQL query: Fill `total_activities`, `num_logins`, `num_messages_sent`, `num_content_posts` with 0 for users with no activities.
    *   Convert all date columns to datetime objects. Calculate the following new features using the `analysis_date` from step 3:
        *   `account_age_days`: Days between `signup_date` and `analysis_date`.
        *   `days_since_last_activity`: Days between `last_activity_date` and `analysis_date`. For users with no activities, fill with a large sentinel value (e.g., `account_age_days` + 30).
        *   `days_since_first_activity`: Days between `first_activity_date` and `analysis_date`. For users with no activities, fill with a large sentinel value (e.g., `account_age_days` + 30).
    *   Define features `X` (`country`, `device_type`, `account_age_days`, `days_since_last_activity`, `days_since_first_activity`, `total_activities`, `num_logins`, `num_messages_sent`, `num_content_posts`) and target `y` (`engagement_level`). Split into training and testing sets (e.g., 70/30 split) using `sklearn.model_selection.train_test_split` (set `random_state=42`).
5. **Data Visualization**: Create two separate plots to visually inspect relationships with `engagement_level`:
    *   A histogram showing the distribution of `total_activities` for each `engagement_level` (e.g., using `hue` in seaborn).
    *   A bar plot or count plot showing the `engagement_level` counts across different `device_type`s.
    Ensure plots have appropriate labels and titles.
6. **ML Pipeline & Evaluation**: 
    *   Create an `sklearn.pipeline.Pipeline` with a `ColumnTransformer` for preprocessing:
        *   For numerical features (e.g., `account_age_days`, `days_since_last_activity`, `days_since_first_activity`, `total_activities`, etc.): Apply `SimpleImputer(strategy='mean')` followed by `StandardScaler`.
        *   For categorical features (`country`, `device_type`): Apply `OneHotEncoder(handle_unknown='ignore')`.
    *   The final estimator in the pipeline should be `sklearn.linear_model.LogisticRegression` (set `random_state=42`, `solver='liblinear'` for reproducibility).
    *   Train the pipeline on the training data. Predict probabilities for the positive class (class 1) on the test set.
    *   Calculate and print the `sklearn.metrics.roc_auc_score` and a `sklearn.metrics.classification_report` for the test set predictions.

## Focus
Advanced SQL Feature Engineering from multiple tables, Pandas for datetime features and data wrangling, ML Pipeline for binary classification with categorical/numerical data, and model evaluation.

## Dataset
Synthetic User Profiles and Activity Logs

## Hint
Remember to use `LEFT JOIN` in your SQL query to ensure all users are included, even those with no activities. For date differences in pandas, subtract `datetime` objects to get `Timedelta` and then extract days. Pay attention to handling `NaN` values, especially for users with no activities, by filling them appropriately before passing to the ML pipeline.
