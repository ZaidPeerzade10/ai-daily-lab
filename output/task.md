# AI Daily Lab — 2026-01-19

## Task
1. **Generate Synthetic Data**: Create two pandas DataFrames:
    *   `users_df`: With 500 rows. Columns: `user_id` (unique integers), `signup_date` (random dates over the last 3 years), `region` (e.g., 'North', 'South', 'East', 'West' with random distribution), `plan_type` (e.g., 'Basic', 'Premium', 'Pro' with random distribution), `churn` (binary target, 0 or 1, with an approximate 20% churn rate).
    *   `events_df`: With 3000-5000 rows. Columns: `event_id` (unique integers), `user_id` (randomly sampled from `users_df` IDs, ensuring some users have many events and a few have no events), `event_date` (random dates occurring *after* their respective `signup_date`), `event_type` (e.g., 'login', 'view_item', 'purchase', 'cancel_subscription' with varying frequencies).
2. **Load into SQLite**: Create an in-memory SQLite database using `sqlite3` and load `users_df` into a table named `users` and `events_df` into `events`.
3. **SQL Feature Engineering (User Activity Aggregation)**: First, determine the `analysis_end_date` by finding the maximum `event_date` in your `events_df` (using pandas). Then, write a single SQL query that performs the following:
    *   **Joins** `users` and `events` tables.
    *   **Aggregates** user-level features: `total_events` (count of all events), `days_since_last_event` (number of days between the `analysis_end_date` and the user's latest `event_date`), `num_logins` (count of 'login' events), `num_purchases` (count of 'purchase' events).
    *   **Ensures** that all users are included, even those with no events, showing appropriate default values (e.g., 0 for counts, `NULL` for `days_since_last_event` if no events).
    *   The query should return `user_id`, `total_events`, `days_since_last_event`, `num_logins`, `num_purchases`.
4. **Retrieve, Merge, and Final Data Prep (Pandas)**:
    *   Fetch the SQL query results into a pandas DataFrame.
    *   Merge this aggregated DataFrame with the original `users_df` on `user_id`.
    *   Handle `NaN` values resulting from the SQL query: Fill `total_events`, `num_logins`, `num_purchases` with 0 for users with no events. For `days_since_last_event` (for users with no events), fill with a large sentinel value (e.g., `365 * 5` or 1825 to represent 5 years of inactivity).
    *   Create a new feature `account_age_days`: Calculate the number of days between `signup_date` and the `analysis_end_date` (from step 3).
    *   Define features `X` (all numerical + `region`, `plan_type`) and target `y` (`churn`). Split into training and testing sets (e.g., 70/30 split) using `sklearn.model_selection.train_test_split`.
5. **Data Visualization**: Create two separate violin plots (or box plots) to visualize the relationship between key engineered features and churn:
    *   `days_since_last_event` vs `churn`.
    *   `num_purchases` vs `churn`.
    Ensure plots have appropriate labels and titles.
6. **ML Pipeline & Evaluation**: 
    *   Create an `sklearn.pipeline.Pipeline` with a `ColumnTransformer` for preprocessing:
        *   For numerical features (e.g., `total_events`, `days_since_last_event`, `num_logins`, `num_purchases`, `account_age_days`): Apply `SimpleImputer(strategy='mean')` followed by `StandardScaler`.
        *   For categorical features (`region`, `plan_type`): Apply `OneHotEncoder(handle_unknown='ignore')`.
    *   The final estimator in the pipeline should be `sklearn.linear_model.LogisticRegression` (set `random_state=42`, `solver='liblinear'` for reproducibility).
    *   Train the pipeline on the training data. Predict on the test set.
    *   Calculate and print the `sklearn.metrics.roc_auc_score` and `sklearn.metrics.accuracy_score` for the test set predictions.

## Focus
Comprehensive end-to-end user churn prediction: data synthesis, SQL-driven feature engineering (recency, frequency, event counts), data preparation, visualization, and a robust ML pipeline.

## Dataset
Synthetic user profiles and event logs for a subscription service.

## Hint
For SQL date calculations in SQLite, use `JULIANDAY()` to convert dates to numbers for subtraction. Remember to dynamically pass your `analysis_end_date` (a Python variable) into your SQL query using f-strings or parameter binding. Use `COALESCE()` or `IFNULL()` in SQL for handling default values for users without events. For pandas NaN filling after SQL, ensure you use appropriate strategies (0 for counts, a suitable large sentinel value for days since last event for inactive users). When setting up the `ColumnTransformer`, explicitly list the column names for numerical and categorical features.
