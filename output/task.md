# AI Daily Lab — 2026-02-02

## Task
1. **Generate Synthetic Data (Pandas/Numpy)**: Create two pandas DataFrames:
    *   `users_df`: With 500-700 rows. Columns: `user_id` (unique integers), `signup_date` (random dates over the last 5 years), `region` (e.g., 'North', 'South', 'East', 'West' with random distribution), `age` (random integers 18-70).
    *   `purchases_df`: With 3000-5000 rows. Columns: `purchase_id` (unique integers), `user_id` (randomly sampled from `users_df` IDs, ensuring some users have many purchases, some have few, and some have none), `purchase_date` (random dates occurring *after* their respective `signup_date`), `amount` (random floats between 10.0 and 1000.0).
    *   **Simulate purchase patterns**: Ensure `purchase_date` is after `signup_date`. Generate data such that some users purchase only in their early days, some purchase consistently over time, and some only start purchasing later.

2. **Load into SQLite & SQL Feature Engineering (Initial Buyer Behavior)**: Create an in-memory SQLite database using `sqlite3`. Load `users_df` into a table named `users` and `purchases_df` into a table named `purchases`. Determine a global `analysis_date` (e.g., the maximum `purchase_date` in `purchases_df` + 60 days, using pandas).
    Write a single SQL query that performs the following for *each user*:
    *   **Joins** `users` and `purchases` tables.
    *   **Aggregates initial purchase behavior** for transactions occurring within the **first 90 days** after their `signup_date` (i.e., `purchase_date` between `signup_date` and `signup_date + 90 days`):
        *   `initial_num_purchases` (count of purchases)
        *   `initial_total_spend` (sum of `amount`)
        *   `initial_avg_purchase_value` (average `amount`)
        *   `initial_days_to_first_purchase` (number of days between `signup_date` and the `MIN(purchase_date)` within the initial 90-day window. `NULL` if no purchase).
    *   **Ensures** all users are included (using a `LEFT JOIN`), showing 0 for counts/sums and `NULL` for averages/days if no purchases in the initial period.
    *   The query should return `user_id`, `region`, `age`, `signup_date`, `initial_num_purchases`, `initial_total_spend`, `initial_avg_purchase_value`, `initial_days_to_first_purchase`.

3. **Pandas Feature Engineering & Binary Target Creation (Future Buyer Prediction)**: Fetch the SQL query results into a pandas DataFrame (`initial_features_df`).
    *   Handle `NaN` values: Fill `initial_num_purchases`, `initial_total_spend`, `initial_avg_purchase_value` with 0. For `initial_days_to_first_purchase` (for users with no initial purchases), fill with a large sentinel value (e.g., 9999 or 120 days).
    *   Convert `signup_date` to datetime. Calculate `account_age_days_at_analysis`: The number of days between `signup_date` and the global `analysis_date` (from step 2).
    *   **Create the Binary Target `made_future_purchase`**: A user is considered to have `made_future_purchase` (1) if they have made *any* purchase with `purchase_date` occurring *after* their `signup_date + 90 days` (the initial observation period) and *before* the global `analysis_date`. Otherwise, 0.
    *   Define features `X` (`region`, `age`, `account_age_days_at_analysis`, `initial_num_purchases`, `initial_total_spend`, `initial_avg_purchase_value`, `initial_days_to_first_purchase`) and target `y` (`made_future_purchase`). Split into training and testing sets (e.g., 70/30 split) using `sklearn.model_selection.train_test_split` (set `random_state=42`, `stratify` on `y` for class balance).

4. **Data Visualization**: Create two separate plots to visually inspect relationships with `made_future_purchase`:
    *   A violin plot (or box plot) showing the distribution of `initial_total_spend` for users who `made_future_purchase=0` vs. `made_future_purchase=1`.
    *   A stacked bar chart showing the proportion of `made_future_purchase` (0 or 1) across different `region`s.
    Ensure plots have appropriate labels and titles.

5. **ML Pipeline & Evaluation (Basic AI Experimentation)**: 
    *   Create an `sklearn.pipeline.Pipeline` with a `sklearn.compose.ColumnTransformer` for preprocessing:
        *   For numerical features (`age`, `account_age_days_at_analysis`, `initial_num_purchases`, `initial_total_spend`, `initial_avg_purchase_value`, `initial_days_to_first_purchase`): Apply `sklearn.preprocessing.StandardScaler`.
        *   For the categorical feature (`region`): Apply `sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')`.
    *   The final estimator in the pipeline should be `sklearn.ensemble.HistGradientBoostingClassifier` (set `random_state=42`).
    *   Train the pipeline on `X_train`, `y_train`. Predict probabilities for the positive class (class 1) on `X_test`.
    *   Calculate and print the `sklearn.metrics.roc_auc_score` and a `sklearn.metrics.classification_report` for the test set predictions.

## Focus
Predicting future buyer status based on initial user behavior and demographics, using SQL for initial feature aggregation and a robust ML pipeline with a chronological perspective.

## Dataset
Synthetic user and purchase transaction data.

## Hint
When defining `made_future_purchase`, first get all unique `user_id`s who made *any* purchase after their `signup_date + 90 days`. Then use this set to mark your target column. Remember to convert all date columns to datetime objects early for calculations. For the SQL query, `DATE()` and `JULIANDAY()` functions in SQLite can be useful for date arithmetic to define the 90-day window.
