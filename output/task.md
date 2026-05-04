# AI Daily Lab — 2026-05-04

## Task
Develop a machine learning pipeline to predict a customer's total monetary spend in the *next 6 months* (a form of Customer Lifetime Value prediction), based on their demographic profile and their historical transaction behavior up to a specific cutoff date.

1.  **Generate Synthetic Data (Pandas/Numpy)**: Create two pandas DataFrames:
    *   `customers_df`: With 1000-1500 rows. Columns: `customer_id` (unique integers), `signup_date` (random dates over the last 3-5 years), `region` (e.g., 'North', 'South', 'East', 'West'), `age` (random integers 18-70), `initial_channel` (e.g., 'Web', 'Mobile App', 'Referral').
    *   `transactions_df`: With 20000-30000 rows. Columns: `transaction_id` (unique integers), `customer_id` (randomly sampled from `customers_df` IDs), `transaction_date` (random dates occurring *after* their respective `signup_date` and up to a `current_max_date` like `pd.Timestamp.now() - pd.Timedelta(weeks=4)`), `amount` (random floats 5.0-1000.0), `product_category` (e.g., 'Electronics', 'Books', 'Groceries', 'Services', 'Apparel').
    *   **Simulate realistic patterns**: Ensure `transaction_date` is always after `signup_date`. Older customers (e.g., age > 45) might have slightly higher average `amount`. 'Mobile App' users might have more frequent but smaller `amount` transactions. 'Electronics' and 'Services' categories might have higher `amount` on average. A small percentage (e.g., 5-10%) of customers could have significantly higher `amount`s to simulate high-value customers.
    *   Sort `transactions_df` by `customer_id` then `transaction_date`.

2.  **Load into SQLite & SQL Feature Engineering (Historical Spend Aggregations)**: Create an in-memory SQLite database using `sqlite3`. Load `customers_df` and `transactions_df` into tables named `customers` and `transactions` respectively.
    *   Define `GLOBAL_PREDICTION_CUTOFF_DATE` as 6 months prior to the latest `transaction_date` in your generated `transactions_df` (e.g., `transactions_df['transaction_date'].max() - pd.Timedelta(months=6)`).
    *   Write a single SQL query that performs the following for *each customer*, aggregating their transaction behavior *within the 6 months immediately preceding `GLOBAL_PREDICTION_CUTOFF_DATE`*:
        *   `current_cutoff_date` (the `GLOBAL_PREDICTION_CUTOFF_DATE` itself, for consistency).
        *   `total_spend_prev_6m` (sum of `amount`).
        *   `num_transactions_prev_6m` (count of `transaction_id`s).
        *   `avg_transaction_value_prev_6m` (average of `amount`).
        *   `days_since_last_transaction_at_cutoff`: Number of days between `GLOBAL_PREDICTION_CUTOFF_DATE` and the most recent `transaction_date` *before or on* the cutoff. Return a large number (e.g., 9999) if no transactions before cutoff.
        *   `num_unique_categories_prev_6m` (count of distinct `product_category`).
    *   **Includes static customer attributes**: `customer_id`, `signup_date`, `region`, `age`, `initial_channel`.
    *   **Ensures** all customers are included (using `LEFT JOIN`), showing 0 for counts/sums/averages if no transactions in the 6-month window.
    *   The query should return `customer_id`, `signup_date`, `region`, `age`, `initial_channel`, `current_cutoff_date`, and all aggregated features.

3.  **Pandas Feature Engineering & Regression Target Creation**: Fetch the SQL query results into a pandas DataFrame (`customer_features_df`).
    *   Convert `signup_date` and `current_cutoff_date` to datetime objects.
    *   Handle `NaN` values: Fill `total_spend_prev_6m`, `num_transactions_prev_6m`, `avg_transaction_value_prev_6m`, `num_unique_categories_prev_6m` with 0. Fill `days_since_last_transaction_at_cutoff` with a large number (e.g., 9999) to indicate no recent activity.
    *   Calculate `customer_age_at_cutoff_days`: Number of days between `signup_date` and `current_cutoff_date`.
    *   Calculate `avg_daily_spend_prev_6m`: `total_spend_prev_6m` / 180.0 (approximately 6 months). Fill any `NaN` or `inf` with 0.
    *   **Create the Regression Target `next_6m_spend`**: For *each customer*, sum their `amount` from `transactions_df` for all transactions that occurred *after* `current_cutoff_date` and *before or on* `current_cutoff_date + pd.Timedelta(days=180)`. Merge this sum into `customer_features_df`, filling `NaN`s with 0 for customers with no spend in the target window.
    *   Define features `X` (numerical: `age`, `total_spend_prev_6m`, `num_transactions_prev_6m`, `avg_transaction_value_prev_6m`, `days_since_last_transaction_at_cutoff`, `num_unique_categories_prev_6m`, `customer_age_at_cutoff_days`, `avg_daily_spend_prev_6m`; categorical: `region`, `initial_channel`) and target `y` (`next_6m_spend`). Split into training and testing sets (e.g., 70/30 split) using `sklearn.model_selection.train_test_split` (set `random_state=42`).

4.  **Data Visualization**: Create two separate plots to visually inspect relationships with `next_6m_spend` (the target):
    *   A scatter plot showing `total_spend_prev_6m` vs. `next_6m_spend`. Consider applying a log transformation to the target or features if the distributions are heavily skewed to improve visibility of relationships.
    *   A box plot (or violin plot) showing the distribution of `next_6m_spend` across different `initial_channel` values.
    *   Ensure appropriate labels and titles for both plots.

5.  **ML Pipeline & Evaluation (Regression)**:
    *   Create an `sklearn.pipeline.Pipeline` with a `sklearn.compose.ColumnTransformer` for preprocessing:
        *   For numerical features: Apply `sklearn.preprocessing.SimpleImputer(strategy='mean')` followed by `sklearn.preprocessing.StandardScaler`.
        *   For categorical features: Apply `sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')`.
    *   The final estimator in the pipeline should be `sklearn.ensemble.HistGradientBoostingRegressor` (set `random_state=42`).
    *   Train the pipeline on `X_train`, `y_train`. Predict `next_6m_spend` on the test set (`X_test`).
    *   Calculate and print the `sklearn.metrics.mean_absolute_error` and `sklearn.metrics.r2_score` for the test set predictions.

## Focus
Customer Lifetime Value Prediction (Regression) based on historical transactions and profile data.

## Dataset
Synthetic customer profiles and their historical transaction records.

## Hint
For SQL, define `GLOBAL_PREDICTION_CUTOFF_DATE` in Python first, then use it within your SQL query. Date calculations in SQLite (e.g., for `days_since_last_transaction_at_cutoff`) can leverage `julianday()` for differences and `MAX(CASE WHEN ... THEN ... END)` for conditional aggregates.
