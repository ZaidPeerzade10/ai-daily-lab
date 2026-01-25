# AI Daily Lab — 2026-01-25

## Task
1. **Generate Synthetic Data (Pandas/Numpy)**: Create two pandas DataFrames:
    *   `customers_df`: With 500 rows. Columns: `customer_id` (unique integers), `signup_date` (random dates over the last 5 years), `region` (e.g., 'North', 'South', 'East', 'West'), `age` (random integers 18-70).
    *   `transactions_df`: With 3000-5000 rows. Columns: `transaction_id` (unique integers), `customer_id` (randomly sampled from `customers_df` IDs, ensuring some customers have only 1 transaction, others 2, and many more), `transaction_date` (random dates *after* their respective `signup_date`), `amount` (random floats between 10.0 and 1000.0), `item_count` (random integers 1-10).
    Ensure `transactions_df` is sorted by `customer_id` and `transaction_date`.

2. **Load into SQLite & SQL Feature Engineering (Initial Buyer Behavior)**: Create an in-memory SQLite database. Load `customers_df` into a table named `customers` and `transactions_df` into a table named `transactions`. Define an `initial_period_days` (e.g., 30 days).
    Write a single SQL query that performs the following for *each customer*:
    *   **Joins** `customers` and `transactions` tables.
    *   **Aggregates** initial purchase behavior for transactions occurring within the `initial_period_days` from their `signup_date`:
        *   `initial_total_spend` (sum of `amount`)
        *   `initial_num_transactions` (count of transactions)
        *   `initial_avg_item_count` (average `item_count`)
    *   **Ensures** all customers are included, showing 0 for aggregates if no transactions in the initial period.
    *   The query should return `customer_id`, `region`, `age`, `signup_date`, `initial_total_spend`, `initial_num_transactions`, `initial_avg_item_count`.

3. **Pandas Feature Engineering & Target Creation (Repeat Buyer)**: Fetch the SQL query results into a pandas DataFrame. Merge it with the original `transactions_df` to get full transaction history (if not already joined properly).
    *   Handle `NaN` values: Fill `initial_total_spend`, `initial_num_transactions`, `initial_avg_item_count` with 0 for customers with no initial transactions.
    *   Calculate `account_age_days`: The number of days between `signup_date` and the latest `transaction_date` in the *entire* `transactions_df`.
    *   **Create Binary Target `is_repeat_buyer`**: A customer is a `repeat_buyer` (1) if they have made at least 2 transactions *and* the `MAX(transaction_date)` is at least 60 days after their `MIN(transaction_date)`. Otherwise, `0`.
    *   Define features `X` (`region`, `age`, `account_age_days`, `initial_total_spend`, `initial_num_transactions`, `initial_avg_item_count`) and target `y` (`is_repeat_buyer`). Split into training and testing sets (e.g., 70/30 split) using `sklearn.model_selection.train_test_split` (set `random_state=42`).

4. **Data Visualization**: Create two separate plots to visually inspect relationships with `is_repeat_buyer`:
    *   A violin plot or box plot showing the distribution of `initial_total_spend` for each `is_repeat_buyer` group.
    *   A bar plot or count plot showing the distribution of `is_repeat_buyer` across different `region`s.
    Ensure plots have appropriate labels and titles.

5. **ML Pipeline & Evaluation**: Create an `sklearn.pipeline.Pipeline` with a `ColumnTransformer` for preprocessing:
    *   For numerical features (`age`, `account_age_days`, `initial_total_spend`, `initial_num_transactions`, `initial_avg_item_count`): Apply `sklearn.preprocessing.StandardScaler`.
    *   For categorical features (`region`): Apply `sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')`.
    *   The final estimator should be `sklearn.linear_model.LogisticRegression` (set `random_state=42`, `solver='liblinear'` for reproducibility).
    *   Train the pipeline on `X_train`, `y_train`. Predict probabilities for the positive class (class 1) on `X_test`.
    *   Calculate and print the `sklearn.metrics.roc_auc_score` and `sklearn.metrics.classification_report` for the test set predictions.

## Focus
SQL-driven feature engineering from initial activity, binary classification for 'repeat buyer' prediction, ML pipeline with ColumnTransformer, and relevant visualizations.

## Dataset
Synthetic customer profiles and transactional data.

## Hint
When defining the `is_repeat_buyer` target, you'll need to group the full `transactions_df` by `customer_id` to count total transactions and calculate date differences. For the SQL initial period aggregation, use a `WHERE` clause based on `transaction_date` and `signup_date`.
