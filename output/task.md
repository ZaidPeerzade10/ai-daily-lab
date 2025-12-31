# AI Daily Lab — 2025-12-31

## Task
1. Generate a synthetic transactional dataset using `pandas`:
    *   Create a DataFrame named `transactions_df` with 800-1000 rows.
    *   Columns should include:
        *   `transaction_id` (unique integer IDs).
        *   `customer_id` (e.g., 50-100 distinct customer IDs).
        *   `transaction_date` (daily dates spanning 1-2 years, starting from '2022-01-01', ensuring multiple transactions per day/customer).
        *   `amount` (random float values, e.g., between 10.0 and 500.0).
        *   `product_category` (3-5 distinct string categories, e.g., 'Electronics', 'Books', 'Groceries', 'Clothing').
    *   Sort the DataFrame by `customer_id` and then `transaction_date`.
2. Create an in-memory SQLite database using `sqlite3` and load the `transactions_df` into a table named `transactions`.
3. **SQL Analytics (Window Function 1 - Running Total)**: Write an SQL query to calculate the `running_total_amount` for each customer, ordered by their `transaction_date`. The query should return `transaction_id`, `customer_id`, `transaction_date`, `amount`, and the new `running_total_amount` column. Retrieve the results into a pandas DataFrame and display its head.
4. **SQL Analytics (Window Function 2 - Rank within Group)**: Write a *separate* SQL query to rank transactions by `amount` in descending order *within each `product_category`*. The query should return `transaction_id`, `product_category`, `amount`, and the new `rank_in_category` column. Retrieve these results into another pandas DataFrame and display its head.
5. Briefly describe what each window function achieves and how it's useful in analytical contexts.

## Focus
SQL Analytics

## Dataset
Synthetic transactional data generated with pandas.

## Hint
Remember to use `pandas.to_sql()` to transfer data to SQLite. For window functions, look into `SUM() OVER (PARTITION BY ... ORDER BY ...)` for running totals and `RANK() OVER (PARTITION BY ... ORDER BY ...) ` or `DENSE_RANK() OVER (...)` for ranking.
