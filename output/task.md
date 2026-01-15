# AI Daily Lab — 2026-01-15

## Task
1. **Generate Synthetic Transactional Data**: Create a pandas DataFrame named `transactions_df` with 1000-1200 rows. Columns should include:
    *   `transaction_id` (unique integer IDs starting from 10000).
    *   `customer_id` (e.g., 100-150 distinct customer IDs, formatted as 'C' followed by a number).
    *   `transaction_date` (daily dates spanning 3 years, starting from '2021-01-01', ensuring multiple transactions per customer over time).
    *   `amount` (random float values, e.g., between 20.0 and 1000.0).
    *   Ensure the DataFrame is sorted by `customer_id` and then `transaction_date`.
2. **Load into SQLite**: Create an in-memory SQLite database using `sqlite3` and load the `transactions_df` into a table named `transactions`.
3. **SQL Analytics (Advanced Window Functions - LAG/LEAD and Date Difference)**: Write a single SQL query that performs the following for *each customer*, ordered by their `transaction_date`:
    *   Calculates `previous_transaction_amount`: The `amount` of the immediately preceding transaction.
    *   Calculates `next_transaction_amount`: The `amount` of the immediately following transaction.
    *   Calculates `days_since_prev_transaction`: The number of days between the current `transaction_date` and the `transaction_date` of the immediately preceding transaction. If it's the first transaction for a customer, this should be `NULL`.
    *   The query should return `transaction_id`, `customer_id`, `transaction_date`, `amount`, `previous_transaction_amount`, `next_transaction_amount`, and `days_since_prev_transaction`.
4. **Retrieve and Display**: Fetch the results into a pandas DataFrame. Display its head and briefly describe what the `LAG` and `LEAD` functions achieve in this context.

## Focus
SQL Analytics (LAG/LEAD Window Functions & Date Differences)

## Dataset
Synthetic transactional data

## Hint
For calculating date differences in SQLite, consider using `julianday()` function, e.g., `julianday(current_date) - julianday(previous_date)`. Remember to use `PARTITION BY customer_id ORDER BY transaction_date` for window functions.
