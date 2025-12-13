# AI Daily Lab — 2025-12-13

## Task
1. Create an in-memory SQLite database using the `sqlite3` module.
2. Create a `transactions` table with the following columns: `transaction_id` (INTEGER PRIMARY KEY), `customer_id` (INTEGER), `product_id` (INTEGER), `transaction_date` (TEXT in 'YYYY-MM-DD' format), and `amount` (REAL).
3. Insert synthetic data into the `transactions` table. Include at least 5 distinct customers, 3 distinct products, and 20-30 transactions spanning a few months.
4. Write a single SQL query that uses **window functions** to calculate the following for each transaction:
    *   `customer_monthly_total`: The sum of `amount` for that specific `customer_id` within the month of the `transaction_date`.
    *   `customer_monthly_avg_transaction`: The average `amount` for that specific `customer_id` within the month of the `transaction_date`.
    *   `customer_cumulative_total`: The running total of `amount` for that specific `customer_id`, ordered by `transaction_date`.
5. Retrieve the results of this SQL query into a pandas DataFrame. Display `transaction_date`, `customer_id`, `amount`, `customer_monthly_total`, `customer_monthly_avg_transaction`, and `customer_cumulative_total`. Show the head of the DataFrame.

## Focus
SQL analytics

## Dataset
Synthetic `transactions` data generated in-memory.

## Hint
Use `strftime('%Y-%m', transaction_date)` to extract the month for window partitioning. For monthly aggregates, use `PARTITION BY customer_id, strftime('%Y-%m', transaction_date)`. For cumulative sums, use `PARTITION BY customer_id ORDER BY transaction_date`.
