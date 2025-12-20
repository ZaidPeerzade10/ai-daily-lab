# AI Daily Lab — 2025-12-20

## Task
1. Create an in-memory SQLite database using the `sqlite3` module.
2. Create two tables:
    *   `customers` with columns: `customer_id` (INTEGER PRIMARY KEY), `name` (TEXT), `region` (TEXT).
    *   `orders` with columns: `order_id` (INTEGER PRIMARY KEY), `customer_id` (INTEGER, FOREIGN KEY), `order_date` (TEXT in 'YYYY-MM-DD' format), `total_amount` (REAL).
3. Insert synthetic data into both tables. Ensure you have at least 5 distinct customers, 3 distinct regions, and 20-30 orders spanning a few months, with some customers having multiple orders.
4. Write a single SQL query to find the `customer_id`, `name`, `region`, and `total_spent` (sum of all their `total_amount`s) for customers whose `total_spent` is greater than the `average_order_value_per_region` (the average `total_amount` of all orders originating from their specific `region`). Order the results by `total_spent` in descending order.
5. Retrieve the results of this SQL query into a pandas DataFrame and display the head of the DataFrame.

## Focus
SQL analytics

## Dataset
Synthetic data created directly in SQLite database tables.

## Hint
Consider using a `JOIN` to combine customer and order information. For the `average_order_value_per_region`, a Common Table Expression (CTE) or a subquery can be helpful to first compute regional averages, which can then be joined or used in a `HAVING` clause for filtering.
