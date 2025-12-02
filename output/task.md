# AI Daily Lab — 2025-12-02

## Task
1. Create an in-memory SQLite database using the `sqlite3` module.
2. Create two tables: `customers` (columns: `customer_id` (INTEGER PRIMARY KEY), `name` (TEXT), `city` (TEXT)) and `orders` (columns: `order_id` (INTEGER PRIMARY KEY), `customer_id` (INTEGER), `amount` (REAL), `order_date` (TEXT)). Ensure `customer_id` in `orders` is a foreign key referencing `customers`.
3. Insert sample data into both tables (at least 5 distinct customers and 10-15 orders, ensuring some customers have multiple orders).
4. Using a single SQL query, calculate the total purchase `amount` for each customer, retrieving the customer's `name` and their `total_revenue`, joining the `customers` and `orders` tables.
5. Retrieve these aggregated results directly into a pandas DataFrame and display the top 3 customers by their `total_revenue`.

## Focus
SQL analytics, database interaction with Python, pandas integration

## Dataset
Synthetic data created in-memory with `sqlite3`

## Hint
Use `sqlite3` for database connection and table/data creation. Leverage `pandas.read_sql_query` to execute your SQL query and fetch results directly into a DataFrame. Your SQL query will need `JOIN`, `GROUP BY`, and `SUM` aggregate function.
