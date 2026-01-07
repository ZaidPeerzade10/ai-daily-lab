# AI Daily Lab — 2026-01-07

## Task
1. **Generate Synthetic Data**: Create two pandas DataFrames:
    *   `customers_df`: With 100-150 rows. Columns: `customer_id` (unique integers), `customer_name` (e.g., 'Customer A'), `registration_date` (random dates over the last 3 years).
    *   `orders_df`: With 800-1000 rows. Columns: `order_id` (unique integers), `customer_id` (randomly sampled from `customers_df` IDs, ensuring some customers have no orders and others have many), `order_date` (random dates after their `registration_date`), `amount` (random floats between 10 and 1000).
2. **Load into SQLite**: Create an in-memory SQLite database using `sqlite3` and load both `customers_df` and `orders_df` into tables named `customers` and `orders` respectively.
3. **SQL Query - Customer Sales Aggregation and Segmentation**: Write a single SQL query that performs the following:
    *   **Joins** the `customers` and `orders` tables.
    *   **Aggregates** to calculate the `total_sales_amount` and `number_of_orders` for each customer. Ensure that customers with no orders are still included in the result, showing 0 for sales and orders.
    *   **Categorizes** each customer into a `customer_segment` based on their `total_sales_amount`:
        *   'High-Value' if `total_sales_amount` > 5000
        *   'Medium-Value' if `total_sales_amount` between 1000 and 5000 (inclusive)
        *   'Low-Value' if `total_sales_amount` between 1 and 999 (inclusive)
        *   'No-Orders' if `total_sales_amount` is 0 or NULL.
    *   The query should return `customer_id`, `customer_name`, `registration_date`, `total_sales_amount`, `number_of_orders`, and `customer_segment`.
4. **Retrieve and Display**: Fetch the results into a pandas DataFrame. Display its head and then print the count of customers within each `customer_segment` to verify your segmentation logic.

## Focus
SQL analytics, Data manipulation (pandas & SQL), Conditional aggregation

## Dataset
Synthetic customer and order transactional data

## Hint
Use a `LEFT JOIN` to ensure all customers are included, even those without orders. The `COALESCE` function can be useful to handle `NULL` values from `LEFT JOIN` aggregations before applying `CASE WHEN` for segmentation.
