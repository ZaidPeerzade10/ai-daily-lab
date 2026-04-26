# AI Daily Lab — 2026-04-26

## Task
Develop a machine learning pipeline to predict if a product will run out of stock within the next 30 days, based on its current inventory, recent sales velocity, and product characteristics.

## Focus
Inventory Management, Binary Classification, Time-Series Aggregation, SQL & Pandas Feature Engineering

## Dataset
1. **Generate Synthetic Data (Pandas/Numpy)**: Create three pandas DataFrames:
    *   `products_df`: With 500-800 rows. Columns: `product_id` (unique integers), `category` (e.g., 'Electronics', 'Apparel', 'Books', 'Food'), `base_price` (random floats 10.0-500.0), `reorder_lead_time_days` (random integers 3-14).
    *   `daily_sales_df`: With 10000-15000 rows. Columns: `product_id` (randomly sampled from `products_df` IDs), `sale_date` (random dates over the last 6 months), `quantity_sold` (random integers 1-50).
    *   `current_inventory_df`: With 500-800 rows (one row per product). Columns: `product_id` (unique, matching `products_df`), `stock_on_hand` (random integers 100-2000).
    *   **Simulate realistic patterns**: Ensure `quantity_sold` is generally higher for products in 'Food' or 'Electronics' categories, and introduce some monthly seasonality (e.g., higher sales in November/December for 'Apparel'). For 10-15% of products, bias `stock_on_hand` to be relatively low compared to their average historical sales, making them more likely to be predicted as 'stock-out' candidates.
    *   Sort `daily_sales_df` by `product_id` then `sale_date`.

2. **Load into SQLite & SQL Feature Engineering (Time-Windowed Aggregations)**: Create an in-memory SQLite database using `sqlite3`. Load `products_df`, `daily_sales_df`, and `current_inventory_df` into tables named `products`, `daily_sales`, and `inventory` respectively.
    Define `analysis_date` as the maximum `sale_date` present in `daily_sales_df` across all products. This `analysis_date` represents 'today' for our prediction.
    Write a single SQL query that performs the following for *each product*:
    *   **Joins** `products` and `inventory` with an aggregated subquery for `daily_sales`.
    *   **Aggregates features based on activities *within the last 30 days ending at `analysis_date`***:
        *   `avg_sales_last_7d` (average of `quantity_sold` in the 7 days prior to or on `analysis_date`)
        *   `total_sales_last_30d` (sum of `quantity_sold` in the 30 days prior to or on `analysis_date`)
        *   `num_selling_days_last_30d` (count of distinct `sale_date`s in the 30 days prior to or on `analysis_date`)
    *   **Includes static product attributes and current inventory**: `product_id`, `category`, `base_price`, `reorder_lead_time_days`, `stock_on_hand`.
    *   **Ensures** all products are included (using `LEFT JOIN`), showing 0 for counts/sums and 0.0 for averages if no sales activity in the specified windows. Use `COALESCE` to handle `NULL`s.
    *   The query should return `product_id`, `category`, `base_price`, `reorder_lead_time_days`, `stock_on_hand`, and all the aggregated features.

3. **Pandas Feature Engineering & Binary Target Creation**: Fetch the SQL query results into a pandas DataFrame (`product_features_df`).
    *   Handle `NaN` values: Fill numerical aggregated features (`avg_sales_last_7d`, etc.) with 0 or 0.0 as appropriate.
    *   Calculate `sales_velocity_30d`: `total_sales_last_30d` / (`num_selling_days_last_30d` + 1e-6). Fill any `NaN` or `inf` with 0.
    *   Calculate `stock_to_avg_daily_sales_7d_ratio`: `stock_on_hand` / (`avg_sales_last_7d` * 30 + 1e-6). Fill any `NaN` or `inf` with a large reasonable number (e.g., 9999).
    *   **Create the Binary Target `will_stock_out_in_next_30_days`**: A product is considered to `will_stock_out_in_next_30_days = 1` if its `stock_on_hand` is less than or equal to (`avg_sales_last_7d` * 30). If `avg_sales_last_7d` is 0, the product will not stock out due to sales, so set its target to 0. Otherwise, `will_stock_out_in_next_30_days` is 0.
    *   Define features `X` (numerical: `base_price`, `reorder_lead_time_days`, `stock_on_hand`, `avg_sales_last_7d`, `total_sales_last_30d`, `num_selling_days_last_30d`, `sales_velocity_30d`, `stock_to_avg_daily_sales_7d_ratio`; categorical: `category`) and target `y` (`will_stock_out_in_next_30_days`). Split into training and testing sets (e.g., 70/30 split) using `sklearn.model_selection.train_test_split` (set `random_state=42`, `stratify` on `y` for class balance).

4. **Data Visualization**: Create two separate plots to visually inspect relationships with `will_stock_out_in_next_30_days`:
    *   A violin plot (or box plot) showing the distribution of `stock_to_avg_daily_sales_7d_ratio` for non-stock-out (0) vs. stock-out (1) predictions. Ensure appropriate labels and titles.
    *   A stacked bar chart showing the proportion of `will_stock_out_in_next_30_days` (0 or 1) across different `category` values. Ensure appropriate labels and titles.

5. **ML Pipeline & Evaluation (Binary Classification)**: 
    *   Create an `sklearn.pipeline.Pipeline` with a `sklearn.compose.ColumnTransformer` for preprocessing:
        *   For numerical features: Apply `sklearn.preprocessing.SimpleImputer(strategy='mean')` followed by `sklearn.preprocessing.StandardScaler`.
        *   For categorical features: Apply `sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')`.
    *   The final estimator in the pipeline should be `sklearn.ensemble.HistGradientBoostingClassifier` (set `random_state=42`).
    *   Train the pipeline on `X_train`, `y_train`. Predict probabilities for the positive class (class 1) on the test set (`X_test`).
    *   Calculate and print the `sklearn.metrics.roc_auc_score` and a `sklearn.metrics.classification_report` for the test set predictions.

## Hint
When generating `daily_sales_df`, consider how to create realistic sales trends (e.g., higher for certain categories, some seasonal peaks). For the SQL query, `julianday()` is useful for date arithmetic in `WHERE` clauses for the time-windowed aggregations. Remember to handle `LEFT JOIN` results carefully with `COALESCE` for aggregated features that might not exist for all products.
