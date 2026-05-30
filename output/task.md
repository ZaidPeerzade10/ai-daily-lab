# AI Daily Lab — 2026-05-30

## Task
Develop a machine learning pipeline to predict the **demand category** ('Low', 'Medium', 'High') for an individual e-commerce product in the next 14 days, based on its attributes and historical sales data up to a specific cutoff date.

## Focus
Inventory Management, Demand Forecasting, Multi-class Classification, Time-Series Feature Engineering

## Dataset
Synthetic e-commerce product and sales data.

## Hint
1.  **Synthetic Data Generation (Pandas/Numpy)**: Create two pandas DataFrames:
    *   `products_df`: With 1000-1500 rows. Columns: `product_id` (unique integers), `category` (e.g., 'Electronics', 'Apparel', 'Home Goods'), `brand` (e.g., 'BrandX', 'BrandY', 'BrandZ'), `base_price` (random floats 10-1000), `launch_date` (random dates over the last 3-5 years).
    *   `sales_df`: With 20000-30000 rows. Columns: `sale_id` (unique integers), `product_id` (randomly sampled from `products_df` IDs), `sale_date` (random datetimes occurring *after* their respective `launch_date` and up to `pd.Timestamp.now() - pd.Timedelta(weeks=2)`), `quantity_sold` (random integers 1-20).
    *   **Simulate realistic patterns**: Ensure `sale_date` is always after `launch_date`. Simulate varying sales patterns: some products with consistently high demand, others low, and some with mild seasonality. `quantity_sold` should vary realistically. Some product categories/brands should naturally have higher sales. Ensure sales for a product gradually decrease if it's an older product. Sort `sales_df` by `product_id` then `sale_date`.

2.  **Load into SQLite & SQL Feature Engineering (Time-Windowed Sales Aggregations)**: Create an in-memory SQLite database using `sqlite3`. Load `products_df` and `sales_df` into tables named `products` and `sales` respectively.
    *   Define `GLOBAL_PREDICTION_CUTOFF_DATE` as 2 months prior to the latest `sale_date` in your generated `sales_df` (e.g., `sales_df['sale_date'].max() - pd.Timedelta(months=2)`).
    *   Write a single SQL query that performs the following for *each product*, aggregating its sales behavior *within the 30 days immediately preceding `GLOBAL_PREDICTION_CUTOFF_DATE`*:
        *   `current_cutoff_date` (the `GLOBAL_PREDICTION_CUTOFF_DATE` itself, for consistency).
        *   `avg_qty_sold_prev_30d` (average `quantity_sold`).
        *   `total_qty_sold_prev_30d` (sum of `quantity_sold`).
        *   `num_sales_prev_30d` (count of sales).
        *   `days_since_last_sale_at_cutoff`: Number of days between `GLOBAL_PREDICTION_CUTOFF_DATE` and the most recent `sale_date` *before or on* the cutoff. Return a large number (e.g., 9999) if no sales before cutoff.
    *   **Includes static product attributes**: `product_id`, `category`, `brand`, `base_price`, `launch_date`.
    *   **Ensures** all products are included (using `LEFT JOIN`), showing 0 for counts/sums and 0.0 for averages if no sales in the 30-day window. Handle `NULL`s appropriately.
    *   The query should return all mentioned fields.

3.  **Pandas Feature Engineering & Multi-class Target Creation**: Fetch the SQL query results into a pandas DataFrame (`product_features_df`).
    *   Convert `launch_date` and `current_cutoff_date` to datetime objects.
    *   Handle `NaN` values: Fill numerical aggregated features with 0 or 0.0 as appropriate. Fill `days_since_last_sale_at_cutoff` with 9999. Fill `base_price` NaNs with median.
    *   Calculate `product_age_at_cutoff_days`: Number of days between `launch_date` and `current_cutoff_date`.
    *   **Create the Multi-class Target `next_14d_demand_category`**: For *each product*, sum their `quantity_sold` from the `sales_df` for all sales that occurred *after* `current_cutoff_date` and *before or on* `current_cutoff_date + pd.Timedelta(days=14)`. Let this sum be `next_14d_total_qty_sold`.
    *   Merge this `next_14d_total_qty_sold` into `product_features_df`, filling `NaN`s with 0 for products with no sales in the target window.
    *   Categorize `next_14d_total_qty_sold` into:
        *   'Low': if `next_14d_total_qty_sold` <= 10
        *   'Medium': if 10 < `next_14d_total_qty_sold` <= 50
        *   'High': if `next_14d_total_qty_sold` > 50
        (Adjust these thresholds based on your data distribution to ensure a reasonable class balance).
    *   Define features `X` (numerical: `base_price`, `avg_qty_sold_prev_30d`, `total_qty_sold_prev_30d`, `num_sales_prev_30d`, `days_since_last_sale_at_cutoff`, `product_age_at_cutoff_days`; categorical: `category`, `brand`) and target `y` (`next_14d_demand_category`).
    *   Split into training and testing sets (e.g., 70/30 split) using `sklearn.model_selection.train_test_split` (set `random_state=42`, `stratify` on `y` for class balance).

4.  **Data Visualization (Matplotlib/Seaborn)**: Create two separate plots to visually inspect relationships with `next_14d_demand_category`:
    *   A violin plot (or box plot) showing the distribution of `total_qty_sold_prev_30d` for each `next_14d_demand_category`. Ensure appropriate labels and titles.
    *   A stacked bar chart showing the proportion of `next_14d_demand_category` (across 'Low', 'Medium', 'High') for different `category` values. Ensure appropriate labels and titles.

5.  **ML Pipeline & Evaluation (Multi-class Classification)**: 
    *   Create an `sklearn.pipeline.Pipeline` with a `sklearn.compose.ColumnTransformer` for preprocessing:
        *   For numerical features: Apply `sklearn.preprocessing.SimpleImputer(strategy='mean')` followed by `sklearn.preprocessing.StandardScaler`.
        *   For categorical features: Apply `sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')`.
    *   The final estimator in the pipeline should be `sklearn.ensemble.HistGradientBoostingClassifier` (set `random_state=42`).
    *   Train the pipeline on `X_train`, `y_train`. Predict the `next_14d_demand_category` on the test set (`X_test`).
    *   Calculate and print a `sklearn.metrics.classification_report` for the test set predictions.
