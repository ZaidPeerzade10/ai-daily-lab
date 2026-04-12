# AI Daily Lab — 2026-04-12

## Task
1. **Generate Synthetic Data (Pandas/Numpy)**: Create three pandas DataFrames:
    *   `customers_df`: With 1000-1500 rows. Columns: `customer_id` (unique integers), `signup_date` (random dates over the last 3 years), `customer_segment` (e.g., 'Bronze', 'Silver', 'Gold', 'Platinum').
    *   `warehouses_df`: With 10-15 rows. Columns: `warehouse_id` (unique integers), `location_city` (e.g., 'New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'), `operational_capacity_pct` (random floats 70.0-100.0).
    *   `orders_df`: With 15000-20000 rows. Columns: `order_id` (unique integers), `customer_id` (randomly sampled from `customers_df` IDs), `order_date` (random dates occurring *after* their respective `signup_date`), `total_order_value` (random floats 20.0-500.0), `shipping_method` (e.g., 'Standard', 'Express', 'Priority'), `warehouse_id` (randomly sampled from `warehouses_df` IDs), `destination_region` (e.g., 'Northeast', 'Southeast', 'Midwest', 'West', 'Southwest').
    *   **Simulate realistic delivery patterns and delays**: 
        *   Define base expected delivery days: 'Standard' (6 days), 'Express' (3 days), 'Priority' (1.5 days).
        *   Calculate `actual_delivery_date` for each order: `order_date + base_expected_days + random_noise_days`.
        *   Bias `random_noise_days` such that:
            *   Orders with 'Standard' `shipping_method` or to 'Midwest' `destination_region` are slightly more prone to larger `random_noise_days` (e.g., +1 to +3 days).
            *   Orders from warehouses with lower `operational_capacity_pct` (e.g., < 80%) are more likely to experience delays (increase `random_noise_days`).
            *   Orders from 'Bronze' `customer_segment` might also have higher delay risk (slightly increased `random_noise_days`).
            *   Overall delay rate should be 15-25%. 
        *   Add `is_delayed` (binary, 0 or 1) as a column to `orders_df`. A shipment `is_delayed=1` if `actual_delivery_date` is more than (`base_expected_days` + 1.5 days) from `order_date`. 
    *   Sort `orders_df` by `customer_id` then `order_date`.

2. **Load into SQLite & SQL Feature Engineering (Order-Level Attributes)**: Create an in-memory SQLite database using `sqlite3`. Load `customers_df`, `warehouses_df`, and `orders_df` into tables named `customers`, `warehouses`, and `orders` respectively.
    Write a single SQL query that performs the following for *each order*:
    *   **Joins** `orders`, `customers`, and `warehouses` data.
    *   **Extracts/Calculates features**:
        *   `order_day_of_week`: Day of the week when the order was placed (1=Monday, 7=Sunday). (SQLite `strftime('%w', ...)` returns 0 for Sunday, adjust if needed).
        *   `order_month`: Month when the order was placed.
        *   `days_since_customer_signup_at_order`: Number of days between `signup_date` and `order_date`.
        *   `operational_capacity_pct` (from `warehouses`).
        *   `actual_delivery_date` (from `orders`).
    *   **Includes static order and customer attributes**: `order_id`, `customer_id`, `order_date`, `total_order_value`, `shipping_method`, `destination_region`, `is_delayed` (the target), `customer_segment`.
    *   The query should return all these attributes and engineered features.
    *   **Hint**: Use `julianday()` for date differences. `strftime('%w', order_date)` for day of week, `strftime('%m', order_date)` for month.

3. **Pandas Feature Engineering & Binary Target Creation**: Fetch the SQL query results into a pandas DataFrame (`order_features_df`).
    *   Convert `order_date` and `actual_delivery_date` to datetime objects.
    *   Calculate `delivery_lead_time_actual_days`: Number of days between `order_date` and `actual_delivery_date`.
    *   Calculate `expected_delivery_days` for each order based on `shipping_method` (e.g., 'Standard': 6, 'Express': 3, 'Priority': 1.5). Fill any `NaN`s with 0.0 if a method is missing.
    *   Calculate `delivery_speed_ratio`: `delivery_lead_time_actual_days` / (`expected_delivery_days` + 0.01) (add a small epsilon to prevent division by zero). Fill any `NaN`s or `inf` with 0.0 or a reasonable maximum.
    *   Calculate `order_value_per_day_since_signup`: `total_order_value` / (`days_since_customer_signup_at_order` + 1). Fill any `NaN`s with 0.0.
    *   Define features `X` (numerical: `total_order_value`, `order_day_of_week`, `order_month`, `days_since_customer_signup_at_order`, `operational_capacity_pct`, `delivery_lead_time_actual_days`, `expected_delivery_days`, `delivery_speed_ratio`, `order_value_per_day_since_signup`; categorical: `shipping_method`, `destination_region`, `customer_segment`) and target `y` (`is_delayed`). Split into training and testing sets (e.g., 70/30 split) using `sklearn.model_selection.train_test_split` (set `random_state=42`, `stratify` on `y` for class balance).

4. **Data Visualization**: Create two separate plots to visually inspect relationships with `is_delayed`:
    *   A violin plot (or box plot) showing the distribution of `delivery_lead_time_actual_days` for non-delayed (0) vs. delayed (1) orders. Ensure appropriate labels and titles.
    *   A stacked bar chart showing the proportion of `is_delayed` (0 or 1) across different `shipping_method` values. Ensure appropriate labels and titles.

5. **ML Pipeline & Evaluation (Binary Classification)**: 
    *   Create an `sklearn.pipeline.Pipeline` with a `sklearn.compose.ColumnTransformer` for preprocessing:
        *   For numerical features: Apply `sklearn.preprocessing.SimpleImputer(strategy='mean')` followed by `sklearn.preprocessing.StandardScaler`.
        *   For categorical features: Apply `sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')`.
    *   The final estimator in the pipeline should be `sklearn.ensemble.HistGradientBoostingClassifier` (set `random_state=42`).
    *   Train the pipeline on `X_train`, `y_train`. Predict probabilities for the positive class (class 1) on the test set (`X_test`).
    *   Calculate and print the `sklearn.metrics.roc_auc_score` and a `sklearn.metrics.classification_report` for the test set predictions.

## Focus
Predicting Shipment Delay Risk (Binary Classification) using customer, warehouse, and order details.

## Dataset
Synthetic `customers`, `warehouses`, and `orders` dataframes.

## Hint
Pay close attention to simulating `actual_delivery_date` and deriving `is_delayed` to ensure a realistic and balanced target. For SQL, remember to adjust `strftime('%w', ...)` output for 1-7 day numbering if desired. In Pandas, ensure proper handling of datetime objects for date differences and numerical stability for ratio calculations.
