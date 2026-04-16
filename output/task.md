# AI Daily Lab — 2026-04-16

## Task
1. **Generate Synthetic Data (Pandas/Numpy)**: Create two pandas DataFrames:
    *   `products_df`: With 100-200 rows. Columns: `product_id` (unique integers), `category` (e.g., 'Electronics', 'Books', 'Apparel', 'Home Goods'), `base_price` (random floats 50.0-1000.0), `launch_date` (random dates over the last 3 years).
    *   `monthly_sales_df`: With 2000-4000 rows. For each product, generate 12-24 months of historical sales data starting from its `launch_date` up to a recent 'current_month'. Columns: `product_id` (sampled from `products_df` IDs), `month` (e.g., 'YYYY-MM' string, sequential for each product), `quantity_sold` (random integers 0-500), `marketing_spend_per_product` (random floats 0.0-1000.0).
    *   **Simulate realistic patterns**: Ensure `quantity_sold` is generally higher for products in 'Electronics' or with higher `marketing_spend_per_product`. Simulate some seasonality (e.g., certain categories might peak in specific months like 'Apparel' in '11'/'12'). Introduce some growth or decline trends over time for individual products.
    *   Sort `monthly_sales_df` by `product_id` then `month`.

2. **Load into SQLite & SQL Feature Engineering (Historical Sales Behavior)**: Create an in-memory SQLite database using `sqlite3`. Load `products_df` and `monthly_sales_df` into tables named `products` and `monthly_sales` respectively.
    Write a single SQL query that performs the following for *each product*, generating features based on its historical sales up to a defined `prediction_cutoff_month`.
    *   Define `prediction_cutoff_month` as the second to last month available in `monthly_sales` for each product. (This leaves one future month for the target variable).
    *   **Joins** `products` with an aggregated subquery for `monthly_sales`.
    *   **Aggregates features based on activities *within the 3 months ending at prediction_cutoff_month*** (i.e., `prediction_cutoff_month`, `prediction_cutoff_month-1`, `prediction_cutoff_month-2`):
        *   `avg_quantity_sold_prev_3m` (average of `quantity_sold`)
        *   `total_marketing_spend_prev_3m` (sum of `marketing_spend_per_product`)
        *   `sales_trend_3m` (Difference in `quantity_sold` between `prediction_cutoff_month` and `prediction_cutoff_month-2`).
    *   **Calculates a time-based feature**:
        *   `num_months_since_launch_at_cutoff`: Number of months from `launch_date` to `prediction_cutoff_month`.
    *   **Includes static product attributes**: `product_id`, `category`, `base_price`.
    *   **Ensures** all products are included (using `LEFT JOIN`), showing 0 for counts/sums if no sales data for that period. Fill `sales_trend_3m` with 0 if no sufficient history.
    *   The query should return `product_id`, `category`, `base_price`, and all aggregated/calculated features.
    *   **Hint**: Use CTEs or subqueries for aggregation over lagged months. `strftime('%Y-%m', date)` for month comparison. Date arithmetic with `DATE(...)` is tricky with month strings, so careful filtering based on month strings is key.

3. **Pandas Feature Engineering & Multi-Class Target Creation (Future Sales Growth Tier)**: Fetch the SQL query results into a pandas DataFrame (`product_features_df`).
    *   Handle `NaN` values: Fill `avg_quantity_sold_prev_3m`, `total_marketing_spend_prev_3m`, `sales_trend_3m` with 0.
    *   Calculate `sales_per_marketing_spend_3m`: `avg_quantity_sold_prev_3m` / (`total_marketing_spend_prev_3m` + 1e-6). Fill `NaN` or `inf` with 0.
    *   **Create the Multi-Class Target `next_month_sales_growth_tier`**: For *each product*, calculate its `quantity_sold` for the month *immediately following* the `prediction_cutoff_month` (`future_quantity_sold`) and its `quantity_sold` for `prediction_cutoff_month` (`current_quantity_sold`) from the *original* `monthly_sales_df`. Merge these, filling `NaN`s with 0 if no sales data.
        *   Calculate `growth_rate = (future_quantity_sold - current_quantity_sold) / (current_quantity_sold + 1e-6)`. Fill `NaN` or `inf` with 0.
        *   Define segments based on `growth_rate` percentiles (e.g., 25th, 50th, 75th percentiles of non-zero growth rates):
            *   'Significant Decline': `growth_rate` <= 25th percentile
            *   'Stable/Slight Decline': `growth_rate` > 25th percentile AND `growth_rate` <= 50th percentile
            *   'Moderate Growth': `growth_rate` > 50th percentile AND `growth_rate` <= 75th percentile
            *   'Strong Growth': `growth_rate` > 75th percentile
    *   Define features `X` (numerical: `avg_quantity_sold_prev_3m`, `total_marketing_spend_prev_3m`, `sales_trend_3m`, `num_months_since_launch_at_cutoff`, `sales_per_marketing_spend_3m`, `base_price`; categorical: `category`) and target `y` (`next_month_sales_growth_tier`). Split into training and testing sets (e.g., 70/30 split) using `sklearn.model_selection.train_test_split` (set `random_state=42`, `stratify` on `y` for class balance).

4. **Data Visualization**: Create two separate plots to visually inspect relationships with `next_month_sales_growth_tier`:
    *   A violin plot (or box plot) showing the distribution of `sales_trend_3m` for each sales growth tier. Ensure appropriate labels and titles.
    *   A stacked bar chart showing the proportion of `next_month_sales_growth_tier` across different `category` values. Ensure appropriate labels and titles.

5. **ML Pipeline & Evaluation (Multi-Class)**: 
    *   Create an `sklearn.pipeline.Pipeline` with a `sklearn.compose.ColumnTransformer` for preprocessing:
        *   For numerical features: Apply `sklearn.preprocessing.SimpleImputer(strategy='mean')` followed by `sklearn.preprocessing.StandardScaler`.
        *   For categorical features: Apply `sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')`.
    *   The final estimator in the pipeline should be `sklearn.ensemble.HistGradientBoostingClassifier` (set `random_state=42`).
    *   Train the pipeline on `X_train`, `y_train`. Predict `next_month_sales_growth_tier` for `X_test`.
    *   Calculate and print the `sklearn.metrics.accuracy_score` and a `sklearn.metrics.classification_report` for the test set predictions.

## Focus
Product Performance Prediction, Time-Series Feature Engineering, Multi-Class Classification, SQL Aggregation, Pipeline Development

## Dataset
Synthetic product and monthly sales data

## Hint
When generating `monthly_sales_df`, use `pandas.date_range` to create monthly periods and format them as 'YYYY-MM' strings. For SQL month comparisons and filtering relative months, careful string comparison or date arithmetic with SQLite's `DATE` function and `printf` can be used to construct month strings. For pandas, convert 'YYYY-MM' strings to datetime objects using `pd.to_datetime` for easier date arithmetic when calculating `num_months_since_launch_at_cutoff` and finding `prediction_cutoff_month` and future/current months.
