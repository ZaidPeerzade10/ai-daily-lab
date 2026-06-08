# AI Daily Lab — 2026-06-08

## Task
Develop a machine learning pipeline to predict if an individual e-commerce order item will be **returned** (binary classification), based on product attributes, customer profile, and historical return patterns up to the order date.

## Focus
SQL time-windowed aggregations, binary classification with imbalanced data, end-to-end ML pipeline, feature engineering from multi-table synthetic data.

## Dataset
Synthetic e-commerce data: customers, products, and order items with return status.

## Hint
When generating `orders_df`, simulate a slightly higher return rate for certain product categories, lower-tier loyalty customers, or higher-priced items. For the SQL aggregations, remember to filter historical data in CTEs strictly *before or on* the `GLOBAL_PREDICTION_CUTOFF_DATE` and ensure `is_returned` is treated as a numerical value (0 or 1) for averages. Use `CAST(SUM(CASE WHEN is_returned = 1 THEN 1 ELSE 0 END) AS REAL) / (COUNT(*) + 1e-6)` for calculating return rates to handle potential zero denominators.
