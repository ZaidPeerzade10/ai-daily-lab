# AI Daily Lab — 2026-06-26

## Task
Develop a machine learning pipeline to predict if a sales lead will **convert into a customer** within the next 30 days (binary classification), based on lead demographics, company attributes, and historical interaction patterns up to a specific cutoff date.

## Focus
Time-series feature engineering (SQL), binary classification, data imputation, scaling, one-hot encoding, model evaluation with imbalanced data.

## Dataset
Synthetic data for sales leads and their interactions.

## Hint
When creating synthetic `_actual_conversion_date`, ensure it's `pd.NaT` for non-converters. For SQL aggregation, use `julianday()` for date comparisons and `CASE WHEN` for conditional counts (e.g., `num_demo_requests_prev_60d`). Remember to `LEFT JOIN` historical aggregates to ensure all leads are present. For the binary target, `GLOBAL_PREDICTION_CUTOFF_DATE` and `_actual_conversion_date` will determine `will_convert_next_30d`.
