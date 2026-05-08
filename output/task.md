# AI Daily Lab — 2026-05-08

## Task
Develop a machine learning pipeline to predict if a customer will **renew their subscription** at their next scheduled renewal point, based on their profile and recent application usage activity.

## Focus
Binary Classification for Customer Subscription Renewal

## Dataset
Simulated customer profiles and application usage logs for a SaaS product.

## Hint
1. **Data Generation**: Create three DataFrames: `customers_df` (static profiles), `usage_df` (event logs), and `prediction_cutoff_df`. The `prediction_cutoff_df` is crucial; it should contain `customer_id`, `cutoff_date` (the date immediately before their *next scheduled renewal*), and the `will_renew` binary target (1 if they renewed after `cutoff_date`, 0 otherwise). Simulate declining usage activity for non-renewing customers *prior to their `cutoff_date`*.
2. **SQL Feature Engineering**: Load `customers_df`, `usage_df`, and `prediction_cutoff_df` into SQLite. Your main query should `LEFT JOIN` `customers` with aggregated usage features derived *up to each customer's specific `cutoff_date`*. Use `julianday()` for date arithmetic in `WHERE` clauses for time-windowed aggregations. Remember `COALESCE` to handle `NULL` values from `LEFT JOIN` (e.g., 0 for counts/sums).
3. **Pandas FE & Target**: After fetching SQL results, handle `NaN`s for aggregated features (e.g., 0 for counts/sums, a large number like 9999 for `days_since_last_activity_at_cutoff`). Ensure `will_renew` is correctly merged as your target. Calculate `customer_tenure_at_cutoff_days` as a useful feature.
4. **ML Pipeline**: Use `sklearn.model_selection.train_test_split` with `stratify=y` to maintain class balance in training and testing sets, as renewal datasets can often be imbalanced. The `ColumnTransformer` should correctly apply `SimpleImputer` and `StandardScaler` to numerical features, and `OneHotEncoder` to categorical features.
