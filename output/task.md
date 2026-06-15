# AI Daily Lab — 2026-06-15

## Task
Develop a machine learning pipeline to predict the **supplier risk category** ('Low', 'Medium', 'High') for an active supplier, based on their profile, historical delivery performance, and product quality up to a specific cutoff date.

1.  **Generate Synthetic Data (Pandas/Numpy)**: Create two pandas DataFrames:
    *   `suppliers_df`: With 100-200 rows. Columns: `supplier_id` (unique integers), `region` (e.g., 'North', 'South', 'East', 'West'), `supplier_tier` (e.g., 'Preferred', 'Standard', 'Backup'), `contract_start_date` (random dates over the last 3-7 years), `_actual_risk_score` (random floats 0-100 for target creation, simulating underlying risk from various factors).
    *   `deliveries_df`: With 10000-15000 rows. Columns: `delivery_id` (unique integers), `supplier_id` (randomly sampled from `suppliers_df` IDs), `order_date` (random datetimes occurring *after* `contract_start_date`), `scheduled_delivery_date` (random datetimes *after* `order_date`), `delivery_date` (random datetimes *after* `scheduled_delivery_date` for delays, or slightly before for early deliveries), `quality_score` (random integers 1-5), `quantity_ordered` (random integers 10-1000).
    *   **Simulate realistic patterns**: Ensure `order_date` is always after `contract_start_date`. `delivery_date` should usually be on or after `scheduled_delivery_date`. Higher `_actual_risk_score` should correlate with more frequent delivery delays (`delivery_date` > `scheduled_delivery_date`) and lower `quality_score`. 'Backup' `supplier_tier` might have higher `_actual_risk_score`. `region` might influence risk. Ensure a good mix of historical delivery scenarios.
    *   Sort `deliveries_df` by `supplier_id` then `order_date`.

2.  **Load into SQLite & SQL Feature Engineering (Time-Windowed Aggregations)**: Create an in-memory SQLite database using `sqlite3`. Load `suppliers_df` and `deliveries_df` into tables named `suppliers` and `deliveries` respectively.
    *   Define `GLOBAL_PREDICTION_CUTOFF_DATE` as 4 weeks prior to the latest `delivery_date` in your generated `deliveries_df`.
    *   Write a single SQL query that performs the following for *each supplier* active up to `GLOBAL_PREDICTION_CUTOFF_DATE`:
        *   Includes static attributes: `supplier_id`, `region`, `supplier_tier`, `contract_start_date`, and the `_actual_risk_score` from `suppliers` table.
        *   Aggregates historical features *for the respective `supplier_id` in the 90 days preceding or on `GLOBAL_PREDICTION_CUTOFF_DATE`*:
            *   `avg_delivery_delay_days_prev_90d`: Average of `(julianday(d.delivery_date) - julianday(d.scheduled_delivery_date))` for deliveries *where `delivery_date` >= `scheduled_delivery_date`* and `delivery_date` is *before or on* `GLOBAL_PREDICTION_CUTOFF_DATE`. Return 0.0 if no relevant deliveries.
            *   `avg_quality_score_prev_90d`: Average `quality_score` for deliveries *before or on* `GLOBAL_PREDICTION_CUTOFF_DATE`.
            *   `num_deliveries_prev_90d`: Count of deliveries *before or on* `GLOBAL_PREDICTION_CUTOFF_DATE`.
            *   `total_quantity_ordered_prev_90d`: Sum of `quantity_ordered` for deliveries *before or on* `GLOBAL_PREDICTION_CUTOFF_DATE`.
            *   `days_since_last_delivery_at_cutoff`: Number of days between `GLOBAL_PREDICTION_CUTOFF_DATE` and the most recent `delivery_date` for this `supplier_id` *before or on* the cutoff. Return 9999 if no prior deliveries.
    *   **Ensures** all active suppliers are included (using `LEFT JOIN`), showing 0 for counts/sums and 0.0 for averages if no activity in the window. Handle `NULL`s appropriately.
    *   The query should return all mentioned fields.
    *   **Hint**: Use CTEs for pre-calculating supplier-level historical aggregates up to `GLOBAL_PREDICTION_CUTOFF_DATE`. Use `julianday()` for date comparisons.

3.  **Pandas Feature Engineering & Multi-class Target Creation**: Fetch the SQL query results into a pandas DataFrame (`supplier_features_df`).
    *   Convert `contract_start_date` to datetime objects.
    *   Handle `NaN` values: Fill numerical historical aggregated features (e.g., averages, sums, counts) with 0.0 or 0 as appropriate. Fill `days_since_last_delivery_at_cutoff` with 9999.
    *   Calculate `supplier_tenure_at_cutoff_days`: Number of days between `contract_start_date` and `GLOBAL_PREDICTION_CUTOFF_DATE`.
    *   **Create the Multi-class Target `risk_category`**: Based on `_actual_risk_score`:
        *   'Low': `_actual_risk_score` <= 33rd percentile
        *   'Medium': `_actual_risk_score` > 33rd percentile and <= 66th percentile
        *   'High': `_actual_risk_score` > 66th percentile
        (Adjust percentile thresholds dynamically to ensure a reasonable class balance).
    *   Define features `X` (numerical: `avg_delivery_delay_days_prev_90d`, `avg_quality_score_prev_90d`, `num_deliveries_prev_90d`, `total_quantity_ordered_prev_90d`, `days_since_last_delivery_at_cutoff`, `supplier_tenure_at_cutoff_days`; categorical: `region`, `supplier_tier`) and target `y` (`risk_category`).
    *   Split into training and testing sets (e.g., 70/30 split) using `sklearn.model_selection.train_test_split` (set `random_state=42`, `stratify` on `y` for class balance).

4.  **Data Visualization (Matplotlib/Seaborn)**: Create two separate plots to visually inspect relationships with `risk_category`:
    *   A violin plot (or box plot) showing the distribution of `avg_delivery_delay_days_prev_90d` for each `risk_category`. Ensure appropriate labels and titles.
    *   A stacked bar chart showing the proportion of `risk_category` across different `supplier_tier` values. Ensure appropriate labels and titles.

5.  **ML Pipeline & Evaluation (Multi-class Classification)**:
    *   Create an `sklearn.pipeline.Pipeline` with a `sklearn.compose.ColumnTransformer` for preprocessing:
        *   For numerical features: Apply `sklearn.preprocessing.SimpleImputer(strategy='mean')` followed by `sklearn.preprocessing.StandardScaler`.
        *   For categorical features: Apply `sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')`.
    *   The final estimator in the pipeline should be `sklearn.ensemble.HistGradientBoostingClassifier` (set `random_state=42`).
    *   Train the pipeline on `X_train`, `y_train`. Predict the `risk_category` on the test set (`X_test`).
    *   Calculate and print a `sklearn.metrics.classification_report` for the test set predictions.

## Focus
Predicting multi-class supplier risk based on historical operational performance using time-windowed SQL aggregations and a complete ML pipeline.

## Dataset
Synthetic data for suppliers and their delivery/quality performance.

## Hint
When calculating `avg_delivery_delay_days_prev_90d` in SQL, ensure you correctly filter for deliveries that were not early (`delivery_date >= scheduled_delivery_date`) within the 90-day window. Use `COALESCE` or `IFNULL` for handling potential `NULL` values from `AVG` or `SUM` on empty sets in aggregates. Pandas' `pd.qcut` is useful for creating percentile-based target categories.
