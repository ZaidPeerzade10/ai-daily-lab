# AI Daily Lab — 2026-04-22

## Task
1. **Generate Synthetic Data (Pandas/Numpy)**: Create three pandas DataFrames:
    *   `machines_df`: With 500-700 rows. Columns: `machine_id` (unique integers), `machine_type` (e.g., 'TypeA', 'TypeB', 'TypeC'), `location` (e.g., 'North', 'South', 'East', 'West'), `installation_date` (random dates over the last 3 years).
    *   `telemetry_df`: With 15000-25000 rows. Columns: `telemetry_id` (unique integers), `machine_id` (randomly sampled from `machines_df` IDs), `timestamp` (random timestamps *after* their respective `installation_date`), `temperature` (random floats 20-100), `vibration` (random floats 0-10).
    *   `maintenance_df`: With 2000-3000 rows. Columns: `maintenance_id` (unique integers), `machine_id` (randomly sampled from `machines_df` IDs), `maintenance_date` (random dates *after* their respective `installation_date`), `maintenance_type` (e.g., 'Routine Check', 'Lubrication', 'Component Repair', 'Software Update').
    *   **Simulate realistic patterns**: Ensure `timestamp` is always after `installation_date`.
        *   For 20-30% of machines, introduce a `major_error_date` (random date *after* `installation_date` and within the last 6 months). For these machines, `temperature` and `vibration` should show a noticeable increasing trend (e.g., +5-15% increase) in the 7-14 days *before* this `major_error_date`.
        *   'TypeA' machines might have slightly higher baseline `vibration` and 'TypeC' higher `temperature`.
        *   `maintenance_type='Component Repair'` should happen *after* some error-like sensor readings.
    *   Sort `telemetry_df` by `machine_id` then `timestamp`. Sort `maintenance_df` by `machine_id` then `maintenance_date`.

2. **Load into SQLite & SQL Feature Engineering (Time-Windowed Aggregations)**: Create an in-memory SQLite database using `sqlite3`. Load `machines_df`, `telemetry_df`, and `maintenance_df` into tables named `machines`, `telemetry`, and `maintenance` respectively.
    Write a single SQL query that performs the following for *each machine*:
    *   First, determine the `latest_telemetry_date` for each machine (the maximum `timestamp` in the `telemetry` table for that machine).
    *   **Joins** `machines` with aggregated subqueries for `telemetry` and `maintenance`.
    *   **Aggregates features based on activities *within the 30 days ending at `latest_telemetry_date`***:
        *   `current_observation_date` (the `latest_telemetry_date` for the machine).
        *   `avg_temp_prev_30d` (average of `temperature`)
        *   `max_vibration_prev_30d` (maximum of `vibration`)
        *   `num_telemetry_readings_prev_30d` (count of `telemetry_id`s)
        *   `num_maintenances_prev_30d` (count of `maintenance_id`s)
        *   `num_repairs_prev_30d` (count of `maintenance_id`s where `maintenance_type` is 'Component Repair').
    *   **Includes static machine attributes**: `machine_id`, `machine_type`, `location`, `installation_date`.
    *   **Ensures** all machines are included (using `LEFT JOIN`), showing 0 for counts/sums and 0.0 for averages if no activity in the 30-day window.
    *   The query should return `machine_id`, `machine_type`, `location`, `installation_date`, `current_observation_date`, and all the aggregated features.
    *   **Hint**: Use CTEs to find `latest_telemetry_date` first. Use `julianday()` for date comparisons in `WHERE` clauses. Aggregate features using `COALESCE` to handle `NULL`s from `LEFT JOIN`s.

3. **Pandas Feature Engineering & Binary Target Creation (Major Error Prediction)**: Fetch the SQL query results into a pandas DataFrame (`machine_features_df`).
    *   Handle `NaN` values: Fill numerical aggregated features (`avg_temp_prev_30d`, etc.) with 0 or 0.0 as appropriate.
    *   Convert `installation_date` and `current_observation_date` to datetime objects.
    *   Calculate `days_since_installation_at_obs`: Number of days between `installation_date` and `current_observation_date`.
    *   Calculate `telemetry_frequency_prev_30d`: `num_telemetry_readings_prev_30d` / 30.0. Fill any `NaN` or `inf` with 0.
    *   **Create the Binary Target `major_error_in_next_7_days`**: For *each machine*, determine if its simulated `major_error_date` (from the original `machines_df` which should contain this column for certain machines) falls within the 7-day period *immediately following* its `current_observation_date`. Merge this aggregate (1 if yes, 0 if no) with `machine_features_df`, filling `NaN`s with 0 for machines without a simulated `major_error_date` or whose error date falls outside the window.
    *   Define features `X` (numerical: `avg_temp_prev_30d`, `max_vibration_prev_30d`, `num_telemetry_readings_prev_30d`, `num_maintenances_prev_30d`, `num_repairs_prev_30d`, `days_since_installation_at_obs`, `telemetry_frequency_prev_30d`; categorical: `machine_type`, `location`) and target `y` (`major_error_in_next_7_days`). Split into training and testing sets (e.g., 70/30 split) using `sklearn.model_selection.train_test_split` (set `random_state=42`, `stratify` on `y` for class balance).

4. **Data Visualization**: Create two separate plots to visually inspect relationships with `major_error_in_next_7_days`:
    *   A violin plot (or box plot) showing the distribution of `max_vibration_prev_30d` for non-error (0) vs. major-error (1) instances. Ensure appropriate labels and titles.
    *   A stacked bar chart showing the proportion of `major_error_in_next_7_days` (0 or 1) across different `machine_type` values. Ensure appropriate labels and titles.

5. **ML Pipeline & Evaluation (Binary Classification)**: 
    *   Create an `sklearn.pipeline.Pipeline` with a `sklearn.compose.ColumnTransformer` for preprocessing:
        *   For numerical features: Apply `sklearn.preprocessing.SimpleImputer(strategy='mean')` followed by `sklearn.preprocessing.StandardScaler`.
        *   For categorical features: Apply `sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')`.
    *   The final estimator in the pipeline should be `sklearn.ensemble.HistGradientBoostingClassifier` (set `random_state=42`).
    *   Train the pipeline on `X_train`, `y_train`. Predict probabilities for the positive class (class 1) on the test set (`X_test`).
    *   Calculate and print the `sklearn.metrics.roc_auc_score` and a `sklearn.metrics.classification_report` for the test set predictions.

## Focus
Predicting future major equipment errors based on time-windowed sensor telemetry and maintenance data, using SQL for advanced feature aggregation and an end-to-end ML pipeline for binary classification.

## Dataset
Synthetic `machines` metadata, `telemetry` (sensor readings), and `maintenance` logs.

## Hint
When generating synthetic data, ensure a subset of machines have a `major_error_date` after their `installation_date`, with corresponding spikes in `temperature` and `vibration` readings in the days leading up to the error. For the SQL query, use CTEs to first find the `latest_telemetry_date` for each machine, then use this date to define a 30-day look-back window for aggregating features. In Pandas, calculate the target by checking if any simulated `major_error_date` falls within 7 days *after* each machine's `current_observation_date` (derived from SQL).
