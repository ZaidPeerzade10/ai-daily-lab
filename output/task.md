# AI Daily Lab — 2026-06-24

## Task
Develop a machine learning pipeline to predict if a piece of industrial equipment will **fail** within the next 30 days (binary classification), based on its static attributes, real-time sensor readings, and historical maintenance records up to a specific cutoff date.

1.  **Generate Synthetic Data (Pandas/Numpy)**: Create three pandas DataFrames:
    *   `machines_df`: With 1000-1500 rows. Columns: `machine_id` (unique integers), `installation_date` (random dates over the last 5 years), `machine_type` (e.g., 'TypeA', 'TypeB', 'TypeC'), `location` (e.g., 'PlantX', 'PlantY'), `_actual_failure_date` (for ~10-15% of machines, a random date *after* `installation_date` and within the general data range; `pd.NaT` for non-failed machines).
    *   `sensor_readings_df`: With 50000-70000 rows. Columns: `reading_id` (unique integers), `machine_id` (randomly sampled from `machines_df` IDs), `reading_datetime` (random datetimes, *after* respective `installation_date` for the `machine_id`, and *before* `_actual_failure_date` if applicable, up to `pd.Timestamp.now()`), `temperature_celsius` (random floats 20.0-100.0), `vibration_hz` (random floats 10.0-100.0), `pressure_psi` (random floats 50.0-500.0).
    *   `maintenance_df`: With 5000-8000 rows. Columns: `maintenance_id` (unique integers), `machine_id` (randomly sampled from `machines_df` IDs), `maintenance_date` (random dates, *after* respective `installation_date` for the `machine_id`, and *before* `_actual_failure_date` if applicable), `maintenance_type` (e.g., 'Routine', 'Repair', 'Inspection').
    *   **Simulate realistic patterns**: Ensure `reading_datetime` and `maintenance_date` are always after `installation_date` and before `_actual_failure_date` for failed machines. For machines with an `_actual_failure_date`, `temperature_celsius` and `vibration_hz` should show a noticeable increasing trend in the 30-60 days leading up to their `_actual_failure_date`. `machine_type` and `location` should influence baseline sensor readings and failure rates. `maintenance_type='Repair'` events should slightly reduce sensor values temporarily after the event for the respective machine. Sort `sensor_readings_df` and `maintenance_df` by `machine_id` then date.

2.  **Load into SQLite & SQL Feature Engineering (Time-Windowed Aggregations)**: Create an in-memory SQLite database using `sqlite3`. Load `machines_df`, `sensor_readings_df`, and `maintenance_df` into tables named `machines`, `sensor_readings`, and `maintenance` respectively.
    *   Define `GLOBAL_PREDICTION_CUTOFF_DATE` as 30 days prior to the latest `reading_datetime` in your generated `sensor_readings_df`.
    *   Write a single SQL query that performs the following for *each machine active up to `GLOBAL_PREDICTION_CUTOFF_DATE`*:
        *   Includes static attributes: `machine_id`, `installation_date`, `machine_type`, `location`, and the target-relevant `_actual_failure_date` from `machines` table.
        *   Aggregates historical features for the *respective `machine_id` in the 30 days preceding or on `GLOBAL_PREDICTION_CUTOFF_DATE`*:
            *   `avg_temperature_prev_30d`, `max_temperature_prev_30d` (from `temperature_celsius`).
            *   `avg_vibration_prev_30d`, `max_vibration_prev_30d` (from `vibration_hz`).
            *   `avg_pressure_prev_30d` (from `pressure_psi`).
            *   `num_maintenance_prev_30d`: Count of maintenance events.
            *   `days_since_last_maintenance_at_cutoff`: Number of days between `GLOBAL_PREDICTION_CUTOFF_DATE` and the most recent `maintenance_date` for this `machine_id` *before or on* the cutoff. Return 9999 if no prior maintenance.
    *   **Ensures** all machines are included (using `LEFT JOIN`), showing 0 for counts and 0.0 for averages if no activity in the window. Handle `NULL`s appropriately.
    *   The query should return all mentioned fields.

3.  **Pandas Feature Engineering & Binary Target Creation**: Fetch the SQL query results into a pandas DataFrame (`machine_features_df`).
    *   Convert `installation_date` and `_actual_failure_date` to datetime objects.
    *   Handle `NaN` values: Fill numerical historical aggregated features (e.g., averages, max, counts) with 0.0 or 0 as appropriate. Fill `days_since_last_maintenance_at_cutoff` with 9999.
    *   Calculate `machine_age_at_cutoff_days`: Number of days between `installation_date` and `GLOBAL_PREDICTION_CUTOFF_DATE`.
    *   **Create the Binary Target `will_fail_next_30d`**: For *each machine*, assign 1 if its `_actual_failure_date` falls *after* `GLOBAL_PREDICTION_CUTOFF_DATE` and *on or before* `GLOBAL_PREDICTION_CUTOFF_DATE + pd.Timedelta(days=30)`. Assign 0 otherwise.
    *   Define features `X` (numerical: `avg_temperature_prev_30d`, `max_temperature_prev_30d`, `avg_vibration_prev_30d`, `max_vibration_prev_30d`, `avg_pressure_prev_30d`, `num_maintenance_prev_30d`, `days_since_last_maintenance_at_cutoff`, `machine_age_at_cutoff_days`; categorical: `machine_type`, `location`) and target `y` (`will_fail_next_30d`).
    *   Split into training and testing sets (e.g., 70/30 split) using `sklearn.model_selection.train_test_split` (set `random_state=42`, `stratify` on `y` for class balance).

4.  **Data Visualization (Matplotlib/Seaborn)**: Create two separate plots to visually inspect relationships with `will_fail_next_30d`:
    *   A violin plot (or box plot) showing the distribution of `machine_age_at_cutoff_days` for 'Not Fail' (0) vs. 'Will Fail' (1) machines. Ensure appropriate labels and titles.
    *   A stacked bar chart showing the proportion of `will_fail_next_30d` (0 or 1) across different `machine_type` values. Ensure appropriate labels and titles.

5.  **ML Pipeline & Evaluation (Binary Classification)**:
    *   Create an `sklearn.pipeline.Pipeline` with a `sklearn.compose.ColumnTransformer` for preprocessing:
        *   For numerical features: Apply `sklearn.preprocessing.SimpleImputer(strategy='mean')` followed by `sklearn.preprocessing.StandardScaler`.
        *   For categorical features: Apply `sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')`.
    *   The final estimator in the pipeline should be `sklearn.ensemble.HistGradientBoostingClassifier` (set `random_state=42`, and consider `class_weight='balanced'` due to potential target imbalance).
    *   Train the pipeline on `X_train`, `y_train`. Predict probabilities for the positive class (class 1) on the test set (`X_test`).
    *   Calculate and print the `sklearn.metrics.roc_auc_score` and a `sklearn.metrics.classification_report` for the test set predictions.

## Focus
Time-series feature engineering from sensor and event data, SQL for historical aggregates, binary classification with imbalanced data.

## Dataset
Synthetic industrial equipment sensor readings and maintenance logs.

## Hint
For SQL feature engineering, create separate CTEs for historical sensor aggregates and maintenance event counts for each machine up to `GLOBAL_PREDICTION_CUTOFF_DATE`. Then, `LEFT JOIN` these CTEs with the main `machines` table to ensure all machines are included. Use `julianday()` for date arithmetic, `MAX(CASE WHEN ... END)` for `days_since_last_maintenance_at_cutoff` to find the latest date, and `COALESCE` for handling `NULL`s in aggregated features. When calculating `days_since_last_maintenance_at_cutoff`, if a machine has no prior maintenance, return 9999 (or a similar large number). For the binary target, ensure `GLOBAL_PREDICTION_CUTOFF_DATE` is consistently defined across SQL and Pandas. Use `class_weight='balanced'` in the classifier for the likely imbalanced failure prediction target.
