# AI Daily Lab — 2026-03-09

## Task
Develop a machine learning pipeline to predict the type of future machine failure based on recent sensor readings and machine attributes.

## Focus
Predictive Maintenance, Multi-Class Classification, Time-Series Feature Engineering (Aggregations over windows), SQL Analytics, ML Pipelines.

## Dataset
1. **Generate Synthetic Data (Pandas/Numpy)**: Create three pandas DataFrames:
    *   `machines_df`: With 100-150 rows. Columns: `machine_id` (unique integers), `installation_date` (random dates over the last 5 years), `machine_type` (e.g., 'TypeA', 'TypeB', 'TypeC'), `location` (e.g., 'Factory1', 'Factory2', 'Factory3'), `maintenance_frequency_days` (random integers 30-365).
    *   `sensor_readings_df`: With 10000-15000 rows. Columns: `reading_id` (unique integers), `machine_id` (randomly sampled from `machines_df` IDs), `timestamp` (random timestamps occurring *after* their respective `installation_date`), `temperature` (random floats 20-100), `vibration` (random floats 0.1-5.0), `pressure` (random floats 50-200).
    *   `failure_events_df`: With 200-300 rows. Columns: `failure_id` (unique integers), `machine_id` (randomly sampled from `machines_df` IDs), `failure_date` (random dates *after* `installation_date`), `failure_type` (e.g., 'Overheating', 'Mechanical_Wear', 'Pressure_Leak', 'Electrical_Fault').
    *   **Simulate realistic patterns**: Ensure `timestamp` and `failure_date` are always after `installation_date`. Bias `sensor_readings_df` such that for machines that eventually fail, their `temperature`, `vibration`, or `pressure` readings show an increasing trend (or unusual fluctuations) in the *days/hours leading up to their `failure_date`*. Different `machine_type`s might be more prone to specific `failure_type`s. Some machines might never fail. Sort `sensor_readings_df` by `machine_id` then `timestamp`.

2. **Load into SQLite & SQL Feature Engineering (Machine Health Snapshot)**: Create an in-memory SQLite database using `sqlite3`. Load `machines_df` into a table named `machines`, `sensor_readings_df` into `sensor_readings`, and `failure_events_df` into `failure_events`. Determine a `global_analysis_date` (e.g., `max(timestamp)` from `sensor_readings_df` + 30 days, using pandas) and a `feature_cutoff_date` (`global_analysis_date` - 90 days).
    Write a single SQL query that performs the following for *each machine*, aggregating its sensor readings *in the 30 days prior to `feature_cutoff_date`*:
    *   **Joins** `machines` with aggregated subqueries for `sensor_readings`.
    *   **Aggregates features based on readings *before* `feature_cutoff_date` (specifically in the 30 days preceding it)**:
        *   `avg_temp_30d_pre_cutoff`: Average `temperature` in the 30 days prior to `feature_cutoff_date`.
        *   `max_vibration_30d_pre_cutoff`: Maximum `vibration` in the 30 days prior to `feature_cutoff_date`.
        *   `std_pressure_30d_pre_cutoff`: Standard deviation of `pressure` in the 30 days prior to `feature_cutoff_date`.
        *   `num_readings_30d_pre_cutoff`: Count of `reading_id`s in the 30 days prior to `feature_cutoff_date`.
        *   `days_since_last_reading_pre_cutoff`: Number of days between `feature_cutoff_date` and `MAX(timestamp)` for the machine (only considering readings before `feature_cutoff_date`).
    *   **Includes static machine attributes**: `machine_id`, `installation_date`, `machine_type`, `location`, `maintenance_frequency_days`.
    *   **Ensures** all machines are included (using `LEFT JOIN`), showing 0 for counts/sums, 0.0 for averages/std dev, and `NULL` for `days_since_last_reading_pre_cutoff` if no readings in the 30-day window.
    *   The query should return `machine_id`, `installation_date`, `machine_type`, `location`, `maintenance_frequency_days`, and all the aggregated features.

3. **Pandas Feature Engineering & Multi-Class Target Creation (Future Failure Type)**: Fetch the SQL query results into a pandas DataFrame (`machine_features_df`).
    *   Handle `NaN` values: Fill `num_readings_30d_pre_cutoff` with 0. Fill `avg_temp_30d_pre_cutoff`, `max_vibration_30d_pre_cutoff`, `std_pressure_30d_pre_cutoff` with 0.0. For `days_since_last_reading_pre_cutoff`, fill with a large sentinel value (e.g., 9999 days).
    *   Convert `installation_date` to datetime objects. Calculate `machine_age_at_cutoff_days`: The number of days between `installation_date` and the `feature_cutoff_date`.
    *   Calculate `reading_frequency_30d_pre_cutoff`: `num_readings_30d_pre_cutoff` / 30.0 (fill any `NaN` or `inf` with 0).
    *   **Create the Multi-Class Target `future_failure_type`**: For each machine, check `failure_events_df` for any `failure_date` that occurs *between `feature_cutoff_date` and `feature_cutoff_date + timedelta(days=60)`*. Merge this `failure_type` (take the *first* failure type if multiple occur in the window) with `machine_features_df` using a left merge. If no failure occurs in this window, assign the class 'No_Failure'.
    *   Define features `X` (all numerical: `maintenance_frequency_days`, `machine_age_at_cutoff_days`, `avg_temp_30d_pre_cutoff`, `max_vibration_30d_pre_cutoff`, `std_pressure_30d_pre_cutoff`, `num_readings_30d_pre_cutoff`, `days_since_last_reading_pre_cutoff`, `reading_frequency_30d_pre_cutoff`; categorical: `machine_type`, `location`) and target `y` (`future_failure_type`). Split into training and testing sets (e.g., 70/30 split) using `sklearn.model_selection.train_test_split` (set `random_state=42`, `stratify` on `y` for class balance).

4. **Data Visualization**: Create two separate plots to visually inspect relationships with `future_failure_type`:
    *   A violin plot (or box plot) showing the distribution of `max_vibration_30d_pre_cutoff` for each `future_failure_type`.
    *   A stacked bar chart showing the proportion of `future_failure_type` across different `machine_type` values.
    Ensure plots have appropriate labels and titles.

5. **ML Pipeline & Evaluation (Multi-Class)**: 
    *   Create an `sklearn.pipeline.Pipeline` with a `sklearn.compose.ColumnTransformer` for preprocessing:
        *   For numerical features: Apply `sklearn.preprocessing.SimpleImputer(strategy='mean')` followed by `sklearn.preprocessing.StandardScaler`.
        *   For categorical features: Apply `sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')`.
    *   The final estimator in the pipeline should be `sklearn.ensemble.HistGradientBoostingClassifier` (set `random_state=42`).
    *   Train the pipeline on `X_train`, `y_train`. Predict `future_failure_type` for `X_test`.
    *   Calculate and print the `sklearn.metrics.accuracy_score` and a `sklearn.metrics.classification_report` for the test set predictions.

## Hint
For SQL step 2, remember to use `LEFT JOIN`s from the `machines` table to aggregated subqueries. For pandas target creation in step 3, filter `failure_events_df` for the target window, group by `machine_id`, take the first `failure_type`, then left merge with your main features DataFrame, filling `NaN` in the merged column with 'No_Failure'.
