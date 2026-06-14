# AI Daily Lab — 2026-06-14

## Task
Develop a machine learning pipeline to predict the **delivery delay minutes** for an individual package, based on package attributes, courier details, real-time weather conditions, and historical delivery performance up to a specific cutoff date.

1.  **Generate Synthetic Data (Pandas/Numpy)**: Create three pandas DataFrames:
    *   `packages_df`: With 10000-15000 rows. Columns: `package_id` (unique integers), `courier_id` (randomly sampled from `couriers_df` IDs), `origin_city` (e.g., 'CityA', 'CityB', 'CityC'), `destination_city` (e.g., 'CityA', 'CityB', 'CityC'), `package_weight_kg` (random floats 0.1-50.0), `package_type` (e.g., 'Standard', 'Express', 'Fragile'), `distance_km` (random integers 10-1000), `scheduled_delivery_datetime` (random datetimes over the last 2 years), `_actual_delivery_datetime` (for target calculation, simulate actual delivery times).
    *   `couriers_df`: With 100-200 rows. Columns: `courier_id` (unique integers), `courier_rating` (random floats 1.0-5.0), `vehicle_type` (e.g., 'Motorbike', 'Van', 'Truck'), `avg_speed_kmph` (random floats 20.0-60.0, adjust slightly by `vehicle_type`).
    *   `weather_conditions_df`: With 2000-3000 rows. Columns: `city` (from `origin_city`/`destination_city` list), `weather_date` (random dates covering package range), `temperature_celsius` (random floats -5.0-35.0), `precipitation_mm` (random floats 0.0-50.0), `is_stormy` (binary: 1 for high precipitation/low temp, 0 otherwise).
    *   **Simulate realistic patterns**: Ensure `_actual_delivery_datetime` is generally `scheduled_delivery_datetime` + `expected_travel_time` + `delay`. `expected_travel_time` should be `distance_km` / `avg_speed_kmph`. `delay` should be positively correlated with `distance_km`, `package_weight_kg` (less for 'Express'), and `is_stormy`; negatively correlated with `courier_rating`. Introduce some packages delivered early (negative delay, which will be floored to 0 in target). Ensure `_actual_delivery_datetime` is generally after `scheduled_delivery_datetime`. Sort `packages_df` by `scheduled_delivery_datetime`.

2.  **Load into SQLite & SQL Feature Engineering (Time-Windowed Aggregations)**: Create an in-memory SQLite database using `sqlite3`. Load `packages_df`, `couriers_df`, and `weather_conditions_df` into tables named `packages`, `couriers`, and `weather` respectively.
    *   Define `GLOBAL_PREDICTION_CUTOFF_DATE` as 7 days prior to the latest `scheduled_delivery_datetime` in your generated `packages_df`.
    *   Write a single SQL query that performs the following for *each package scheduled AFTER `GLOBAL_PREDICTION_CUTOFF_DATE`*:
        *   Joins the `packages` (filtered for events after cutoff) with the `couriers` table.
        *   Joins with `weather` table on `destination_city` and `CAST(scheduled_delivery_datetime AS DATE)` matching `weather_date`. (Use `CAST(strftime('%Y-%m-%d %H:%M:%S', scheduled_delivery_datetime) AS DATE)` for date part, `strftime('%Y-%m-%d', weather_date)` for weather date).
        *   Aggregates historical features based on *actual historical deliveries up to and including `GLOBAL_PREDICTION_CUTOFF_DATE`*:
            *   `avg_actual_delay_dest_city_prev_30d`: Average `(julianday(p._actual_delivery_datetime) - julianday(p.scheduled_delivery_datetime)) * 24 * 60` (floored at 0 for negative values) for the `destination_city` in the 30 days preceding `GLOBAL_PREDICTION_CUTOFF_DATE`.
            *   `num_deliveries_dest_city_prev_30d`: Count of deliveries for the `destination_city` in prior 30 days.
            *   `avg_actual_delay_courier_prev_30d`: Average `delay_minutes` for the `courier_id` in prior 30 days.
            *   `num_deliveries_courier_prev_30d`: Count of deliveries for the `courier_id` in prior 30 days.
        *   Extracts time-based features from the `scheduled_delivery_datetime` of the *current* package: `scheduled_day_of_week` (0-6), `scheduled_hour_of_day` (0-23), `scheduled_month_of_year` (1-12).
        *   Includes static product and listing attributes: `package_id`, `courier_id`, `origin_city`, `destination_city`, `package_weight_kg`, `package_type`, `distance_km`, `scheduled_delivery_datetime`, `_actual_delivery_datetime` (the actual values for *this specific future package* for target calculation), `courier_rating`, `vehicle_type`, `avg_speed_kmph`, `temperature_celsius`, `precipitation_mm`, `is_stormy`.
    *   **Ensures** all packages *after* the cutoff are included. Handle `NULL`s for historical aggregates (e.g., 0.0 for averages, 0 for counts if no prior activity).
    *   The query should return all mentioned fields.
    *   **Hint**: Use CTEs for pre-calculating historical aggregates up to the `GLOBAL_PREDICTION_CUTOFF_DATE`, then `LEFT JOIN` these with the future packages. Use `julianday()` for date arithmetic and `COALESCE` for NULL handling.

3.  **Pandas Feature Engineering & Regression Target Creation**: Fetch the SQL query results into a pandas DataFrame (`delivery_features_df`).
    *   Convert all relevant date/datetime columns (`scheduled_delivery_datetime`, `_actual_delivery_datetime`) to appropriate types.
    *   Handle `NaN` values: Fill numerical historical aggregated features (e.g., averages, counts) with 0.0 or 0 as appropriate. Fill `temperature_celsius` with its mean, `precipitation_mm` with 0.0, and `is_stormy` with 0 (assuming no weather data means mild weather).
    *   Calculate `scheduled_expected_duration_hours`: `distance_km` / (`avg_speed_kmph` + 1e-6). Fill any `NaN` or `inf` with 0.
    *   **Create the Regression Target `delivery_delay_minutes`**: Calculate `(df['_actual_delivery_datetime'] - df['scheduled_delivery_datetime']).dt.total_seconds() / 60`. Floor negative values to 0 (meaning on-time or early deliveries have 0 delay).
    *   Define features `X` (numerical: `package_weight_kg`, `distance_km`, `courier_rating`, `avg_speed_kmph`, `temperature_celsius`, `precipitation_mm`, `is_stormy`, `avg_actual_delay_dest_city_prev_30d`, `num_deliveries_dest_city_prev_30d`, `avg_actual_delay_courier_prev_30d`, `num_deliveries_courier_prev_30d`, `scheduled_expected_duration_hours`, `scheduled_day_of_week`, `scheduled_hour_of_day`, `scheduled_month_of_year`; categorical: `origin_city`, `destination_city`, `package_type`, `vehicle_type`) and target `y` (`delivery_delay_minutes`).
    *   Split into training and testing sets (e.g., 70/30 split) using `sklearn.model_selection.train_test_split` (set `random_state=42`).

4.  **Data Visualization (Matplotlib/Seaborn)**: Create two separate plots to visually inspect relationships with `delivery_delay_minutes`:
    *   A violin plot (or box plot) showing the distribution of `delivery_delay_minutes` for each `package_type`. Ensure appropriate labels and titles.
    *   A scatter plot showing `delivery_delay_minutes` vs `distance_km`. Ensure appropriate labels and titles.

5.  **ML Pipeline & Evaluation (Regression)**:
    *   Create an `sklearn.pipeline.Pipeline` with a `sklearn.compose.ColumnTransformer` for preprocessing:
        *   For numerical features: Apply `sklearn.preprocessing.SimpleImputer(strategy='mean')` followed by `sklearn.preprocessing.StandardScaler`.
        *   For categorical features: Apply `sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')`.
    *   The final estimator in the pipeline should be `sklearn.ensemble.HistGradientBoostingRegressor` (set `random_state=42`).
    *   Train the pipeline on `X_train`, `y_train`. Predict the `delivery_delay_minutes` on the test set (`X_test`).
    *   Calculate and print `sklearn.metrics.mean_absolute_error` and `sklearn.metrics.r2_score` for the test set predictions.

## Focus
Regression for Delivery Time Prediction using time-windowed historical aggregates and external features.

## Dataset
Logistics Delivery Data (Packages, Couriers, Weather)

## Hint
When calculating historical averages for delay in SQL, remember to convert datetimes to a numerical representation (like `julianday()`) to perform arithmetic, and multiply by `24 * 60` to get minutes. Also, ensure you handle potential `NULL`s from `LEFT JOIN` operations and floor negative delays to zero for a consistent target variable definition.
