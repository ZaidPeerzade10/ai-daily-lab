# AI Daily Lab — 2026-05-10

## Task
Develop a machine learning pipeline to predict the **delay category** of a flight (multi-class: 'On Time', 'Slight Delay', 'Significant Delay'), based on its schedule, origin/destination, airline, and simulated weather conditions.

1.  **Generate Synthetic Data (Pandas/Numpy)**: Create two pandas DataFrames:
    *   `flights_df`: With 10000-15000 rows. Columns: `flight_id` (unique integers), `airline` (e.g., 'AA', 'DL', 'UA', 'WN', 'AS'), `origin_airport` (e.g., 'JFK', 'LAX', 'ORD', 'DFW'), `destination_airport` (same set as origin), `scheduled_departure` (random datetimes over the last year), `scheduled_duration_minutes` (random integers 60-360), `actual_delay_minutes` (target variable: random integers, can be negative for early, 0-30 for minor, 30-180 for significant).
    *   `airport_weather_df`: With 2000-3000 rows. Columns: `airport_code` (e.g., 'JFK', 'LAX'), `weather_date` (random dates over the last year, daily granularity per airport), `weather_condition` (e.g., 'Clear', 'Rain', 'Snow', 'Fog', 'Thunderstorm').
    *   **Simulate realistic patterns**: Ensure `destination_airport` is different from `origin_airport`. For 10-15% of flights, introduce a 'Significant Delay' by increasing `actual_delay_minutes`. Delays should be more common for 'Fog' or 'Snow' `weather_condition`, especially for certain `origin_airport`s (e.g., ORD in winter). 'TypeA' airlines might have slightly better on-time performance. `actual_delay_minutes` can be negative for early arrivals.
    *   Sort `flights_df` by `scheduled_departure`.

2.  **Load into SQLite & SQL Feature Engineering (Time-Windowed Aggregations)**: Create an in-memory SQLite database using `sqlite3`. Load `flights_df` and `airport_weather_df` into tables named `flights` and `airport_weather` respectively.
    *   Write a single SQL query that performs the following for *each flight*:
        *   Joins `flights` with `airport_weather` on `origin_airport` and `scheduled_departure` date to get `departure_weather_condition`.
        *   **Aggregates historical features based on activities *within the 30 days preceding each flight's `scheduled_departure` date***:
            *   `avg_airline_delay_prev_30d` (average `actual_delay_minutes` for the flight's `airline`).
            *   `num_flights_origin_prev_30d` (count of flights from the `origin_airport`).
            *   `avg_origin_delay_prev_30d` (average `actual_delay_minutes` for the `origin_airport`).
        *   **Extracts time-based features**: `day_of_week`, `hour_of_day`, `month_of_year` from `scheduled_departure`.
        *   **Includes static flight attributes**: `flight_id`, `airline`, `origin_airport`, `destination_airport`, `scheduled_departure`, `scheduled_duration_minutes`, `actual_delay_minutes`.
        *   **Ensures** all flights are included (using `LEFT JOIN`), showing 0 for counts/sums and 0.0 for averages if no historical activity. Handle `NULL` `departure_weather_condition` with 'Unknown'.
    *   The query should return all mentioned fields.
    *   **Hint**: Use CTEs to calculate historical aggregates per airline/airport. Use `julianday()` for date comparisons. Aggregate features using `COALESCE` to handle `NULL`s from `LEFT JOIN`s.

3.  **Pandas Feature Engineering & Multi-class Target Creation**: Fetch the SQL query results into a pandas DataFrame (`flight_features_df`).
    *   Convert `scheduled_departure` to datetime objects.
    *   Handle `NaN` values: Fill numerical aggregated features (`avg_airline_delay_prev_30d`, etc.) with 0 or 0.0 as appropriate.
    *   **Create the Multi-class Target `delay_category`**: Based on `actual_delay_minutes`:
        *   'On Time': `actual_delay_minutes` <= 15
        *   'Slight Delay': 15 < `actual_delay_minutes` <= 60
        *   'Significant Delay': `actual_delay_minutes` > 60
    *   Define features `X` (numerical: `scheduled_duration_minutes`, `avg_airline_delay_prev_30d`, `num_flights_origin_prev_30d`, `avg_origin_delay_prev_30d`, `day_of_week`, `hour_of_day`, `month_of_year`; categorical: `airline`, `origin_airport`, `destination_airport`, `departure_weather_condition`) and target `y` (`delay_category`). Split into training and testing sets (e.g., 70/30 split) using `sklearn.model_selection.train_test_split` (set `random_state=42`, `stratify` on `y` for class balance).

4.  **Data Visualization**: Create two separate plots to visually inspect relationships with `delay_category`:
    *   A violin plot (or box plot) showing the distribution of `scheduled_duration_minutes` for each `delay_category`. Ensure appropriate labels and titles.
    *   A stacked bar chart showing the proportion of `delay_category` across different `departure_weather_condition` values. Ensure appropriate labels and titles.

5.  **ML Pipeline & Evaluation (Multi-class Classification)**: 
    *   Create an `sklearn.pipeline.Pipeline` with a `sklearn.compose.ColumnTransformer` for preprocessing:
        *   For numerical features: Apply `sklearn.preprocessing.SimpleImputer(strategy='mean')` followed by `sklearn.preprocessing.StandardScaler`.
        *   For categorical features: Apply `sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')`.
    *   The final estimator in the pipeline should be `sklearn.ensemble.HistGradientBoostingClassifier` (set `random_state=42`).
    *   Train the pipeline on `X_train`, `y_train`. Predict the `delay_category` on the test set (`X_test`).
    *   Calculate and print a `sklearn.metrics.classification_report` for the test set predictions.

## Focus
Multi-class classification, time-series feature engineering, SQL aggregations, handling datetime and categorical features.

## Dataset
Synthetic flight schedules, actual delays, airport weather conditions, and historical flight performance metrics.

## Hint
When generating synthetic data, consider how specific weather conditions or peak hours at certain airports might realistically impact `actual_delay_minutes`. For the multi-class target, `pd.cut()` or simple conditional logic can effectively categorize `actual_delay_minutes`. In SQL, remember to join `airport_weather` using `DATE()` on both sides to match daily weather conditions with scheduled departure dates.
