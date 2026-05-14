# AI Daily Lab — 2026-05-14

## Task
Develop a machine learning pipeline to predict if a restaurant reservation will be a 'no-show' (binary classification) for upcoming bookings, based on reservation details and the customer's historical activity up to a fixed global prediction cutoff date.

## Focus
Predicting Reservation No-Shows, Time-Series Feature Engineering with SQL, Binary Classification, Data Leakage Prevention.

## Dataset
1.  **Synthetic Data Generation (Pandas/Numpy)**:
    *   `bookings_df`: 5000-8000 rows. Columns: `booking_id` (unique), `customer_id` (random, some repeat customers), `reservation_datetime` (random datetimes over the last 2 years, up to `pd.Timestamp.now()`), `num_guests` (random integers 1-8), `booking_channel` (e.g., 'Online', 'Phone', 'Walk-in'), `special_requests_flag` (boolean, e.g., 20% True), `customer_segment` (e.g., 'Bronze', 'Silver', 'Gold', 'Platinum'), `is_no_show` (binary, ~15-20% True).
    *   `customer_activity_df`: 15000-25000 rows. Columns: `activity_id` (unique), `customer_id` (sampled from `bookings_df`), `activity_datetime` (random datetimes, always *before* their associated reservation or general customer activity history). `activity_type` (e.g., 'reservation_made', 'reservation_cancelled', 'website_visit', 'loyalty_points_redeemed', 'takeout_order').
    *   **Simulate realistic patterns**: 'Platinum' customers have lower `is_no_show` rates. 'Online' bookings might have slightly higher no-show rates. `reservation_cancelled` activity is likely for those who don't no-show. Customers with more `loyalty_points_redeemed` activity are less likely to no-show. Ensure `activity_datetime` is always before `reservation_datetime` for customer-specific historical activities.
    *   Sort `customer_activity_df` by `customer_id` then `activity_datetime`.

2.  **Load into SQLite & SQL Feature Engineering (Historical Customer Behavior)**:
    *   Create an in-memory SQLite database using `sqlite3`. Load `bookings_df` and `customer_activity_df` into tables named `bookings` and `customer_activities` respectively.
    *   Define `GLOBAL_PREDICTION_CUTOFF_DATE` (e.g., `bookings_df['reservation_datetime'].max() - pd.Timedelta(weeks=4)`). This represents the 'present moment' from which we are making predictions.
    *   Write a single SQL query that performs the following for *each booking* where its `reservation_datetime` is *after* `GLOBAL_PREDICTION_CUTOFF_DATE` (these are our target bookings):
        *   `current_cutoff_date` (the `GLOBAL_PREDICTION_CUTOFF_DATE` itself, for consistency).
        *   `num_prev_bookings_customer_12m` (count of `reservation_made` activities by `customer_id` in the 12 months *before or on* `GLOBAL_PREDICTION_CUTOFF_DATE`).
        *   `num_prev_cancellations_customer_12m` (count of `reservation_cancelled` activities by `customer_id` in the 12 months *before or on* `GLOBAL_PREDICTION_CUTOFF_DATE`).
        *   `days_since_last_activity_at_cutoff`: Number of days between `GLOBAL_PREDICTION_CUTOFF_DATE` and the customer's most recent `activity_datetime` (any type) *before or on* `GLOBAL_PREDICTION_CUTOFF_DATE`. Return 9999 if no prior activity.
        *   `num_loyalty_redeemed_customer_12m` (count of `loyalty_points_redeemed` activities by `customer_id` in the 12 months *before or on* `GLOBAL_PREDICTION_CUTOFF_DATE`).
        *   **Includes static booking attributes**: `booking_id`, `customer_id`, `reservation_datetime`, `num_guests`, `booking_channel`, `special_requests_flag`, `customer_segment`, `is_no_show`.
        *   **Ensures** all relevant future bookings are included (using `LEFT JOIN`), showing 0 for counts/sums and 9999 for `days_since_last_activity` if no historical activity.
    
3.  **Pandas Feature Engineering & Binary Target Creation**:
    *   Fetch the SQL query results into a pandas DataFrame (`booking_features_df`).
    *   Convert `reservation_datetime` and `current_cutoff_date` to datetime objects.
    *   Handle `NaN` values: Fill numerical aggregated features (e.g., `num_prev_bookings_customer_12m`) with 0 or 0.0. Fill `days_since_last_activity_at_cutoff` with 9999.
    *   Calculate `time_until_reservation_days_at_cutoff`: Number of days between `current_cutoff_date` and `reservation_datetime`. (Should be positive for all records).
    *   Extract `hour_of_day`, `day_of_week`, and `month_of_year` from `reservation_datetime`.
    *   Define features `X` (numerical: `num_guests`, `num_prev_bookings_customer_12m`, `num_prev_cancellations_customer_12m`, `days_since_last_activity_at_cutoff`, `num_loyalty_redeemed_customer_12m`, `time_until_reservation_days_at_cutoff`, `hour_of_day`, `day_of_week`, `month_of_year`; categorical: `booking_channel`, `special_requests_flag`, `customer_segment`) and target `y` (`is_no_show`).
    *   Split into training and testing sets (e.g., 70/30 split) using `sklearn.model_selection.train_test_split` (`random_state=42`, `stratify` on `y` for class balance).

4.  **Data Visualization (Matplotlib/Seaborn)**:
    *   Create a stacked bar chart showing the proportion of `is_no_show` (0 or 1) across different `booking_channel` values. Ensure appropriate labels and titles.
    *   Create a violin plot (or box plot) showing the distribution of `num_guests` for 'no-show' (1) vs. 'show' (0) bookings. Ensure appropriate labels and titles.

5.  **ML Pipeline & Evaluation (Binary Classification)**:
    *   Create an `sklearn.pipeline.Pipeline` with a `sklearn.compose.ColumnTransformer` for preprocessing:
        *   For numerical features: Apply `sklearn.preprocessing.SimpleImputer(strategy='mean')` followed by `sklearn.preprocessing.StandardScaler`.
        *   For categorical features: Apply `sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')`.
    *   The final estimator in the pipeline should be `sklearn.ensemble.HistGradientBoostingClassifier` (set `random_state=42`).
    *   Train the pipeline on `X_train`, `y_train`. Predict probabilities for the positive class (class 1: no-show) on the test set (`X_test`).
    *   Calculate and print the `sklearn.metrics.roc_auc_score` and a `sklearn.metrics.classification_report` for the test set predictions.

## Hint
When performing SQL aggregations, use `julianday()` for date comparisons to filter activities correctly based on `GLOBAL_PREDICTION_CUTOFF_DATE`. Remember to cast boolean `special_requests_flag` to a string or integer if `OneHotEncoder` needs it. Focus on ensuring no future information 'leaks' into your features based on the `GLOBAL_PREDICTION_CUTOFF_DATE`.
