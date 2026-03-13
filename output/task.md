# AI Daily Lab — 2026-03-13

## Task
Develop a machine learning pipeline to predict restaurant reservation no-shows, leveraging customer history and reservation details.

## Focus
SQL-based sequential feature engineering, binary classification, class imbalance handling.

## Dataset
1. **Generate Synthetic Data (Pandas/Numpy)**: Create two pandas DataFrames:
    *   `customers_df`: With 500-700 rows. Columns: `customer_id` (unique integers), `signup_date` (random dates over the last 5 years), `loyalty_tier` (e.g., 'Bronze', 'Silver', 'Gold'), `average_party_size_preference` (random floats 1.5-6.0).
    *   `reservations_df`: With 5000-8000 rows. Columns: `reservation_id` (unique integers), `customer_id` (randomly sampled from `customers_df` IDs), `booking_date_time` (random datetime over the last 2 years), `reservation_date_time` (random datetime *after* `booking_date_time` but within 1 hour to 30 days of it), `party_size` (random integers 1-10), `cuisine_type` (e.g., 'Italian', 'Mexican', 'Asian', 'American', 'French'). Generate a hidden `is_no_show` (binary, 0 or 1) *for each reservation*, with an approximate 10-15% no-show rate overall.
    *   **Simulate realistic no-show patterns**: Ensure `booking_date_time` is after `signup_date`, and `reservation_date_time` is after `booking_date_time`. Bias `is_no_show` such that:
        *   Customers with `loyalty_tier='Bronze'` have a significantly higher no-show rate.
        *   `party_size` greater than 6 might correlate with higher no-show rates.
        *   Reservations booked very far in advance (e.g., > 15 days lead time) or very last minute (< 6 hours lead time) have slightly different no-show propensities.
        *   Customers with a history of prior no-shows should have a higher propensity to no-show again.
    *   Sort `reservations_df` by `customer_id` then `booking_date_time` for easier sequential processing.

2. **Load into SQLite & SQL Feature Engineering (Customer's Prior Reservation Behavior)**: Create an in-memory SQLite database using `sqlite3`. Load `customers_df` into a table named `customers` and `reservations_df` into a table named `reservations`.
    Write a single SQL query that performs the following for *each reservation event* in `reservations`:
    *   **Joins** `customers` and `reservations` tables.
    *   **Calculates sequential features based on the customer's *prior reservations* (excluding the current one), relative to the current `booking_date_time`**:
        *   `customer_prior_total_reservations`: Count of all *previous* reservations for the same customer.
        *   `customer_prior_no_shows`: Count of *previous* reservations that were `is_no_show=1` for the same customer.
        *   `customer_prior_no_show_rate`: `customer_prior_no_shows` / `customer_prior_total_reservations` (0.0 if no prior reservations).
        *   `customer_avg_prior_party_size`: Average `party_size` of *previous* reservations for the same customer.
        *   `days_since_last_customer_reservation`: Number of days between the current `booking_date_time` and the customer's *most recent prior* `booking_date_time`. If it's the customer's first reservation, use the number of days between `signup_date` and the current `booking_date_time`.
    *   **Includes static customer and current reservation attributes**: `reservation_id`, `customer_id`, `booking_date_time`, `reservation_date_time`, `party_size`, `cuisine_type`, `is_no_show` (the target), `signup_date`, `loyalty_tier`, `average_party_size_preference`.
    *   The query should return all these attributes and engineered features. Missing values for prior aggregates/dates should be `NULL`.
    *   **Hint**: Use window functions with `LAG` and `SUM(CASE WHEN ... END)` over `PARTITION BY customer_id ORDER BY booking_date_time`. Use `julianday()` for date differences.

3. **Pandas Feature Engineering & Binary Target Creation**: Fetch the SQL query results into a pandas DataFrame (`reservation_features_df`).
    *   Handle `NaN` values: Fill `customer_prior_total_reservations`, `customer_prior_no_shows` with 0. Fill `customer_prior_no_show_rate`, `customer_avg_prior_party_size` with 0.0. For `days_since_last_customer_reservation` (for a customer's first reservation), fill with the `days_since_signup_at_booking` (calculated below).
    *   Convert `signup_date`, `booking_date_time`, `reservation_date_time` to datetime objects.
    *   Calculate `reservation_lead_time_hours`: Difference in hours between `reservation_date_time` and `booking_date_time`.
    *   Calculate `days_since_signup_at_booking`: Days between `signup_date` and `booking_date_time`.
    *   Calculate `party_size_deviation_from_avg_preference`: `party_size` - `average_party_size_preference`.
    *   Define features `X` (all numerical: `party_size`, `reservation_lead_time_hours`, `customer_prior_total_reservations`, `customer_prior_no_shows`, `customer_prior_no_show_rate`, `customer_avg_prior_party_size`, `days_since_last_customer_reservation`, `days_since_signup_at_booking`, `average_party_size_preference`, `party_size_deviation_from_avg_preference`; categorical: `loyalty_tier`, `cuisine_type`) and target `y` (`is_no_show`). Split into training and testing sets (e.g., 70/30 split) using `sklearn.model_selection.train_test_split` (set `random_state=42`, `stratify` on `y` to handle class imbalance).

4. **Data Visualization**: Create two separate plots to visually inspect relationships with `is_no_show`:
    *   A violin plot (or box plot) showing the distribution of `reservation_lead_time_hours` for no-shows (1) vs. shows (0). Ensure appropriate labels and titles.
    *   A stacked bar chart showing the proportion of `is_no_show` (0 or 1) across different `loyalty_tier` values. Ensure appropriate labels and titles.

5. **ML Pipeline & Evaluation (Binary Classification)**: 
    *   Create an `sklearn.pipeline.Pipeline` with a `sklearn.compose.ColumnTransformer` for preprocessing:
        *   For numerical features: Apply `sklearn.preprocessing.SimpleImputer(strategy='mean')` followed by `sklearn.preprocessing.StandardScaler`.
        *   For categorical features: Apply `sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')`.
    *   The final estimator in the pipeline should be `sklearn.ensemble.HistGradientBoostingClassifier` (set `random_state=42`).
    *   Train the pipeline on `X_train`, `y_train`. Predict probabilities for the positive class (class 1) on the test set (`X_test`).
    *   Calculate and print the `sklearn.metrics.roc_auc_score` and a `sklearn.metrics.classification_report` for the test set predictions.

## Hint
Pay close attention to handling `NULL` values and potential division by zero when calculating ratios and average lead times in both SQL and Pandas. The `stratify` argument in `train_test_split` is crucial for binary classification with imbalanced datasets like no-shows.
