# AI Daily Lab — 2026-05-13

## Task
Develop a machine learning pipeline to predict if a loan will default within the next 90 days, based on applicant demographics, loan attributes, and historical payment behavior up to a specific cutoff date.

1.  **Synthetic Data Generation (Pandas/Numpy)**: Create three pandas DataFrames:
    *   `applicants_df`: With 1000-1500 rows. Columns: `applicant_id` (unique integers), `age` (random integers 20-70), `income` (random floats 2000-15000), `credit_score` (random integers 300-850), `employment_status` (e.g., 'Employed', 'Self-Employed', 'Unemployed', 'Retired').
    *   `loans_df`: With 1000-1500 rows. Columns: `loan_id` (unique integers), `applicant_id` (randomly sampled from `applicants_df` IDs), `loan_amount` (random floats 1000-50000), `loan_term_months` (random integers 12-60), `interest_rate` (random floats 0.05-0.20), `loan_date` (random dates over the last 5 years), `default_date` (random dates, for ~10-15% of loans, occurring *after* `loan_date` and within the last 2 years, `NaT` otherwise).
    *   `payments_df`: With 15000-25000 rows. Columns: `payment_id` (unique integers), `loan_id` (randomly sampled from `loans_df` IDs), `payment_date` (random dates *after* respective `loan_date` and before `default_date` if applicable), `paid_amount` (random floats 100-2000, generally proportional to `loan_amount` and `loan_term_months`).
    *   **Simulate realistic patterns**: Ensure `payment_date` is always after `loan_date` and before `default_date`. For defaulted loans, simulate fewer or smaller `paid_amount`s, especially in the 3-6 months leading to `default_date`. Lower `credit_score` and `income` should correlate with higher `interest_rate` and `default_date`.
    *   Sort `payments_df` by `loan_id` then `payment_date`.

2.  **Load into SQLite & SQL Feature Engineering (Payment History Aggregations)**: Create an in-memory SQLite database using `sqlite3`. Load `applicants_df`, `loans_df`, and `payments_df` into tables named `applicants`, `loans`, and `payments` respectively.
    *   Define `GLOBAL_PREDICTION_CUTOFF_DATE` as 3 months prior to the latest `payment_date` in your generated `payments_df` (e.g., `payments_df['payment_date'].max() - pd.Timedelta(months=3)`).
    *   Write a single SQL query that performs the following for *each loan*, aggregating its payment behavior *within the 6 months immediately preceding `GLOBAL_PREDICTION_CUTOFF_DATE`*:
        *   `current_cutoff_date` (the `GLOBAL_PREDICTION_CUTOFF_DATE` itself, for consistency).
        *   `num_payments_prev_6m` (count of payments).
        *   `total_paid_prev_6m` (sum of `paid_amount`).
        *   `avg_payment_prev_6m` (average of `paid_amount`).
        *   `days_since_last_payment_at_cutoff`: Number of days between `GLOBAL_PREDICTION_CUTOFF_DATE` and the most recent `payment_date` *before or on* the cutoff. Return a large number (e.g., 9999) if no payments before cutoff.
        *   `outstanding_balance_at_cutoff`: `loan_amount` - (sum of `paid_amount` for all payments *before or on* `GLOBAL_PREDICTION_CUTOFF_DATE`). Return `loan_amount` if no payments.
    *   **Includes static applicant and loan attributes**: `loan_id`, `applicant_id`, `loan_amount`, `loan_term_months`, `interest_rate`, `loan_date`, `age`, `income`, `credit_score`, `employment_status`.
    *   **Ensures** all loans are included (using `LEFT JOIN`), showing 0 for counts/sums/averages if no transactions in the 6-month window. Handle `NULL`s appropriately.
    *   The query should return all mentioned fields.

3.  **Pandas Feature Engineering & Binary Target Creation**: Fetch the SQL query results into a pandas DataFrame (`loan_features_df`).
    *   Convert all relevant date columns (`loan_date`, `current_cutoff_date`) to datetime objects.
    *   Handle `NaN` values: Fill numerical aggregated features with 0 or 0.0 as appropriate. Fill `days_since_last_payment_at_cutoff` with 9999.
    *   Calculate `loan_age_at_cutoff_days`: Number of days between `loan_date` and `current_cutoff_date`.
    *   Calculate `payment_frequency_prev_6m`: `num_payments_prev_6m` / 180.0. Fill any `NaN` or `inf` with 0.
    *   **Create the Binary Target `will_default_in_next_90_days`**: For *each loan*, determine if its simulated `default_date` (from the original `loans_df` merged back) falls within the 90-day period *immediately following* its `current_cutoff_date`. Merge this aggregate (1 if yes, 0 if no) with `loan_features_df`, filling `NaN`s with 0 for loans that did not default or whose default date falls outside the window.
    *   Define features `X` (numerical: `loan_amount`, `loan_term_months`, `interest_rate`, `age`, `income`, `credit_score`, `num_payments_prev_6m`, `total_paid_prev_6m`, `avg_payment_prev_6m`, `days_since_last_payment_at_cutoff`, `outstanding_balance_at_cutoff`, `loan_age_at_cutoff_days`, `payment_frequency_prev_6m`; categorical: `employment_status`) and target `y` (`will_default_in_next_90_days`). Split into training and testing sets (e.g., 70/30 split) using `sklearn.model_selection.train_test_split` (set `random_state=42`, `stratify` on `y` for class balance).

4.  **Data Visualization (Matplotlib/Seaborn)**: Create two separate plots to visually inspect relationships with `will_default_in_next_90_days`:
    *   A violin plot (or box plot) showing the distribution of `credit_score` for non-defaulters (0) vs. defaulters (1). Ensure appropriate labels and titles.
    *   A stacked bar chart showing the proportion of `will_default_in_next_90_days` (0 or 1) across different `employment_status` values. Ensure appropriate labels and titles.

5.  **ML Pipeline & Evaluation (Binary Classification)**: 
    *   Create an `sklearn.pipeline.Pipeline` with a `sklearn.compose.ColumnTransformer` for preprocessing:
        *   For numerical features: Apply `sklearn.preprocessing.SimpleImputer(strategy='mean')` followed by `sklearn.preprocessing.StandardScaler`.
        *   For categorical features: Apply `sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')`.
    *   The final estimator in the pipeline should be `sklearn.ensemble.HistGradientBoostingClassifier` (set `random_state=42`).
    *   Train the pipeline on `X_train`, `y_train`. Predict probabilities for the positive class (class 1) on the test set (`X_test`).
    *   Calculate and print the `sklearn.metrics.roc_auc_score` and a `sklearn.metrics.classification_report` for the test set predictions.

## Focus
Loan Default Prediction, Time-Series SQL Aggregations, Binary Classification

## Dataset
Synthetic financial/loan data (applicant profiles, loan details, payment history).

## Hint
When performing SQL aggregations, use `julianday()` for date comparisons and `LEFT JOIN` with `COALESCE` to ensure all loans are included and `NULL`s are handled gracefully for loans with no activity in the aggregation window. For `outstanding_balance_at_cutoff`, ensure you sum payments only up to the cutoff date. Remember to handle potential `NaT` values from merging `default_date` when creating the target variable.
