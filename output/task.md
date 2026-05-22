# AI Daily Lab — 2026-05-22

## Task
Develop a machine learning pipeline to predict if a financial transaction is **fraudulent** (binary classification), based on transaction details and the account's historical behavior up to a specific cutoff date.

1.  **Generate Synthetic Data (Pandas/Numpy)**: Create two pandas DataFrames:
    *   `accounts_df`: With 1000-1500 rows. Columns: `account_id` (unique integers), `signup_date` (random dates over the last 3-5 years), `region` (e.g., 'North', 'South', 'East', 'West'), `account_type` (e.g., 'Standard', 'Premium', 'Basic'), `credit_limit` (random floats 1000-50000, higher for 'Premium').
    *   `transactions_df`: With 20000-30000 rows. Columns: `transaction_id` (unique integers), `account_id` (randomly sampled from `accounts_df` IDs), `transaction_date` (random datetimes occurring *after* their respective `signup_date` and up to `pd.Timestamp.now()`), `amount` (random floats 10.0-5000.0), `transaction_type` (e.g., 'Online Purchase', 'POS Swipe', 'ATM Withdrawal', 'Transfer'), `location_city` (e.g., 'NYC', 'LA', 'Chicago', 'Remote', 'London'), `is_fraud` (binary: 0 or 1).
    *   **Simulate realistic patterns**: Ensure `transaction_date` is always after `signup_date`. Introduce a small percentage (e.g., 3-5%) of `is_fraud=1`. Fraudulent transactions should tend to have: higher `amount` (e.g., 2x-5x average), often 'Online Purchase' or 'Remote' `location_city`, or occur at unusual `hour_of_day`. 'Premium' accounts might have higher `amount`s on average. Ensure a mix of transaction types and locations for each account.
    *   Sort `transactions_df` by `account_id` then `transaction_date`.

2.  **Load into SQLite & SQL Feature Engineering (Historical Transaction Aggregations)**: Create an in-memory SQLite database using `sqlite3`. Load `accounts_df` and `transactions_df` into tables named `accounts` and `transactions` respectively.
    *   Define `GLOBAL_PREDICTION_CUTOFF_DATE` as 7 days prior to the latest `transaction_date` in your generated `transactions_df` (e.g., `transactions_df['transaction_date'].max() - pd.Timedelta(days=7)`).
    *   Write a single SQL query that performs the following for *each transaction that occurred AFTER `GLOBAL_PREDICTION_CUTOFF_DATE`*:
        *   Joins the `transactions` (filtered for events after cutoff) with the `accounts` table.
        *   Aggregates historical features *for the respective `account_id` up to and including `GLOBAL_PREDICTION_CUTOFF_DATE`*:
            *   `avg_amount_account_prev_30d`: Average `amount` for this `account_id` in the 30 days *prior to or on* `GLOBAL_PREDICTION_CUTOFF_DATE`.
            *   `num_transactions_account_prev_30d`: Count of transactions for this `account_id` in the 30 days *prior to or on* `GLOBAL_PREDICTION_CUTOFF_DATE`.
            *   `days_since_last_transaction_account_at_cutoff`: Number of days between `GLOBAL_PREDICTION_CUTOFF_DATE` and the most recent `transaction_date` for this `account_id` *before or on* the cutoff. Return a large number (e.g., 9999) if no prior transactions.
            *   `num_distinct_locations_account_prev_30d`: Count of distinct `location_city` for this `account_id` in the 30 days *prior to or on* `GLOBAL_PREDICTION_CUTOFF_DATE`.
        *   Extracts time-based features from the `transaction_date` of the *current* transaction (e.g., `hour_of_day`, `day_of_week`, `month_of_year`).
        *   Includes static attributes: `transaction_id`, `account_id`, `amount`, `transaction_type`, `location_city`, `region`, `account_type`, `credit_limit`, and the target `is_fraud` for the current transaction.
    *   **Ensures** all transactions *after* the cutoff are included. Handle `NULL`s for historical aggregates (e.g., 0.0 for averages, 0 for counts if no prior activity).
    *   The query should return all mentioned fields.
    *   **Hint**: Use CTEs for pre-calculating account-level historical aggregates up to the `GLOBAL_PREDICTION_CUTOFF_DATE`, then join these with the future transactions.

3.  **Pandas Feature Engineering & Binary Target Creation**: Fetch the SQL query results into a pandas DataFrame (`transaction_features_df`).
    *   Convert all relevant date/datetime columns to appropriate types.
    *   Handle `NaN` values: Fill numerical historical aggregates (e.g., averages, counts) with 0.0 or 0 as appropriate. Fill `days_since_last_transaction_account_at_cutoff` with 9999.
    *   Calculate `amount_ratio_to_credit_limit`: `amount` / (`credit_limit` + 1e-6) to avoid division by zero. Fill any `NaN` or `inf` with 0.
    *   Define features `X` (numerical: `amount`, `credit_limit`, `avg_amount_account_prev_30d`, `num_transactions_account_prev_30d`, `days_since_last_transaction_account_at_cutoff`, `num_distinct_locations_account_prev_30d`, `hour_of_day`, `day_of_week`, `month_of_year`, `amount_ratio_to_credit_limit`; categorical: `region`, `account_type`, `transaction_type`, `location_city`) and target `y` (`is_fraud`).
    *   Split into training and testing sets (e.g., 70/30 split) using `sklearn.model_selection.train_test_split` (set `random_state=42`, `stratify` on `y` to handle class imbalance).

4.  **Data Visualization (Matplotlib/Seaborn)**: Create two separate plots to visually inspect relationships with `is_fraud`:
    *   A violin plot (or box plot) showing the distribution of `amount` for non-fraudulent (0) vs. fraudulent (1) transactions. Consider applying a log transformation to `amount` if the distribution is heavily skewed to improve visibility of relationships.
    *   A stacked bar chart showing the proportion of `is_fraud` (0 or 1) across different `transaction_type` values.
    *   Ensure appropriate labels and titles for both plots.

5.  **ML Pipeline & Evaluation (Binary Classification)**:
    *   Create an `sklearn.pipeline.Pipeline` with a `sklearn.compose.ColumnTransformer` for preprocessing:
        *   For numerical features: Apply `sklearn.preprocessing.SimpleImputer(strategy='mean')` followed by `sklearn.preprocessing.StandardScaler`.
        *   For categorical features: Apply `sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')`.
    *   The final estimator in the pipeline should be `sklearn.ensemble.HistGradientBoostingClassifier` (set `random_state=42`, and consider `class_weight='balanced'` due to imbalance).
    *   Train the pipeline on `X_train`, `y_train`. Predict probabilities for the positive class (class 1) on the test set (`X_test`).
    *   Calculate and print the `sklearn.metrics.roc_auc_score` and a `sklearn.metrics.classification_report` for the test set predictions.

## Focus
Transaction Fraud Detection (Binary Classification)

## Dataset
Synthetic financial transactions and account details.

## Hint
Pay close attention to defining the `GLOBAL_PREDICTION_CUTOFF_DATE` and ensuring all historical features are calculated *only* using data up to this fixed point, applied to predict future transactions. The `stratify` parameter in `train_test_split` and `class_weight` in `HistGradientBoostingClassifier` are crucial for handling the rare `is_fraud=1` class.
