# AI Daily Lab — 2026-02-20

## Task
1. **Generate Synthetic Data (Pandas/Numpy)**: Create two pandas DataFrames:
    *   `users_df`: With 500-700 rows. Columns: `user_id` (unique integers), `signup_date` (random dates over the last 5 years), `region` (e.g., 'North', 'South', 'East', 'West'), `age` (random integers 18-70), `is_fraudulent` (binary target, 0 or 1, with an approximate 5-10% fraud rate).
    *   `transactions_df`: With 5000-8000 rows. Columns: `transaction_id` (unique integers), `user_id` (randomly sampled from `users_df` IDs), `transaction_date` (random dates *after* their respective `signup_date`), `amount` (random floats between 10.0 and 5000.0), `merchant_category` (e.g., 'Groceries', 'Retail', 'Dining', 'Travel', 'Online_Service'), `location_country` (e.g., 'USA', 'Canada', 'UK', 'Mexico', 'Japan').
    *   **Simulate realistic fraud patterns**: Ensure `transaction_date` is always after `signup_date`. For fraudulent users (`is_fraudulent=1`), simulate patterns like:
        *   Higher average `amount`s or very large `amount`s in some transactions.
        *   More frequent transactions within shorter periods (bursts of activity).
        *   Transactions often occurring from multiple `location_country`s within a short timeframe.
        *   Transaction activity might be concentrated closer to their `signup_date` and then stop abruptly compared to non-fraudulent users.
    *   Sort `transactions_df` by `user_id` then `transaction_date` for easier sequential processing.

2. **Load into SQLite & SQL Feature Engineering (User Transaction Patterns)**: Create an in-memory SQLite database using `sqlite3`. Load `users_df` into a table named `users` and `transactions_df` into a table named `transactions`. Determine a `global_analysis_date` (e.g., `max(transaction_date)` from `transactions_df` + 60 days, using pandas) and a `feature_cutoff_date` (`global_analysis_date` - 90 days).
    Write a single SQL query that performs the following for *each user*, aggregating their transaction behavior *before* the `feature_cutoff_date`:
    *   **Joins** `users` and `transactions` tables.
    *   **Aggregates features based on transactions *before* `feature_cutoff_date`**: 
        *   `total_spend_pre_cutoff` (sum of `amount`)
        *   `num_transactions_pre_cutoff` (count of `transaction_id`s)
        *   `avg_transaction_value_pre_cutoff` (average `amount`)
        *   `max_transaction_value_pre_cutoff` (maximum `amount`)
        *   `num_unique_merchant_categories_pre_cutoff` (count of distinct `merchant_category`s).
        *   `num_unique_location_countries_pre_cutoff` (count of distinct `location_country`s).
        *   `days_since_last_transaction_pre_cutoff`: Number of days between `feature_cutoff_date` and `MAX(transaction_date)` for the user (only considering transactions before `feature_cutoff_date`).
        *   `transaction_span_days_pre_cutoff`: Number of days between `MIN(transaction_date)` and `MAX(transaction_date)` for the user (only considering transactions before `feature_cutoff_date`). If only one transaction, this should be 0.
    *   **Includes static user attributes**: `user_id`, `age`, `region`, `signup_date`, `is_fraudulent` (the true target from `users_df`).
    *   **Ensures** all users are included (using a `LEFT JOIN`), showing 0 for counts/sums, 0.0 for averages/max, and `NULL` for `days_since_last_transaction_pre_cutoff`/`transaction_span_days_pre_cutoff` if no transactions before cutoff.
    *   The query should return `user_id`, `age`, `region`, `signup_date`, `is_fraudulent`, and all the aggregated features.
    *   **Hint**: Use `strftime('%J', ...)` for Julian day differences to calculate days in SQLite, then convert to integer days.

3. **Pandas Feature Engineering & Binary Target Creation**: Fetch the SQL query results into a pandas DataFrame (`user_fraud_features_df`).
    *   Handle `NaN` values: Fill `total_spend_pre_cutoff`, `num_transactions_pre_cutoff`, `avg_transaction_value_pre_cutoff`, `max_transaction_value_pre_cutoff`, `num_unique_merchant_categories_pre_cutoff`, `num_unique_location_countries_pre_cutoff`, `transaction_span_days_pre_cutoff` with 0. For `days_since_last_transaction_pre_cutoff` (for users with no activities before cutoff), fill with a large sentinel value (e.g., 9999 days).
    *   Convert `signup_date` to datetime objects. Calculate `account_age_at_cutoff_days`: The number of days between `signup_date` and the `feature_cutoff_date`.
    *   Calculate `transaction_frequency_pre_cutoff`: `num_transactions_pre_cutoff` / (`account_age_at_cutoff_days` + 1). Use `+1` to prevent division by zero for very new accounts at cutoff.
    *   Calculate `avg_transaction_per_span_pre_cutoff`: `num_transactions_pre_cutoff` / (`transaction_span_days_pre_cutoff` + 1) for users with `transaction_span_days_pre_cutoff` > 0, else 0. (Handles users with 0 or 1 transaction in the span).
    *   Define features `X` (all numerical and categorical features engineered) and target `y` (`is_fraudulent`). Split into training and testing sets (e.g., 70/30 split) using `sklearn.model_selection.train_test_split` (set `random_state=42`, `stratify` on `y` for class balance due to potential imbalance in fraud labels).

4. **Data Visualization**: Create two separate plots to visually inspect relationships with `is_fraudulent`:
    *   A violin plot (or box plot) showing the distribution of `total_spend_pre_cutoff` for users with `is_fraudulent=0` vs. `is_fraudulent=1`. Ensure appropriate labels and titles.
    *   A stacked bar chart showing the proportion of `is_fraudulent` (0 or 1) across different `region` values. Ensure appropriate labels and titles.

5. **ML Pipeline & Evaluation (Binary Classification)**: 
    *   Create an `sklearn.pipeline.Pipeline` with a `sklearn.compose.ColumnTransformer` for preprocessing:
        *   For numerical features (e.g., `age`, `account_age_at_cutoff_days`, `total_spend_pre_cutoff`, `num_transactions_pre_cutoff`, `avg_transaction_value_pre_cutoff`, `max_transaction_value_pre_cutoff`, `num_unique_merchant_categories_pre_cutoff`, `num_unique_location_countries_pre_cutoff`, `days_since_last_transaction_pre_cutoff`, `transaction_span_days_pre_cutoff`, `transaction_frequency_pre_cutoff`, `avg_transaction_per_span_pre_cutoff`): Apply `sklearn.preprocessing.SimpleImputer(strategy='mean')` followed by `sklearn.preprocessing.StandardScaler`.
        *   For categorical features (`region`): Apply `sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')`.
    *   The final estimator in the pipeline should be `sklearn.ensemble.HistGradientBoostingClassifier` (set `random_state=42`).
    *   Train the pipeline on `X_train`, `y_train`. Predict probabilities for the positive class (class 1) on the test set (`X_test`).
    *   Calculate and print the `sklearn.metrics.roc_auc_score` and a `sklearn.metrics.classification_report` for the test set predictions.

## Focus
User-Level Fraud Detection based on Aggregated Transaction Patterns

## Dataset
Synthetic data of user profiles and their historical transactions, with a hidden 'is_fraudulent' label for users.

## Hint
Focus on aggregating various transaction metrics (counts, sums, averages, diversity, time-based differences) at the user level before the cutoff date to capture behavioral anomalies indicative of fraud. The `is_fraudulent` flag from `users_df` is your direct target for this user-level classification.
