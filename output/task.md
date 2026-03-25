# AI Daily Lab — 2026-03-25

## Task
Develop a machine learning pipeline to predict transactional fraud, leveraging user profiles, merchant characteristics, and historical transactional patterns.

## Focus
Feature engineering from sequential transactional data (user and merchant context) for binary classification, with a focus on fraud detection patterns.

## Dataset
1. **Generate Synthetic Data (Pandas/Numpy)**: Create three pandas DataFrames:
    *   `users_df`: With 500-700 rows. Columns: `user_id` (unique integers), `signup_date` (random dates over the last 5 years), `user_tier` (e.g., 'Bronze', 'Silver', 'Gold'), `region` (e.g., 'North', 'South', 'East', 'West', 'International').
    *   `merchants_df`: With 100-200 rows. Columns: `merchant_id` (unique integers), `category` (e.g., 'Electronics', 'Travel', 'Groceries', 'Utilities', 'Online_Services'), `risk_score` (random integers 0-100, where higher indicates potentially riskier merchant), `is_international` (binary, 0 or 1).
    *   `transactions_df`: With 10000-15000 rows. Columns: `transaction_id` (unique integers), `user_id` (randomly sampled from `users_df` IDs), `merchant_id` (randomly sampled from `merchants_df` IDs), `transaction_date` (random dates occurring *after* `signup_date` of the user), `amount` (random floats 10-5000), `transaction_type` (e.g., 'Card_Present', 'Online', 'ATM_Withdrawal').
    *   **Simulate realistic fraud patterns**: Define `is_fraud` (binary, 0 or 1) for each `transactions_df` row, with an overall 3-5% fraud rate. Bias `is_fraud` such that:
        *   Users in 'Bronze' `user_tier` or 'International' `region` are more prone to fraud.
        *   Merchants with higher `risk_score` or `is_international=1` are more prone to fraud.
        *   Transactions with large `amount` (e.g., >$1000) or `transaction_type='Online'` are more likely to be fraud.
        *   Users who have *previously* committed fraud are significantly more likely to commit fraud again (i.e., sequential fraud detection).
        *   A user's *first* transaction with a new merchant has a slightly higher fraud probability.
    *   Sort `transactions_df` by `user_id` then `transaction_date` for easier sequential processing.

2. **Load into SQLite & SQL Feature Engineering (Transaction-Level Contextual Features)**: Create an in-memory SQLite database using `sqlite3`. Load `users_df`, `merchants_df`, and `transactions_df` into tables named `users`, `merchants`, and `transactions` respectively.
    Write a single SQL query that performs the following for *each transaction* in `transactions`:
    *   **Joins** `users`, `merchants`, and `transactions` tables.
    *   **Calculates sequential features based on the user's *prior transactions* and the merchant's *prior transactions* (excluding the current one), relative to the current `transaction_date`**:
        *   `user_prior_num_transactions`: Count of all *previous* transactions for the same user.
        *   `user_prior_total_spend`: Sum of `amount` for all *previous* transactions for the same user.
        *   `user_avg_prior_transaction_amount`: Average `amount` of all *previous* transactions for the same user.
        *   `user_prior_num_fraud_transactions`: Count of *previous* transactions for the same user that were marked `is_fraud=1`.
        *   `days_since_last_user_transaction`: Number of days between the current `transaction_date` and the user's *most recent prior* `transaction_date`. If no prior transaction, use the number of days between `signup_date` and the current `transaction_date`.
        *   `merchant_prior_num_transactions`: Count of all *previous* transactions for the same merchant (across all users).
        *   `merchant_avg_prior_transaction_amount`: Average `amount` of *previous* transactions for the same merchant.
        *   `merchant_prior_num_fraud_transactions`: Count of *previous* transactions for the same merchant that were marked `is_fraud=1`.
    *   **Includes static user, merchant, and current transaction attributes**: `transaction_id`, `user_id`, `merchant_id`, `transaction_date`, `amount`, `transaction_type`, `is_fraud` (the target), `user_tier`, `region`, `category`, `risk_score`, `is_international`, `signup_date`.
    *   The query should return all these attributes and engineered features. Missing values for prior aggregates/dates should be `NULL`.

3. **Pandas Feature Engineering & Binary Target Creation**: Fetch the SQL query results into a pandas DataFrame (`transaction_features_df`).
    *   Handle `NaN` values: Fill `user_prior_num_transactions`, `user_prior_total_spend`, `user_prior_num_fraud_transactions`, `merchant_prior_num_transactions`, `merchant_prior_num_fraud_transactions` with 0. Fill `user_avg_prior_transaction_amount`, `merchant_avg_prior_transaction_amount` with 0.0. For `days_since_last_user_transaction` (for a user's first transaction), if `NaN`s remain, fill with `days_since_signup_at_transaction`.
    *   Convert `signup_date` and `transaction_date` to datetime objects.
    *   Calculate `days_since_signup_at_transaction`: Days between `signup_date` and `transaction_date`.
    *   Calculate `user_prior_fraud_rate`: `user_prior_num_fraud_transactions` / (`user_prior_num_transactions` if `user_prior_num_transactions` > 0 else 1.0). Fill any `NaN`s with 0.
    *   Calculate `merchant_prior_fraud_rate`: `merchant_prior_num_fraud_transactions` / (`merchant_prior_num_transactions` if `merchant_prior_num_transactions` > 0 else 1.0). Fill any `NaN`s with 0.
    *   Calculate `amount_deviation_from_user_avg_prior`: `amount` - `user_avg_prior_transaction_amount`. Fill `NaN`s with 0.
    *   Define features `X` (all numerical: `amount`, `risk_score`, `user_prior_num_transactions`, `user_prior_total_spend`, `user_avg_prior_transaction_amount`, `user_prior_num_fraud_transactions`, `days_since_last_user_transaction`, `merchant_prior_num_transactions`, `merchant_avg_prior_transaction_amount`, `merchant_prior_num_fraud_transactions`, `days_since_signup_at_transaction`, `user_prior_fraud_rate`, `merchant_prior_fraud_rate`, `amount_deviation_from_user_avg_prior`; categorical: `transaction_type`, `user_tier`, `region`, `category`, `is_international`) and target `y` (`is_fraud`). Split into training and testing sets (e.g., 70/30 split) using `sklearn.model_selection.train_test_split` (set `random_state=42`, `stratify` on `y` due to low fraud rate).

4. **Data Visualization**: Create two separate plots to visually inspect relationships with `is_fraud`:
    *   A violin plot (or box plot) showing the distribution of `amount` for non-fraud (0) vs. fraud (1) transactions. Ensure appropriate labels and titles.
    *   A stacked bar chart showing the proportion of `is_fraud` (0 or 1) across different `category` values. Ensure appropriate labels and titles.

5. **ML Pipeline & Evaluation (Binary Classification)**: 
    *   Create an `sklearn.pipeline.Pipeline` with a `sklearn.compose.ColumnTransformer` for preprocessing:
        *   For numerical features: Apply `sklearn.preprocessing.SimpleImputer(strategy='mean')` followed by `sklearn.preprocessing.StandardScaler`.
        *   For categorical features: Apply `sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')`.
    *   The final estimator in the pipeline should be `sklearn.ensemble.HistGradientBoostingClassifier` (set `random_state=42`, and consider `class_weight='balanced'` if the dataset is highly imbalanced, though `stratify` helps). A `RandomForestClassifier` or `XGBoostClassifier` could also be good alternatives.
    *   Train the pipeline on `X_train`, `y_train`. Predict probabilities for the positive class (class 1) on the test set (`X_test`).
    *   Calculate and print the `sklearn.metrics.roc_auc_score` and a `sklearn.metrics.classification_report` for the test set predictions.

## Hint
When simulating `is_fraud`, ensure that the sequential nature (e.g., users who committed fraud before are more likely to do so again) is captured. For SQL, use `LAG` with `PARTITION BY user_id ORDER BY transaction_date` to get prior values for user-specific features. For merchant-specific features, use `PARTITION BY merchant_id ORDER BY transaction_date`. Remember to handle `NULL` results from `LAG` (for first transactions) appropriately. `julianday(date2) - julianday(date1)` calculates days difference.
