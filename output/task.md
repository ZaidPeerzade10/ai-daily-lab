# AI Daily Lab — 2026-04-21

## Task
Develop a machine learning pipeline to predict fraudulent transactions at the point of occurrence, based on transaction details, user profile, and recent user transaction history.

## Focus
Transaction Fraud Detection (Binary Classification) with time-series feature engineering (lagged/rolling features).

## Dataset
Synthetic financial transaction data with user profiles.

## Hint
1. **Generate Synthetic Data (Pandas/Numpy)**: Create `users_df` (user_id, signup_date, country, device_type, age) and `transactions_df` (transaction_id, user_id, transaction_date, amount, merchant_category, transaction_ip, is_fraudulent). Simulate realistic fraud patterns: e.g., higher amounts for fraudulent, unusual `merchant_category`, `transaction_ip` inconsistencies with `country`, faster successive transactions for fraudulent accounts. Keep fraud rate low (e.g., 2-5%). Ensure `transaction_date` is after `signup_date`. Sort `transactions_df` by `user_id` then `transaction_date`.

2. **Load into SQLite & SQL Feature Engineering (Transaction Attributes)**: Create an in-memory SQLite database. Load `users_df` and `transactions_df` into tables. Write a single SQL query for *each transaction* that joins `transactions` and `users` to include `country`, `device_type`, `age`, `signup_date`. Extract `transaction_hour_of_day` and `transaction_day_of_week`. Calculate `days_since_user_signup` at the time of transaction. Simulate `ip_country_match` (binary: 1 if `transaction_ip` broadly aligns with `user.country`, 0 otherwise – e.g., a simple prefix match for synthetic data).

3. **Pandas Feature Engineering & Binary Target Creation**: Fetch SQL results into a pandas DataFrame (`transaction_features_df`). Convert date columns to datetime objects. For *each user*, calculate historical transaction features *up to the current transaction* using Pandas `groupby()` and `rolling()`/`shift()` with a time window (e.g., '7D' or '30D'):
    *   `time_since_last_transaction_seconds` (for the same user)
    *   `user_avg_amount_past_7d` (average amount of previous transactions for the user in the last 7 days)
    *   `user_num_transactions_past_7d` (count of previous transactions for the user in the last 7 days)
    *   `user_max_amount_past_7d`
    *   Ensure `closed='left'` for `rolling()` windows to prevent data leakage from the current transaction. Fill `NaN`s for these new features (e.g., with 0 or mean/median). Calculate additional features like `amount_per_age` (`amount` / `age`). Define `X` and `y` (`is_fraudulent`). Split data (e.g., 70/30) using `train_test_split` with `random_state=42` and `stratify=y`.

4. **Data Visualization**: Create two plots:
    *   A violin plot (or box plot) showing the distribution of `amount` for fraudulent (1) vs. non-fraudulent (0) transactions.
    *   A stacked bar chart showing the proportion of `is_fraudulent` (0 or 1) across different `device_type` values.

5. **ML Pipeline & Evaluation (Binary Classification)**: Create an `sklearn.pipeline.Pipeline`. Use a `ColumnTransformer` for preprocessing: `SimpleImputer(strategy='mean')` and `StandardScaler` for numerical features; `OneHotEncoder(handle_unknown='ignore')` for categorical features. The final estimator should be `sklearn.ensemble.HistGradientBoostingClassifier(random_state=42)`. Train the pipeline, predict probabilities on the test set, and print `sklearn.metrics.roc_auc_score` and a `sklearn.metrics.classification_report`.
