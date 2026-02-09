# AI Daily Lab — 2026-02-09

## Task
1. **Generate Synthetic Data (Pandas/Numpy)**: Create two pandas DataFrames:
    *   `users_df`: With 500-700 rows. Columns: `user_id` (unique integers), `signup_date` (random dates over the last 5 years), `region` (e.g., 'North', 'South', 'East', 'West'), `age` (random integers 18-70).
    *   `transactions_df`: With 5000-8000 rows. Columns: `transaction_id` (unique integers), `user_id` (randomly sampled from `users_df` IDs), `transaction_date` (random dates *after* their respective `signup_date`), `amount` (random floats between 10.0 and 2000.0), `merchant_category` (e.g., 'Groceries', 'Retail', 'Dining', 'Travel', 'Online_Service'), `location_country` (e.g., 'USA', 'Canada', 'UK', 'Mexico', 'Japan').
    *   **Simulate realistic transaction patterns**: Ensure `transaction_date` is always after `signup_date`. Generate data such that users have varying frequencies and amounts. Sort `transactions_df` by `user_id` then `transaction_date` for easier sequential processing.

2. **Load into SQLite & SQL Feature Engineering (Sequential Transaction Analysis)**: Create an in-memory SQLite database using `sqlite3`. Load `users_df` into a table named `users` and `transactions_df` into a table named `transactions`.
    Write a single SQL query that performs the following for *each transaction* in `transactions`:
    *   **Joins** `transactions` with `users` to get user attributes.
    *   **Calculates sequential features based on the user's *prior* transactions (excluding the current one)**:
        *   `user_avg_spend_prior`: Average `amount` of all *previous* transactions by the same user. If no previous transactions, use 0.0.
        *   `user_max_spend_prior`: Maximum `amount` of all *previous* transactions by the same user. If no previous transactions, use 0.0.
        *   `user_num_transactions_prior`: Count of all *previous* transactions by the same user. If no previous transactions, use 0.
        *   `days_since_last_transaction`: Number of days between the current `transaction_date` and the user's *most recent prior* `transaction_date`. If it's the user's first transaction, use the number of days between `signup_date` and `transaction_date`.
    *   The query should return `transaction_id`, `user_id`, `transaction_date`, `amount`, `merchant_category`, `location_country`, `region`, `age`, `signup_date`, `user_avg_spend_prior`, `user_max_spend_prior`, `user_num_transactions_prior`, `days_since_last_transaction`.
    *   **Hint**: Use window functions with `ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING` and `LAG()` with a default value.

3. **Pandas Feature Engineering & Binary Target Creation (Suspicious Transaction Detection)**: Fetch the SQL query results into a pandas DataFrame (`transaction_features_df`).
    *   Handle `NaN` values: Ensure `user_avg_spend_prior`, `user_max_spend_prior`, `user_num_transactions_prior` are filled with 0 where appropriate (SQL should handle this, but double check). Fill `days_since_last_transaction` with a large sentinel value (e.g., 9999) if any `NaN`s remain (e.g., in a specific edge case for first transaction, although SQL should correctly populate it).
    *   Calculate `amount_vs_avg_prior_ratio`: `amount` / (`user_avg_spend_prior` if `user_avg_spend_prior` > 0 else `amount`). This creates a ratio where 1.0 means current amount equals prior average, and handles division by zero by using the current amount itself (ratio of 1) for first transactions. 
    *   Calculate `is_first_transaction`: A binary flag (1 if `user_num_transactions_prior == 0`, else 0).
    *   **Create Binary Target `is_suspicious`**: A transaction is considered 'suspicious' (1) if `amount` > 1000 (absolute high amount) 
        OR (`amount_vs_avg_prior_ratio` > 2.5 AND `days_since_last_transaction` < 1.0 AND `user_num_transactions_prior` > 0). Otherwise, 0.
    *   Define features `X` (`region`, `age`, `merchant_category`, `location_country`, `amount`, `user_avg_spend_prior`, `user_max_spend_prior`, `user_num_transactions_prior`, `days_since_last_transaction`, `amount_vs_avg_prior_ratio`, `is_first_transaction`) and target `y` (`is_suspicious`). Split into training and testing sets (e.g., 70/30 split) using `sklearn.model_selection.train_test_split` (set `random_state=42`, `stratify` on `y` for class balance).

4. **Data Visualization**: Create two separate plots to visually inspect relationships with `is_suspicious`:
    *   A violin plot (or box plot) showing the distribution of `amount_vs_avg_prior_ratio` for `is_suspicious=0` vs. `is_suspicious=1`.
    *   A stacked bar chart showing the proportion of `is_suspicious` (0 or 1) across different `merchant_category` values.
    Ensure plots have appropriate labels and titles.

5. **ML Pipeline & Evaluation (Binary Classification)**: 
    *   Create an `sklearn.pipeline.Pipeline` with a `sklearn.compose.ColumnTransformer` for preprocessing:
        *   For numerical features (`age`, `amount`, `user_avg_spend_prior`, `user_max_spend_prior`, `user_num_transactions_prior`, `days_since_last_transaction`, `amount_vs_avg_prior_ratio`): Apply `sklearn.preprocessing.SimpleImputer(strategy='mean')` followed by `sklearn.preprocessing.StandardScaler`.
        *   For categorical features (`region`, `merchant_category`, `location_country`, `is_first_transaction`): Apply `sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')`.
    *   The final estimator in the pipeline should be `sklearn.ensemble.GradientBoostingClassifier` (set `random_state=42`, `n_estimators=100`, `learning_rate=0.1`).
    *   Train the pipeline on `X_train`, `y_train`. Predict probabilities for the positive class (class 1) on the test set (`X_test`).
    *   Calculate and print the `sklearn.metrics.roc_auc_score` and a `sklearn.metrics.classification_report` for the test set predictions.

## Focus
Transaction-level Anomaly Detection, Sequential Feature Engineering, Imbalanced Classification

## Dataset
Synthetic transaction and user data, including historical spending patterns.

## Hint
Pay close attention to the SQL window functions (`LAG`, `ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING`) to correctly calculate features based on *prior* transactions for each row. When creating the `amount_vs_avg_prior_ratio`, carefully handle cases where `user_avg_spend_prior` is zero (for a user's first transaction) to avoid division by zero errors.
