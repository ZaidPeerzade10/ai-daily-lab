# AI Daily Lab — 2026-02-15

## Task
1. **Generate Synthetic Data (Pandas/Numpy)**: Create two pandas DataFrames:
    *   `users_df`: With 500-700 rows. Columns: `user_id` (unique integers), `signup_date` (random dates over the last 5 years), `region` (e.g., 'North', 'South', 'East', 'West'), `age` (random integers 18-70), `acquisition_channel` (e.g., 'Organic', 'Social', 'Referral', 'Paid_Ad').
    *   `transactions_df`: With 3000-5000 rows. Columns: `transaction_id` (unique integers), `user_id` (randomly sampled from `users_df` IDs, ensuring some users have many transactions, some few, and some none), `transaction_date` (random dates occurring *after* their respective `signup_date`), `amount` (random floats between 10.0 and 1000.0), `product_category` (e.g., 'Electronics', 'Books', 'Clothing', 'Groceries', 'Services').
    *   **Simulate Realistic Behavior**: Ensure `transaction_date` is always after `signup_date`. Generate data such that users have varying frequencies and amounts. Some users might primarily purchase from specific categories or have bursts of activity.

2. **Load into SQLite & SQL Feature Engineering**: Create an in-memory SQLite database using `sqlite3`. Load `users_df` into a table named `users` and `transactions_df` into a table named `transactions`. Determine a `global_analysis_date` (e.g., `max(transaction_date)` from `transactions_df` + 60 days, using pandas) and a `feature_cutoff_date` (`global_analysis_date` - 90 days).
    Write a single SQL query that performs the following for *each user*, aggregating their transaction behavior *before* the `feature_cutoff_date`:
    *   **Joins** `users` and `transactions` tables.
    *   **Aggregates features based on transactions *before* `feature_cutoff_date`**: 
        *   `total_spend_pre_cutoff` (sum of `amount`)
        *   `num_transactions_pre_cutoff` (count of `transaction_id`s)
        *   `avg_transaction_value_pre_cutoff` (average `amount`)
        *   `num_unique_categories_pre_cutoff` (count of distinct `product_category`s).
        *   `days_since_last_transaction_pre_cutoff`: Number of days between `feature_cutoff_date` and `MAX(transaction_date)` for the user (only considering transactions before `feature_cutoff_date`).
    *   **Includes static user attributes**: `user_id`, `age`, `region`, `acquisition_channel`, `signup_date`.
    *   **Ensures** all users are included (using a `LEFT JOIN`), showing 0 for counts/sums, 0.0 for averages, and `NULL` for `days_since_last_transaction_pre_cutoff` if no transactions before cutoff.
    *   The query should return `user_id`, `age`, `region`, `acquisition_channel`, `signup_date`, and all the aggregated features.

3. **Pandas Feature Engineering & Multi-Class Target Creation**: Fetch the SQL query results into a pandas DataFrame (`user_features_df`).
    *   Handle `NaN` values: Fill `total_spend_pre_cutoff`, `num_transactions_pre_cutoff`, `num_unique_categories_pre_cutoff` with 0. Fill `avg_transaction_value_pre_cutoff` with 0.0. For `days_since_last_transaction_pre_cutoff` (for users with no activities before cutoff), fill with a large sentinel value (e.g., `account_age_at_cutoff_days` + 30).
    *   Convert `signup_date` to datetime objects. Calculate `account_age_at_cutoff_days`: The number of days between `signup_date` and the `feature_cutoff_date`.
    *   **Calculate Future Spend**: From the original `transactions_df`, calculate `total_spend_future` (sum of `amount`) for each user for transactions occurring *between `feature_cutoff_date` and `global_analysis_date`*. Merge this aggregate with `user_features_df` (left join), filling `NaN`s with 0.
    *   **Create the Multi-Class Target `future_spending_tier`**: Based on `total_spend_future`. First, calculate the 33rd and 66th percentiles for *non-zero* `total_spend_future`. Then, define segments:
        *   'No_Future_Spend': `total_spend_future` == 0.
        *   'Low_Spender': `total_spend_future` > 0 AND `total_spend_future` <= 33rd percentile.
        *   'Medium_Spender': `total_spend_future` > 33rd percentile AND `total_spend_future` <= 66th percentile.
        *   'High_Spender': `total_spend_future` > 66th percentile.
    *   Define features `X` (all numerical and categorical features engineered) and target `y` (`future_spending_tier`). Split into training and testing sets (e.g., 70/30 split) using `sklearn.model_selection.train_test_split` (set `random_state=42`, `stratify` on `y` for class balance).

4. **Data Visualization**: Create two separate plots to visually inspect relationships with `future_spending_tier`:
    *   A violin plot (or box plot) showing the distribution of `total_spend_pre_cutoff` for each `future_spending_tier`.
    *   A stacked bar chart showing the distribution of `future_spending_tier` across different `region`s.
    Ensure plots have appropriate labels and titles.

5. **ML Pipeline & Evaluation (Multi-Class)**: 
    *   Create an `sklearn.pipeline.Pipeline` with a `sklearn.compose.ColumnTransformer` for preprocessing:
        *   For numerical features (e.g., `age`, `account_age_at_cutoff_days`, `total_spend_pre_cutoff`, `num_transactions_pre_cutoff`, `avg_transaction_value_pre_cutoff`, `num_unique_categories_pre_cutoff`, `days_since_last_transaction_pre_cutoff`): Apply `sklearn.preprocessing.SimpleImputer(strategy='mean')` followed by `sklearn.preprocessing.StandardScaler`.
        *   For categorical features (`region`, `acquisition_channel`): Apply `sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')`.
    *   The final estimator in the pipeline should be `sklearn.ensemble.RandomForestClassifier` (set `random_state=42`, `n_estimators=100`, `class_weight='balanced'` for potential class imbalance).
    *   Train the pipeline on `X_train`, `y_train`. Predict `future_spending_tier` for `X_test`.
    *   Calculate and print the `sklearn.metrics.accuracy_score` and a `sklearn.metrics.classification_report` for the test set predictions.

## Focus
Customer Segmentation, Time-Windowed Feature Engineering, Multi-Class Classification

## Dataset
Synthetic customer and transaction data.

## Hint
Pay close attention to the time windows defined by `global_analysis_date` and `feature_cutoff_date` for both feature aggregation in SQL and target calculation in pandas. Use SQL's `LEFT JOIN` and `GROUP BY` with `CASE` statements or `FILTER` clause for conditional aggregation. Remember to handle `NaN` values gracefully, especially for users with no activity in specific windows.
