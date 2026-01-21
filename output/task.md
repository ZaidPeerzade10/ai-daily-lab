# AI Daily Lab — 2026-01-21

## Task
1. **Generate Synthetic Data (Pandas/Numpy)**: Create two pandas DataFrames:
    *   `customers_df`: With 500 rows. Columns: `customer_id` (unique integers), `signup_date` (random dates over the last 5 years), `region` (e.g., 'North', 'South', 'East', 'West' with random distribution), `age` (random integers 18-70).
    *   `transactions_df`: With 2000-3000 rows. Columns: `transaction_id` (unique integers), `customer_id` (randomly sampled from `customers_df` IDs, ensuring some customers have no transactions and others have many), `transaction_date` (random dates after their `signup_date`), `amount` (random floats between 10.0 and 1000.0).

2. **Load into SQLite**: Create an in-memory SQLite database using `sqlite3` and load `customers_df` into a table named `customers` and `transactions_df` into `transactions`.

3. **SQL Feature Engineering (Customer Value)**: First, determine an `analysis_date` (e.g., the maximum `transaction_date` in `transactions_df` plus 30 days, using pandas). Then, write a single SQL query that performs the following for *each customer*:
    *   **Joins** `customers` and `transactions` tables.
    *   **Aggregates** to calculate `total_spend`, `num_transactions`, `avg_transaction_value` (total_spend / num_transactions).
    *   Calculates `days_since_last_purchase`: The number of days between the `analysis_date` and the customer's `MAX(transaction_date)`.
    *   Calculates `days_since_first_purchase`: The number of days between the `analysis_date` and the customer's `MIN(transaction_date)`.
    *   **Ensures** that all customers are included in the result, even those with no transactions, showing 0 or NULL for aggregated values as appropriate.
    *   The query should return `customer_id`, `region`, `age`, `signup_date`, `total_spend`, `num_transactions`, `avg_transaction_value`, `days_since_last_purchase`, `days_since_first_purchase`.

4. **Retrieve, Merge, and Pandas Feature Engineering**: 
    *   Fetch the SQL query results into a pandas DataFrame.
    *   Calculate `account_age_days`: The number of days between `signup_date` and the `analysis_date` (from step 3).
    *   Handle `NaN` values resulting from customers with no transactions:
        *   Fill `total_spend`, `num_transactions`, `avg_transaction_value` with 0.
        *   For `days_since_last_purchase` and `days_since_first_purchase` (for customers with no transactions), fill with a large sentinel value, e.g., `account_age_days` + 30 days (or 5 years in days).
    *   Define features `X` (`age`, `region`, `account_age_days`, `total_spend`, `num_transactions`, `avg_transaction_value`, `days_since_last_purchase`, `days_since_first_purchase`) and target `y` (`total_spend`).
    *   Split the data into training and testing sets (e.g., 70/30 split) using `sklearn.model_selection.train_test_split` (set `random_state=42`).

5. **Data Visualization**: Create two visualizations:
    *   A histogram showing the distribution of the target variable `total_spend`.
    *   A box plot or violin plot showing the distribution of `total_spend` for each `region`.

6. **ML Pipeline & Evaluation**: 
    *   Create an `sklearn.pipeline.Pipeline` with a `ColumnTransformer` for preprocessing:
        *   For numerical features (e.g., `age`, `account_age_days`, `total_spend`, `num_transactions`, `avg_transaction_value`, `days_since_last_purchase`, `days_since_first_purchase`): Apply `StandardScaler`.
        *   For categorical features (`region`): Apply `OneHotEncoder(handle_unknown='ignore')`.
    *   The final estimator in the pipeline should be `sklearn.ensemble.RandomForestRegressor` (set `random_state=42`, `n_estimators=100`).
    *   Train the pipeline on the training data (`X_train`, `y_train`). Predict `total_spend` for the test set (`X_test`).
    *   Calculate and print the `sklearn.metrics.mean_absolute_error` (MAE) and `sklearn.metrics.r2_score` for the test set predictions.

## Focus
Customer Value Prediction (Regression) using SQL-driven RFM-like Feature Engineering and ML Pipelines.

## Dataset
Synthetic customer demographic and transactional data.

## Hint
Pay close attention to handling dates (converting to datetime, calculating differences) and managing `NULL`/`NaN` values from `LEFT JOIN` operations in SQL and during Pandas feature engineering, especially for customers with no transactions. For `analysis_date`, calculate it based on the maximum transaction date in your synthetic data to simulate a realistic cut-off point.
