# AI Daily Lab — 2026-01-26

## Task
1. **Generate Synthetic Data (Pandas/Numpy)**: Create two pandas DataFrames:
    *   `customers_df`: With 500 rows. Columns: `customer_id` (unique integers), `signup_date` (random dates over the last 5 years), `region` (e.g., 'North', 'South', 'East', 'West'), `age` (random integers 18-70).
    *   `transactions_df`: With 3000-5000 rows. Columns: `transaction_id` (unique integers), `customer_id` (randomly sampled from `customers_df` IDs, ensuring some customers have many transactions and a few have no transactions), `transaction_date` (random dates *after* their respective `signup_date`), `amount` (random floats between 10.0 and 1000.0).

2. **Load into SQLite & SQL Feature Engineering (RFM)**: Create an in-memory SQLite database. Load `customers_df` into a table named `customers` and `transactions_df` into a table named `transactions`. Determine an `analysis_date` (e.g., the latest `transaction_date` from `transactions_df` + 30 days, using pandas).
    Write a single SQL query that calculates Recency, Frequency, and Monetary (RFM) values for *each customer*:
    *   `recency_days`: Number of days between the `analysis_date` and the customer's `MAX(transaction_date)`. If no transactions, this should be `NULL`.
    *   `frequency`: Total number of transactions for the customer.
    *   `monetary`: Sum of all `amount`s for the customer.
    *   **Ensures** all customers are included, showing 0 for `frequency` and `monetary`, and `NULL` for `recency_days` if no transactions.
    *   The query should return `customer_id`, `region`, `age`, `signup_date`, `recency_days`, `frequency`, `monetary`.

3. **Pandas Feature Engineering & Target Creation**: Fetch the SQL query results into a pandas DataFrame. 
    *   Handle `NaN` values: Fill `frequency` and `monetary` with 0. For `recency_days` (for customers with no transactions), fill with a large sentinel value, e.g., `365 * 5` (1825 days).
    *   Calculate `account_age_days`: Days between `signup_date` and the `analysis_date` (from step 2).
    *   **Create Binary Target `is_high_value_customer`**: A customer is 'high value' (1) if their `monetary` value is in the top 30% *and* their `frequency` is in the top 30%. Otherwise, 0. (Hint: Use `quantile()` to find thresholds).
    *   Define features `X` (`region`, `age`, `account_age_days`, `recency_days`, `frequency`, `monetary`) and target `y` (`is_high_value_customer`). Split into training and testing sets (e.g., 70/30 split) using `sklearn.model_selection.train_test_split` (set `random_state=42`).

4. **Data Visualization**: Create two separate plots to visually inspect relationships with `is_high_value_customer`:
    *   A violin plot or box plot showing the distribution of `recency_days` for each `is_high_value_customer` group.
    *   A bar plot or count plot showing the distribution of `is_high_value_customer` across different `region`s.
    Ensure plots have appropriate labels and titles.

5. **ML Pipeline & Evaluation**: 
    *   Create an `sklearn.pipeline.Pipeline` with a `ColumnTransformer` for preprocessing:
        *   For numerical features (`age`, `account_age_days`, `recency_days`, `frequency`, `monetary`): Apply `sklearn.preprocessing.StandardScaler`.
        *   For the categorical feature (`region`): Apply `sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')`.
    *   The final estimator in the pipeline should be `sklearn.linear_model.LogisticRegression` (set `random_state=42`, `solver='liblinear'` for reproducibility).
    *   Train the pipeline on the training data (`X_train`, `y_train`). Predict probabilities for the positive class (class 1) on the test set (`X_test`).
    *   Calculate and print the `sklearn.metrics.roc_auc_score` for the test set predictions.
    *   Generate and display an ROC curve for the model using `sklearn.metrics.RocCurveDisplay.from_estimator` with the trained pipeline and test data.

## Focus
RFM Feature Engineering with SQL, Customer Segmentation/Classification, ML Pipelines, and Evaluation.

## Dataset
Synthetic Customer and Transactional Data

## Hint
When calculating `recency_days` in SQL, use `JULIANDAY(analysis_date) - JULIANDAY(MAX(transaction_date))` to get the difference in days. For the target variable, remember to calculate quantiles *before* defining the 'top 30%' thresholds.
