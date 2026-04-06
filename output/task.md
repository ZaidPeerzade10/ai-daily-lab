# AI Daily Lab — 2026-04-06

## Task
Develop a machine learning pipeline to predict the likelihood of a product being returned, based on product characteristics, customer attributes, and transaction details.

## Focus
Predicting Product Returns (Binary Classification)

## Dataset
1. **Generate Synthetic Data (Pandas/Numpy)**: Create three pandas DataFrames:
    *   `customers_df`: With 500-700 rows. Columns: `customer_id` (unique integers), `signup_date` (random dates over the last 3 years), `region` (e.g., 'North', 'South', 'East', 'West'), `loyalty_status` (e.g., 'Bronze', 'Silver', 'Gold').
    *   `products_df`: With 100-150 rows. Columns: `product_id` (unique integers), `category` (e.g., 'Electronics', 'Books', 'Apparel', 'Home Goods'), `unit_price` (random floats 20.0-1500.0), `brand` (e.g., 'BrandA', 'BrandB', 'BrandC').
    *   `transactions_df`: With 10000-15000 rows. Columns: `transaction_id` (unique integers), `customer_id` (randomly sampled from `customers_df` IDs), `product_id` (randomly sampled from `products_df` IDs), `purchase_date` (random dates occurring *after* their respective `signup_date`), `quantity` (random integers 1-3), `is_returned` (binary: 0 or 1).
    *   **Simulate realistic patterns**: Ensure `purchase_date` is always after `signup_date`. The overall return rate should be 10-15%. Bias `is_returned` such that:
        *   Products with higher `unit_price` or from specific `category`s (e.g., 'Apparel') or `brand`s might have higher return rates.
        *   Newer customers (less time since `signup_date`) or those with 'Bronze' `loyalty_status` might have slightly higher return rates.
        *   `quantity` could have a minor inverse relationship with `is_returned` (buying more might indicate higher commitment, or perhaps buying less means less confidence).
    *   Sort `transactions_df` by `customer_id` then `purchase_date`.

2. **Load into SQLite & SQL Feature Engineering (Transaction-Level Aggregates)**: Create an in-memory SQLite database using `sqlite3`. Load `customers_df`, `products_df`, and `transactions_df` into tables named `customers`, `products`, and `transactions` respectively.
    Write a single SQL query that performs the following for *each transaction*:
    *   **Joins** `transactions` with `customers` and `products` tables.
    *   **Calculates transaction-level features**:
        *   `transaction_value`: `quantity` * `unit_price`.
        *   `days_since_signup_at_purchase`: Number of days between `customers.signup_date` and `transactions.purchase_date`.
    *   **Includes static product, customer, and transaction attributes**: `transaction_id`, `customer_id`, `product_id`, `purchase_date`, `quantity`, `is_returned` (the target), `region`, `loyalty_status`, `category`, `unit_price`, `brand`.
    *   The query should return all these attributes and engineered features.

3. **Pandas Feature Engineering & Binary Target Creation**: Fetch the SQL query results into a pandas DataFrame (`transaction_features_df`).
    *   Handle `NaN` values: Ensure all numerical features derived from SQL (especially `days_since_signup_at_purchase`) are appropriately filled (e.g., with 0 or mean, or consider imputation strategy). If no NaNs from calculation, skip.
    *   Convert `signup_date` and `purchase_date` to datetime objects (if not already handled correctly by SQLite).
    *   Define features `X` (all numerical: `quantity`, `unit_price`, `transaction_value`, `days_since_signup_at_purchase`; categorical: `region`, `loyalty_status`, `category`, `brand`) and target `y` (`is_returned`). Split into training and testing sets (e.g., 70/30 split) using `sklearn.model_selection.train_test_split` (set `random_state=42`, `stratify` on `y` for class balance due to return rate).

4. **Data Visualization**: Create two separate plots to visually inspect relationships with `is_returned`:
    *   A violin plot (or box plot) showing the distribution of `unit_price` for non-returned (0) vs. returned (1) items. Ensure appropriate labels and titles.
    *   A stacked bar chart showing the proportion of `is_returned` (0 or 1) across different `category` values. Ensure appropriate labels and titles.

5. **ML Pipeline & Evaluation (Binary Classification)**: 
    *   Create an `sklearn.pipeline.Pipeline` with a `sklearn.compose.ColumnTransformer` for preprocessing:
        *   For numerical features: Apply `sklearn.preprocessing.SimpleImputer(strategy='mean')` followed by `sklearn.preprocessing.StandardScaler`.
        *   For categorical features: Apply `sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')`.
    *   The final estimator in the pipeline should be `sklearn.ensemble.HistGradientBoostingClassifier` (set `random_state=42`).
    *   Train the pipeline on `X_train`, `y_train`. Predict probabilities for the positive class (class 1) on the test set (`X_test`).
    *   Calculate and print the `sklearn.metrics.roc_auc_score` and a `sklearn.metrics.classification_report` for the test set predictions.

## Hint
When generating synthetic data for `is_returned`, use `np.random.rand()` and apply conditions to bias the probabilities. For example, `is_returned = ((df['unit_price'] > 500) & (np.random.rand(len(df)) < 0.25)) | (df['category'] == 'Apparel' & (np.random.rand(len(df)) < 0.20))` and then combine with a general low probability for other cases to achieve the desired overall return rate.
