# AI Daily Lab — 2026-02-07

## Task
1. **Generate Synthetic Data (Pandas/Numpy)**: Create two pandas DataFrames:
    *   `customers_df`: With 500-700 rows. Columns: `customer_id` (unique integers), `signup_date` (random dates over the last 5 years), `initial_plan` (e.g., 'Trial', 'Basic', 'Premium' with random distribution), `age` (random integers 18-70), `acquisition_channel` (e.g., 'Organic', 'Social', 'Referral', 'Paid_Ad').
    *   `payments_df`: With 3000-5000 rows. Columns: `payment_id` (unique integers), `customer_id` (randomly sampled from `customers_df` IDs, ensuring some customers have many payments, some have few, and some have none), `payment_date` (random dates occurring *after* their respective `signup_date`), `amount` (random floats between 10.0 and 150.0, potentially higher for 'Premium' plans).
    *   **Simulate realistic patterns**: Ensure `payment_date` is always after `signup_date`. Generate data such that customers on 'Trial' plans might have zero or few payments, and customers with 'Premium' plans might have higher amounts.

2. **Load into SQLite & SQL Feature Engineering (Early Payment Behavior)**: Create an in-memory SQLite database using `sqlite3`. Load `customers_df` into a table named `customers` and `payments_df` into a table named `payments`. Determine a `global_analysis_date` (e.g., the maximum `payment_date` in `payments_df` + 60 days, using pandas).
    Define an `initial_observation_days` (e.g., 90 days).
    Write a single SQL query that performs the following for *each user*:
    *   **Joins** `customers` and `payments` tables.
    *   **Aggregates initial payment behavior** for payments occurring within the **first `initial_observation_days`** after their `signup_date` (i.e., `payment_date` between `signup_date` and `signup_date + initial_observation_days`):
        *   `initial_num_payments` (count of payments)
        *   `initial_total_revenue` (sum of `amount`)
        *   `initial_avg_payment_value` (average `amount`)
        *   `initial_days_to_first_payment` (number of days between `signup_date` and the `MIN(payment_date)` within the initial window. `NULL` if no initial payment).
    *   **Includes static customer attributes**: `age`, `acquisition_channel`, `initial_plan`, `signup_date`.
    *   **Ensures** all customers are included (using a `LEFT JOIN`), showing 0 for counts/sums and `NULL` for averages/days if no payments in the initial period.
    *   The query should return `customer_id`, `age`, `acquisition_channel`, `initial_plan`, `signup_date`, `initial_num_payments`, `initial_total_revenue`, `initial_avg_payment_value`, `initial_days_to_first_payment`.

3. **Pandas Feature Engineering & Multi-Class Target Creation (CLV Segment)**: Fetch the SQL query results into a pandas DataFrame (`customer_initial_features_df`).
    *   Handle `NaN` values: Fill `initial_num_payments`, `initial_total_revenue`, `initial_avg_payment_value` with 0. For `initial_days_to_first_payment` (for customers with no initial payments), fill with a large sentinel value (e.g., `initial_observation_days` + 30 or 120 days).
    *   Convert `signup_date` to datetime. Calculate `account_age_at_cutoff_days`: The number of days between `signup_date` and the date representing `signup_date + initial_observation_days` (for consistency with the feature window).
    *   **Calculate Total Lifetime Revenue**: From the original `payments_df`, calculate `total_lifetime_revenue` (sum of `amount` for all payments up to `global_analysis_date`) for each customer. Merge this aggregate with `customer_initial_features_df` (left join).
    *   Fill `NaN` in `total_lifetime_revenue` with 0 for customers with no payments ever.
    *   **Create the Multi-Class Target `clv_segment`**: Based on `total_lifetime_revenue` (excluding those with 0 lifetime revenue). First, calculate quantiles (e.g., 33rd and 66th percentiles) for *non-zero* `total_lifetime_revenue`. Then, define segments:
        *   'High_Value': `total_lifetime_revenue` is above the 66th percentile of non-zero revenues.
        *   'Medium_Value': `total_lifetime_revenue` is between the 33rd and 66th percentiles of non-zero revenues.
        *   'Low_Value': `total_lifetime_revenue` is below the 33rd percentile of non-zero revenues *and* greater than 0.
        *   'Churned_No_Revenue': `total_lifetime_revenue` is 0.
    *   Define features `X` (`age`, `acquisition_channel`, `initial_plan`, `account_age_at_cutoff_days`, `initial_num_payments`, `initial_total_revenue`, `initial_avg_payment_value`, `initial_days_to_first_payment`) and target `y` (`clv_segment`). Split into training and testing sets (e.g., 70/30 split) using `sklearn.model_selection.train_test_split` (set `random_state=42`, `stratify` on `y` for class balance).

4. **Data Visualization**: Create two separate plots to visually inspect relationships with `clv_segment`:
    *   A violin plot (or box plot) showing the distribution of `initial_total_revenue` for each `clv_segment`.
    *   A stacked bar chart showing the distribution of `clv_segment` across different `initial_plan` values.
    Ensure plots have appropriate labels and titles.

5. **ML Pipeline & Evaluation (Multi-Class)**: 
    *   Create an `sklearn.pipeline.Pipeline` with a `sklearn.compose.ColumnTransformer` for preprocessing:
        *   For numerical features (`age`, `account_age_at_cutoff_days`, `initial_num_payments`, `initial_total_revenue`, `initial_avg_payment_value`, `initial_days_to_first_payment`): Apply `sklearn.preprocessing.SimpleImputer(strategy='mean')` followed by `sklearn.preprocessing.StandardScaler`.
        *   For categorical features (`acquisition_channel`, `initial_plan`): Apply `sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')`.
    *   The final estimator in the pipeline should be `sklearn.ensemble.GradientBoostingClassifier` (set `random_state=42`, `n_estimators=100`, `learning_rate=0.1`).
    *   Train the pipeline on `X_train`, `y_train`. Predict `clv_segment` for `X_test`.
    *   Calculate and print the `sklearn.metrics.accuracy_score` and a `sklearn.metrics.classification_report` for the test set predictions.

## Focus
Predicting multi-class Customer Lifetime Value (CLV) segments based on early subscription and payment behavior using SQL feature engineering and an ML pipeline.

## Dataset
Synthetic customer subscription and payment data.

## Hint
When defining the `clv_segment`, ensure you calculate quantiles only on customers with non-zero total lifetime revenue before segmenting, and then explicitly handle the zero-revenue (churned) group. Use a `FeatureUnion` or `ColumnTransformer` for feature preprocessing in the ML pipeline to handle different feature types effectively.
