# AI Daily Lab — 2026-02-18

## Task
1. **Generate Synthetic Data (Pandas/Numpy)**: Create three pandas DataFrames:
    *   `customers_df`: With 500-700 rows. Columns: `customer_id` (unique integers), `signup_date` (random dates over the last 5 years), `industry` (e.g., 'SaaS', 'E-commerce', 'Fintech'), `subscription_tier` (e.g., 'Bronze', 'Silver', 'Gold').
    *   `usage_logs_df`: With 5000-8000 rows. Columns: `log_id` (unique integers), `customer_id` (randomly sampled from `customers_df` IDs), `log_date` (random dates occurring *after* `signup_date`), `feature_used` (e.g., 'Login', 'Dashboard_View', 'Report_Download', 'Data_Export', 'Billing_Access'), `success_status` (binary, 0=fail, 1=success).
    *   `support_tickets_df`: With 800-1200 rows. Columns: `ticket_id` (unique integers), `customer_id` (randomly sampled from `customers_df` IDs), `ticket_open_date` (random dates occurring *after* `signup_date`), `ticket_category` (e.g., 'Bug', 'Feature_Request', 'Billing', 'Technical_Support', 'Onboarding', 'General_Inquiry'), `ticket_severity` (e.g., 'Low', 'Medium', 'High').
    *   **Simulate realistic patterns**: Ensure `log_date` and `ticket_open_date` are always after `signup_date`. Introduce a correlation: customers who open 'Bug' or 'Technical_Support' tickets should, in the *days leading up to that ticket_open_date*, have a higher proportion of `success_status=0` for related `feature_used` (e.g., 'Report_Download', 'Data_Export', 'Login') in their `usage_logs_df`. Users with higher `subscription_tier` might have fewer tickets, or more 'Feature_Request' tickets.

2. **Load into SQLite & SQL Feature Engineering**: Create an in-memory SQLite database using `sqlite3`. Load `customers_df`, `usage_logs_df`, and `support_tickets_df` into tables named `customers`, `usage_logs`, and `support_tickets` respectively. Determine a `global_analysis_date` (e.g., `max(log_date)` from `usage_logs_df` + 30 days, using pandas) and a `feature_cutoff_date` (`global_analysis_date` - 30 days).
    Write a single SQL query that performs the following for *each customer*, aggregating their usage and support behavior *before* the `feature_cutoff_date`:
    *   **Joins** `customers`, `usage_logs`, and `support_tickets` tables.
    *   **Aggregates features based on activity *before* `feature_cutoff_date`**: 
        *   `total_usage_logs_pre_cutoff` (count of all `log_id`s)
        *   `num_failed_attempts_pre_cutoff` (count of `success_status=0` in `usage_logs`)
        *   `avg_usage_success_rate_pre_cutoff` (average of `success_status`)
        *   `days_since_last_failed_usage_pre_cutoff`: Number of days between `feature_cutoff_date` and `MAX(log_date)` where `success_status = 0`.
        *   `num_prior_tickets_pre_cutoff` (count of `ticket_id`s)
        *   `num_high_severity_tickets_pre_cutoff` (count of `ticket_id`s where `ticket_severity = 'High'`)
        *   `days_since_last_ticket_pre_cutoff`: Number of days between `feature_cutoff_date` and `MAX(ticket_open_date)`.
    *   **Includes static customer attributes**: `customer_id`, `industry`, `subscription_tier`, `signup_date`.
    *   **Ensures** all customers are included (using a `LEFT JOIN`), showing 0 for counts/sums, 0.0 for averages, and `NULL` for `days_since_last_failed_usage_pre_cutoff`/`days_since_last_ticket_pre_cutoff` if no relevant activity before cutoff.
    *   The query should return `customer_id`, `industry`, `subscription_tier`, `signup_date`, and all the aggregated features.

3. **Pandas Feature Engineering & Multi-Class Target Creation**: Fetch the SQL query results into a pandas DataFrame (`customer_features_df`).
    *   Handle `NaN` values: Fill `total_usage_logs_pre_cutoff`, `num_failed_attempts_pre_cutoff`, `num_prior_tickets_pre_cutoff`, `num_high_severity_tickets_pre_cutoff` with 0. Fill `avg_usage_success_rate_pre_cutoff` with 1.0 (assuming perfect success if no logs). For `days_since_last_failed_usage_pre_cutoff` and `days_since_last_ticket_pre_cutoff` (for customers with no relevant activity), fill with a large sentinel value (e.g., 9999 days).
    *   Convert `signup_date` to datetime objects. Calculate `account_age_at_cutoff_days`: The number of days between `signup_date` and the `feature_cutoff_date`.
    *   **Create the Multi-Class Target `main_pain_point_category`**: For each customer, determine the *most frequent `ticket_category`* from `support_tickets_df` for tickets opened *between `feature_cutoff_date` and `global_analysis_date`*. If a customer has *no tickets* in this future period, assign them a special category, e.g., 'No_Future_Tickets'. This will require aggregating future ticket data separately and merging.
    *   Define features `X` (all numerical and categorical features engineered) and target `y` (`main_pain_point_category`). Split into training and testing sets (e.g., 70/30 split) using `sklearn.model_selection.train_test_split` (set `random_state=42`, `stratify` on `y` for class balance, as some categories might be less frequent).

4. **Data Visualization**: Create two separate plots to visually inspect relationships with `main_pain_point_category`:
    *   A violin plot (or box plot) showing the distribution of `avg_usage_success_rate_pre_cutoff` for each `main_pain_point_category`.
    *   A stacked bar chart showing the distribution of `main_pain_point_category` across different `subscription_tier` values.
    Ensure plots have appropriate labels and titles.

5. **ML Pipeline & Evaluation (Multi-Class)**: 
    *   Create an `sklearn.pipeline.Pipeline` with a `sklearn.compose.ColumnTransformer` for preprocessing:
        *   For numerical features (e.g., `account_age_at_cutoff_days`, `total_usage_logs_pre_cutoff`, `num_failed_attempts_pre_cutoff`, `avg_usage_success_rate_pre_cutoff`, `days_since_last_failed_usage_pre_cutoff`, `num_prior_tickets_pre_cutoff`, `num_high_severity_tickets_pre_cutoff`, `days_since_last_ticket_pre_cutoff`): Apply `sklearn.preprocessing.SimpleImputer(strategy='mean')` followed by `sklearn.preprocessing.StandardScaler`.
        *   For categorical features (`industry`, `subscription_tier`): Apply `sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')`.
    *   The final estimator in the pipeline should be `sklearn.ensemble.RandomForestClassifier` (set `random_state=42`, `n_estimators=100`, `class_weight='balanced'` for potential class imbalance).
    *   Train the pipeline on `X_train`, `y_train`. Predict `main_pain_point_category` for `X_test`.
    *   Calculate and print the `sklearn.metrics.accuracy_score` and a `sklearn.metrics.classification_report` for the test set predictions.

## Focus
Predicting a customer's primary future support pain point category (multi-class classification) based on their historical product usage logs and past support ticket interactions.

## Dataset
Synthetic customer activity data: `customers_df` (user demographics), `usage_logs_df` (product feature usage with success/failure status), `support_tickets_df` (historical support requests).

## Hint
For SQL step 2, utilize `LEFT JOIN` on `customers` to both `usage_logs` and `support_tickets` subqueries (filtered by `feature_cutoff_date`) to ensure all customers are included. For Pandas step 3, calculate the target by grouping future `support_tickets_df` by `customer_id` and finding the `idxmax()` of `value_counts()` for `ticket_category` for each customer, then merge back. Remember to handle customers with no future tickets.
