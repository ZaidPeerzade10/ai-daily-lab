# AI Daily Lab — 2026-03-15

## Task
1. **Generate Synthetic Data (Pandas/Numpy)**: Create three pandas DataFrames:
    *   `customers_df`: With 500-700 rows. Columns: `customer_id` (unique integers), `signup_date` (random dates over the last 5 years), `acquisition_channel` (e.g., 'Organic', 'Paid_Social', 'Referral', 'Email'), `demographic_segment` (e.g., 'Young_Adult', 'Middle_Age', 'Senior').
    *   `purchases_df`: With 5000-8000 rows. Columns: `purchase_id` (unique integers), `customer_id` (randomly sampled from `customers_df` IDs), `purchase_date` (random dates occurring *after* their respective `signup_date`), `amount` (random floats 10.0-1500.0), `product_category` (e.g., 'Electronics', 'Books', 'Groceries', 'Apparel', 'Services').
    *   `browsing_df`: With 10000-15000 rows. Columns: `browse_id` (unique integers), `customer_id` (randomly sampled from `customers_df` IDs), `browse_date` (random timestamps occurring *after* their respective `signup_date`), `page_view_type` (e.g., 'Product_Page', 'Category_Page', 'Homepage', 'Checkout_Page', 'Help_Page'), `time_on_page_seconds` (random integers 5-300).
    *   **Simulate realistic patterns**: Ensure `purchase_date` and `browse_date` are always after `signup_date`. Bias data such that customers acquired via 'Paid_Social' might have higher initial browsing activity but lower conversion rates. 'Referral' customers might have higher `amount`s. Some customers should show patterns indicative of higher future lifetime value (more purchases, higher average `amount`, more browsing on 'Product_Page'/'Checkout_Page').
    *   Sort `purchases_df` and `browsing_df` by `customer_id` then `date` for easier sequential processing.

2. **Load into SQLite & SQL Feature Engineering (Early Customer Behavior)**: Create an in-memory SQLite database using `sqlite3`. Load `customers_df`, `purchases_df`, and `browsing_df` into tables named `customers`, `purchases`, and `browsing` respectively. For each customer, define their `early_window_cutoff_date` as `signup_date + 60 days`.
    Write a single SQL query that performs the following for *each customer*, aggregating their purchase and browsing behavior *within their first 60 days post-signup* (i.e., before or on `early_window_cutoff_date`):
    *   **Joins** `customers` with aggregated subqueries for `purchases` and `browsing` tables.
    *   **Aggregates features based on activities *within the first 60 days* post-signup**:
        *   `num_purchases_first_60d` (count of `purchase_id`s)
        *   `total_spend_first_60d` (sum of `amount`)
        *   `avg_purchase_amount_first_60d` (average `amount`)
        *   `num_browsing_events_first_60d` (count of `browse_id`s)
        *   `total_browse_duration_first_60d` (sum of `time_on_page_seconds`)
        *   `num_unique_product_categories_first_60d` (count of distinct `product_category` from purchases)
        *   `has_browsed_checkout_page_first_60d` (binary: 1 if 'Checkout_Page' `page_view_type` exists, else 0)
        *   `days_since_first_purchase_first_60d`: Number of days between `signup_date` and `MIN(purchase_date)` for the customer's first purchase (if within the 60-day window).
    *   **Includes static customer attributes**: `customer_id`, `signup_date`, `acquisition_channel`, `demographic_segment`.
    *   **Ensures** all customers are included (using `LEFT JOIN`s to aggregated subqueries), showing 0 for counts/sums/binary flags, 0.0 for averages, and `NULL` for `days_since_first_purchase_first_60d` if no purchases in the first 60 days.
    *   The query should return `customer_id`, `signup_date`, `acquisition_channel`, `demographic_segment`, and all the aggregated features.
    *   **Hint**: Use `julianday()` for date differences. Filter activities based on `date BETWEEN c.signup_date AND DATE(c.signup_date, '+60 days')`.

3. **Pandas Feature Engineering & Multi-Class Target Creation (Future LTV Tier)**: Fetch the SQL query results into a pandas DataFrame (`customer_features_df`).
    *   Handle `NaN` values: Fill `num_purchases_first_60d`, `total_spend_first_60d`, `num_browsing_events_first_60d`, `total_browse_duration_first_60d`, `num_unique_product_categories_first_60d`, `has_browsed_checkout_page_first_60d` with 0. Fill `avg_purchase_amount_first_60d` with 0.0. For `days_since_first_purchase_first_60d` (for customers with no purchases in the first 60 days), fill with 60 (representing purchase started on day 60, or no purchase).
    *   Convert `signup_date` to datetime objects. Calculate `account_age_at_cutoff_days`: The number of days between `signup_date` and `early_window_cutoff_date` (which is always 60 days).
    *   Calculate `purchase_frequency_first_60d`: `num_purchases_first_60d` / 60.0. Fill any `NaN`s with 0.
    *   Calculate `browse_to_purchase_ratio_first_60d`: `num_browsing_events_first_60d` / (`num_purchases_first_60d` + 1). Use `+1` to prevent division by zero.
    *   **Create the Multi-Class Target `future_ltv_tier`**: Calculate `total_future_spend` (sum of `amount`) for each user from the *original* `purchases_df` for purchases occurring *after* their `signup_date + 60 days`. Merge this aggregate with `customer_features_df` (left join), filling `NaN`s with 0.
        *   Calculate the 33rd and 66th percentiles for *non-zero* `total_future_spend`.
        *   Define segments:
            *   'Low_LTV': `total_future_spend` == 0.
            *   'Medium_LTV': `total_future_spend` > 0 AND `total_future_spend` <= 33rd percentile.
            *   'High_LTV': `total_future_spend` > 33rd percentile AND `total_future_spend` <= 66th percentile.
            *   'Very_High_LTV': `total_future_spend` > 66th percentile.
    *   Define features `X` (all numerical: `total_spend_first_60d`, `num_purchases_first_60d`, `avg_purchase_amount_first_60d`, `num_browsing_events_first_60d`, `total_browse_duration_first_60d`, `num_unique_product_categories_first_60d`, `days_since_first_purchase_first_60d`, `account_age_at_cutoff_days`, `purchase_frequency_first_60d`, `browse_to_purchase_ratio_first_60d`; categorical: `acquisition_channel`, `demographic_segment`, `has_browsed_checkout_page_first_60d`) and target `y` (`future_ltv_tier`). Split into training and testing sets (e.g., 70/30 split) using `sklearn.model_selection.train_test_split` (set `random_state=42`, `stratify` on `y` for class balance).

4. **Data Visualization**: Create two separate plots to visually inspect relationships with `future_ltv_tier`:
    *   A violin plot (or box plot) showing the distribution of `total_spend_first_60d` for each `future_ltv_tier`. Ensure appropriate labels and titles.
    *   A stacked bar chart showing the proportion of `future_ltv_tier` across different `acquisition_channel` values. Ensure appropriate labels and titles.

5. **ML Pipeline & Evaluation (Multi-Class)**: 
    *   Create an `sklearn.pipeline.Pipeline` with a `sklearn.compose.ColumnTransformer` for preprocessing:
        *   For numerical features: Apply `sklearn.preprocessing.SimpleImputer(strategy='mean')` followed by `sklearn.preprocessing.StandardScaler`.
        *   For categorical features: Apply `sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')`.
    *   The final estimator in the pipeline should be `sklearn.ensemble.HistGradientBoostingClassifier` (set `random_state=42`).
    *   Train the pipeline on `X_train`, `y_train`. Predict `future_ltv_tier` for `X_test`.
    *   Calculate and print the `sklearn.metrics.accuracy_score` and a `sklearn.metrics.classification_report` for the test set predictions.

## Focus
Predicting multi-class customer lifetime value (LTV) tiers using early purchase and browsing behavior via SQL-driven feature engineering and an ML pipeline.

## Dataset
Synthetic customer, purchase, and browsing activity dataframes.

## Hint
When creating the `future_ltv_tier` target, carefully handle cases where `total_future_spend` is 0 before calculating percentiles. For date calculations in SQL, `julianday()` provides a convenient way to get a numeric representation of dates for differences. Remember to explicitly convert relevant columns to datetime objects in pandas before performing date calculations.
