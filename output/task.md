# AI Daily Lab — 2026-05-02

## Task
Develop a machine learning pipeline to predict if a user will make a purchase (convert) within 7 days of being exposed to a specific A/B test variant.

## Focus
Predicting user conversion within an A/B test variant exposure, integrating SQL for time-windowed event aggregation and Pandas for advanced feature engineering.

## Dataset
1. **Generate Synthetic Data (Pandas/Numpy)**: Create three pandas DataFrames:
    *   `users_df`: With 1000-1500 rows. Columns: `user_id` (unique integers), `signup_date` (random dates over the last 3 years), `country` (e.g., 'USA', 'Canada', 'UK', 'Germany'), `device_type` (e.g., 'Mobile', 'Desktop', 'Tablet'), `age_group` (e.g., '18-24', '25-34', '35-44', '45+').
    *   `ab_test_assignments_df`: With 2000-3000 rows (users can be in multiple tests over time, but only one variant per test). Columns: `assignment_id` (unique integers), `user_id` (randomly sampled from `users_df` IDs), `test_name` (e.g., 'HomepageRedesign', 'CheckoutFlow', 'SearchAlgorithm'), `variant_name` (e.g., 'Control', 'VariantA', 'VariantB'), `assignment_date` (random dates after `signup_date`).
    *   `user_events_df`: With 30000-50000 rows. Columns: `event_id` (unique integers), `user_id` (randomly sampled from `users_df` IDs), `event_timestamp` (random timestamps *after* their respective `signup_date`), `event_type` (e.g., 'page_view', 'click', 'add_to_cart', 'purchase'), `revenue` (random floats 0.0-500.0, >0 for 'purchase' events).
    *   **Simulate realistic patterns**:
        *   Ensure `event_timestamp` is always after `signup_date`.
        *   Each user in `ab_test_assignments_df` is exposed to a variant starting on `assignment_date`.
        *   Introduce a base conversion rate (e.g., 5-10% of users make a purchase). Purchases should typically occur within 7 days of `assignment_date` for assigned users.
        *   Some variants (e.g., 'VariantA' for 'HomepageRedesign') should have a slightly higher conversion rate (e.g., +2-5% increase) compared to their 'Control'. Other variants might perform worse.
        *   'Desktop' users or specific `country` values might exhibit higher engagement/conversion.
        *   `revenue` should be 0 for non-'purchase' events.
    *   Sort `user_events_df` by `user_id` then `event_timestamp`.

2. **Load into SQLite & SQL Feature Engineering (User-Variant-Level Event Aggregations)**: Create an in-memory SQLite database using `sqlite3`. Load `users_df`, `ab_test_assignments_df`, and `user_events_df` into tables named `users`, `assignments`, and `events` respectively.
    Write a single SQL query that performs the following for *each unique A/B test assignment* (i.e., each row in `assignments`):
    *   **Joins** `assignments` with `users` and an aggregated subquery for `events`.
    *   **Aggregates features based on user event activities *within the first 7 days* after the `assignment_date`**:
        *   `assignment_id`, `user_id`, `test_name`, `variant_name`, `assignment_date`.
        *   `country`, `device_type`, `age_group` (from `users`).
        *   `num_page_views_7d` (count of 'page_view' events).
        *   `num_clicks_7d` (count of 'click' events).
        *   `num_add_to_carts_7d` (count of 'add_to_cart' events).
        *   `total_purchases_7d` (count of 'purchase' events).
        *   `total_revenue_7d` (sum of `revenue` for all events).
        *   `total_events_7d` (total count of all events).
    *   **Ensures** all assignments are included (using `LEFT JOIN`), showing 0 for counts/sums if no activity in the 7-day window.
    *   The query should return `assignment_id`, `user_id`, `test_name`, `variant_name`, `assignment_date`, `country`, `device_type`, `age_group`, and all aggregated features.
    *   **Hint**: Use `julianday()` for precise date comparisons to define the 7-day window. Use `SUM(CASE WHEN ... END)` for conditional counts and sums.

3. **Pandas Feature Engineering & Binary Target Creation**: Fetch the SQL query results into a pandas DataFrame (`ab_features_df`).
    *   Handle `NaN` values: Fill numerical aggregated features (e.g., `num_page_views_7d`, `total_revenue_7d`) with 0 or 0.0 as appropriate.
    *   Convert `signup_date` (from `users_df`, merge if not already present) and `assignment_date` to datetime objects.
    *   Calculate `days_since_signup_at_assignment`: Number of days between `signup_date` and `assignment_date`.
    *   Calculate `click_through_rate_7d`: `num_clicks_7d` / (`num_page_views_7d` + 1e-6). Fill `NaN` or `inf` with 0.
    *   Calculate `add_to_cart_rate_7d`: `num_add_to_carts_7d` / (`num_page_views_7d` + 1e-6). Fill `NaN` or `inf` with 0.
    *   Calculate `revenue_per_event_7d`: `total_revenue_7d` / (`total_events_7d` + 1e-6). Fill `NaN` or `inf` with 0.
    *   **Create the Binary Target `is_purchased_7d`**: A user-assignment is considered to have converted (1) if `total_purchases_7d` > 0 within the 7-day window; otherwise, it's 0.
    *   Define features `X` (numerical: `num_page_views_7d`, `num_clicks_7d`, `num_add_to_carts_7d`, `total_revenue_7d`, `total_events_7d`, `days_since_signup_at_assignment`, `click_through_rate_7d`, `add_to_cart_rate_7d`, `revenue_per_event_7d`; categorical: `test_name`, `variant_name`, `country`, `device_type`, `age_group`) and target `y` (`is_purchased_7d`). Split into training and testing sets (e.g., 70/30 split) using `sklearn.model_selection.train_test_split` (set `random_state=42`, `stratify` on `y` for class balance).

4. **Data Visualization**: Create two separate plots to visually inspect relationships with `is_purchased_7d`:
    *   A violin plot (or box plot) showing the distribution of `total_revenue_7d` for non-purchasers (0) vs. purchasers (1). Ensure appropriate labels and titles.
    *   A stacked bar chart showing the proportion of `is_purchased_7d` (0 or 1) across different `variant_name` values within a specific `test_name` (e.g., 'HomepageRedesign'). Ensure appropriate labels and titles.

5. **ML Pipeline & Evaluation (Binary Classification)**: 
    *   Create an `sklearn.pipeline.Pipeline` with a `sklearn.compose.ColumnTransformer` for preprocessing:
        *   For numerical features: Apply `sklearn.preprocessing.SimpleImputer(strategy='mean')` followed by `sklearn.preprocessing.StandardScaler`.
        *   For categorical features: Apply `sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')`.
    *   The final estimator in the pipeline should be `sklearn.ensemble.HistGradientBoostingClassifier` (set `random_state=42`).
    *   Train the pipeline on `X_train`, `y_train`. Predict probabilities for the positive class (class 1) on the test set (`X_test`).
    *   Calculate and print the `sklearn.metrics.roc_auc_score` and a `sklearn.metrics.classification_report` for the test set predictions.

## Hint
When generating synthetic data, ensure a clear distinction in conversion rates between control and treatment variants for at least one A/B test to make the prediction task meaningful. In SQL, accurately filter events to be *after* the `assignment_date` and *within 7 days* of it for each user's assignment. For the stacked bar chart, you might want to filter for a single `test_name` to make the comparison clearer between its variants.
