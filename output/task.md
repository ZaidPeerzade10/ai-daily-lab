# AI Daily Lab — 2026-04-07

## Task
Develop a machine learning pipeline to predict the likelihood of a new user making their *first purchase* within 30 days of signup, based on their initial 7 days of browsing activity and profile attributes.

## Focus
User activation, binary classification, time-based feature engineering (early behavior vs. future target), synthetic data generation with specific temporal and behavioral biases, SQL analytics for aggregation, ML pipeline for prediction.

## Dataset
1. **Generate Synthetic Data (Pandas/Numpy)**: Create three pandas DataFrames:
    *   `users_df`: With 500-700 rows. Columns: `user_id` (unique integers), `signup_date` (random dates over the last 3 years), `region` (e.g., 'North', 'South', 'East', 'West'), `referral_source` (e.g., 'Organic', 'Paid Search', 'Social Media', 'Referral'), `age_group` (categorical: '18-24', '25-34', '35-49', '50+').
    *   `browsing_events_df`: With 15000-25000 rows. Columns: `event_id` (unique integers), `user_id` (randomly sampled from `users_df` IDs), `event_timestamp` (random timestamps occurring *after* their respective `signup_date`), `event_type` (e.g., 'view_product', 'add_to_cart', 'view_category', 'search', 'homepage_visit'), `product_id` (randomly sampled if `event_type` is 'view_product' or 'add_to_cart', else `None`).
    *   `purchases_df`: With 2000-3000 rows. Columns: `purchase_id` (unique integers), `user_id` (randomly sampled from `users_df` IDs), `purchase_date` (random dates occurring *after* their respective `signup_date`), `total_amount` (random floats 10.0-500.0).
    *   **Simulate realistic patterns**: Ensure `event_timestamp` and `purchase_date` are always after `signup_date`. Bias data such that:
        *   Users from certain `referral_source`s (e.g., 'Paid Search', 'Referral') or `age_group`s (e.g., '25-34') might have a higher likelihood of making a purchase.
        *   Users with higher counts of 'view_product' or 'add_to_cart' events within their first 7 days are more likely to make a purchase within 30 days.
        *   Overall first-time purchase rate within 30 days should be around 20-30%.
    *   Sort `browsing_events_df` by `user_id` then `event_timestamp` and `purchases_df` by `user_id` then `purchase_date`.

2. **Load into SQLite & SQL Feature Engineering (Early User Browsing)**: Create an in-memory SQLite database using `sqlite3`. Load `users_df`, `browsing_events_df`, and `purchases_df` into tables named `users`, `browsing_events`, and `purchases` respectively. For each user, define their `early_behavior_cutoff_date` as `signup_date + 7 days`.
    Write a single SQL query that performs the following for *each user*, aggregating their browsing behavior *within their first 7 days post-signup* (i.e., `event_timestamp` before or on `early_behavior_cutoff_date`):
    *   **Joins** `users` with an aggregated subquery for `browsing_events`.
    *   **Aggregates features based on activities *within the first 7 days* post-signup**:
        *   `num_events_first_7d` (count of `event_id`s)
        *   `num_product_views_first_7d` (count of `event_type = 'view_product'`)
        *   `num_add_to_cart_first_7d` (count of `event_type = 'add_to_cart'`)
        *   `num_searches_first_7d` (count of `event_type = 'search'`)
        *   `days_with_activity_first_7d` (count of distinct dates from `event_timestamp`)
        *   `avg_time_between_events_first_7d` (average difference in seconds between consecutive events for a user within the 7-day window. If only 0 or 1 event, default to 0.0 or `NULL`).
    *   **Includes static user attributes**: `user_id`, `signup_date`, `region`, `referral_source`, `age_group`.
    *   **Ensures** all users are included (using `LEFT JOIN` to the aggregated subquery), showing 0 for counts/sums and `NULL` or 0.0 for averages if no activity or insufficient activity in the first 7 days.
    *   The query should return `user_id`, `signup_date`, `region`, `referral_source`, `age_group`, and all the aggregated features.
    *   **Hint**: Use `julianday()` for date comparisons. Aggregate event types using `SUM(CASE WHEN event_type = '...' THEN 1 ELSE 0 END)`. Use `strftime('%Y-%m-%d', event_timestamp)` for distinct dates, and `DATE(u.signup_date, '+7 days')` for the cutoff. For `avg_time_between_events_first_7d`, you might need a CTE/subquery with `LAG` and then aggregate. If too complex for 45 min, you can simplify to `avg_session_duration_first_7d` if you add session info, or just skip this one and focus on counts.

3. **Pandas Feature Engineering & Binary Target Creation**: Fetch the SQL query results into a pandas DataFrame (`user_early_features_df`).
    *   Handle `NaN` values: Fill numerical aggregated features (e.g., `num_events_first_7d`, `num_product_views_first_7d`, etc.) with 0 or 0.0 as appropriate.
    *   Convert `signup_date` to datetime objects.
    *   Calculate `days_since_signup_at_cutoff`: The number of days between `signup_date` and `early_behavior_cutoff_date` (which is always 7 days).
    *   Calculate `engagement_rate_first_7d`: (`num_product_views_first_7d` + `num_add_to_cart_first_7d`) / (`num_events_first_7d` + 1). Use `+1` to prevent division by zero. Fill any `NaN`s with 0.
    *   **Create the Binary Target `made_first_purchase_within_30d`**: For *each user*, determine if they made *any* purchase from the *original* `purchases_df` where `purchase_date` is after their `signup_date` AND before or on `signup_date + 30 days`. Merge this aggregate (a binary flag) with `user_early_features_df` (left join), filling `NaN`s with 0 for users with no purchases in the target window.
    *   Define features `X` (all numerical: `num_events_first_7d`, `num_product_views_first_7d`, `num_add_to_cart_first_7d`, `num_searches_first_7d`, `days_with_activity_first_7d`, `days_since_signup_at_cutoff`, `engagement_rate_first_7d`; categorical: `region`, `referral_source`, `age_group`) and target `y` (`made_first_purchase_within_30d`). Split into training and testing sets (e.g., 70/30 split) using `sklearn.model_selection.train_test_split` (set `random_state=42`, `stratify` on `y` for class balance due to purchase rate).

4. **Data Visualization**: Create two separate plots to visually inspect relationships with `made_first_purchase_within_30d`:
    *   A violin plot (or box plot) showing the distribution of `num_add_to_cart_first_7d` for non-purchasers (0) vs. purchasers (1). Ensure appropriate labels and titles.
    *   A stacked bar chart showing the proportion of `made_first_purchase_within_30d` (0 or 1) across different `referral_source` values. Ensure appropriate labels and titles.

5. **ML Pipeline & Evaluation (Binary Classification)**: 
    *   Create an `sklearn.pipeline.Pipeline` with a `sklearn.compose.ColumnTransformer` for preprocessing:
        *   For numerical features: Apply `sklearn.preprocessing.SimpleImputer(strategy='mean')` followed by `sklearn.preprocessing.StandardScaler`.
        *   For categorical features: Apply `sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')`.
    *   The final estimator in the pipeline should be `sklearn.ensemble.HistGradientBoostingClassifier` (set `random_state=42`).
    *   Train the pipeline on `X_train`, `y_train`. Predict probabilities for the positive class (class 1) on the test set (`X_test`).
    *   Calculate and print the `sklearn.metrics.roc_auc_score` and a `sklearn.metrics.classification_report` for the test set predictions.

## Hint
When generating synthetic `event_timestamp` for `browsing_events_df`, ensure they are always after the respective user's `signup_date`. For `purchases_df`, ensure `purchase_date` is also after `signup_date`. For the SQL `avg_time_between_events_first_7d` feature, if `LAG` proves too time-consuming, focus on the other count-based features and simplify or remove it. For the target calculation, `merge` the aggregated purchase status with your early features, and fill `NaN`s for users who didn't purchase with 0.
