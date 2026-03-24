# AI Daily Lab — 2026-03-24

## Task
Develop a machine learning pipeline to predict whether a customer will exhibit negative sentiment in future feedback, based on their initial interaction patterns and profile.

## Focus
Predicting future customer sentiment (binary classification) using early behavioral features derived from SQL aggregations and an end-to-end ML pipeline.

## Dataset
1. **Generate Synthetic Data (Pandas/Numpy)**: Create three pandas DataFrames:
    *   `customers_df`: With 500-700 rows. Columns: `customer_id` (unique integers), `signup_date` (random dates over the last 5 years), `region` (e.g., 'North', 'South', 'East', 'West'), `subscription_tier` (e.g., 'Basic', 'Premium', 'Pro').
    *   `interactions_df`: With 10000-15000 rows. Columns: `interaction_id` (unique integers), `customer_id` (randomly sampled from `customers_df` IDs), `interaction_date` (random dates occurring *after* their respective `signup_date`), `interaction_type` (e.g., 'Support_Call', 'Chat', 'FAQ_Visit', 'Forum_Post'), `duration_minutes` (random integers 5-60), `successful_resolution` (binary, 0 or 1, only relevant for 'Support_Call' or 'Chat' types).
    *   `feedback_df`: With 1500-2500 rows. Columns: `feedback_id` (unique integers), `customer_id` (randomly sampled from `customers_df` IDs), `feedback_date` (random dates occurring *after* their respective `signup_date`), `sentiment_score` (random integers 1-5).
    *   **Simulate realistic patterns**: Ensure `interaction_date` and `feedback_date` are always after `signup_date`. Bias `successful_resolution` (overall 60-80% success rate) such that 'Premium' tier users have higher success rates. Bias `sentiment_score` such that lower scores (1-2) are more likely for customers with multiple 'Support_Call' or 'Chat' interactions that have `successful_resolution=0` in a recent period. Sort `interactions_df` and `feedback_df` by `customer_id` then `date` for easier sequential processing.

2. **Load into SQLite & SQL Feature Engineering (Early Interaction Behavior)**: Create an in-memory SQLite database using `sqlite3`. Load `customers_df`, `interactions_df`, and `feedback_df` into tables named `customers`, `interactions`, and `feedback` respectively. For each customer, define their `early_window_cutoff_date` as `signup_date + 30 days`.
    Write a single SQL query that performs the following for *each customer*, aggregating their interaction behavior *within their first 30 days post-signup* (i.e., `interaction_date` before or on `early_window_cutoff_date`):
    *   **Joins** `customers` with aggregated subqueries for `interactions`.
    *   **Aggregates features based on activities *within the first 30 days* post-signup**:
        *   `num_interactions_first_30d` (count of `interaction_id`s)
        *   `total_duration_first_30d` (sum of `duration_minutes`)
        *   `avg_duration_first_30d` (average `duration_minutes`)
        *   `num_support_contacts_first_30d` (count of `interaction_id`s where `interaction_type` is 'Support_Call' or 'Chat')
        *   `num_failed_resolutions_first_30d` (count of `interaction_id`s where `interaction_type` is 'Support_Call' or 'Chat' AND `successful_resolution = 0`)
        *   `days_since_first_interaction_first_30d`: Number of days between `signup_date` and `MIN(interaction_date)` for the customer's first interaction (if within the 30-day window).
    *   **Includes static customer attributes**: `customer_id`, `signup_date`, `region`, `subscription_tier`.
    *   **Ensures** all customers are included (using `LEFT JOIN`s to aggregated subqueries), showing 0 for counts/sums, 0.0 for averages, and `NULL` for `days_since_first_interaction_first_30d` if no interactions in the first 30 days.
    *   The query should return `customer_id`, `signup_date`, `region`, `subscription_tier`, and all the aggregated features.
    *   **Hint**: Use `julianday()` for date differences. Filter interactions based on `i.interaction_date BETWEEN c.signup_date AND DATE(c.signup_date, '+30 days')`.

3. **Pandas Feature Engineering & Binary Target Creation (Future Negative Sentiment)**: Fetch the SQL query results into a pandas DataFrame (`customer_initial_features_df`).
    *   Handle `NaN` values: Fill `num_interactions_first_30d`, `total_duration_first_30d`, `num_support_contacts_first_30d`, `num_failed_resolutions_first_30d` with 0. Fill `avg_duration_first_30d` with 0.0. For `days_since_first_interaction_first_30d` (for customers with no interactions in the first 30 days), fill with 30 (representing activity started on day 30, or no activity).
    *   Convert `signup_date` to datetime objects.
    *   Calculate `support_contact_rate_first_30d`: `num_support_contacts_first_30d` / (`num_interactions_first_30d` if `num_interactions_first_30d` > 0 else 1.0).
    *   Calculate `failed_resolution_rate_first_30d`: `num_failed_resolutions_first_30d` / (`num_support_contacts_first_30d` if `num_support_contacts_first_30d` > 0 else 1.0).
    *   **Create the Binary Target `is_negative_future_sentiment`**: For each user, define their `future_sentiment_window_start_date` as `signup_date + 60 days`. Check the *original* `feedback_df` for any feedback where `sentiment_score` is 1 or 2 (representing negative sentiment) AND `feedback_date` occurs *after* `future_sentiment_window_start_date`. Assign `1` if at least one such negative feedback exists, otherwise `0`. Perform a left merge, filling `NaN`s from the merge with 0.
    *   Define features `X` (all numerical: `num_interactions_first_30d`, `total_duration_first_30d`, `avg_duration_first_30d`, `num_support_contacts_first_30d`, `num_failed_resolutions_first_30d`, `days_since_first_interaction_first_30d`, `support_contact_rate_first_30d`, `failed_resolution_rate_first_30d`; categorical: `region`, `subscription_tier`) and target `y` (`is_negative_future_sentiment`). Split into training and testing sets (e.g., 70/30 split) using `sklearn.model_selection.train_test_split` (set `random_state=42`, `stratify` on `y` for class balance).

4. **Data Visualization**: Create two separate plots to visually inspect relationships with `is_negative_future_sentiment`:
    *   A violin plot (or box plot) showing the distribution of `avg_duration_first_30d` for non-negative (0) vs. negative (1) future sentiment users. Ensure appropriate labels and titles.
    *   A stacked bar chart showing the proportion of `is_negative_future_sentiment` (0 or 1) across different `subscription_tier` values. Ensure appropriate labels and titles.

5. **ML Pipeline & Evaluation (Binary Classification)**: 
    *   Create an `sklearn.pipeline.Pipeline` with a `sklearn.compose.ColumnTransformer` for preprocessing:
        *   For numerical features: Apply `sklearn.preprocessing.SimpleImputer(strategy='mean')` followed by `sklearn.preprocessing.StandardScaler`.
        *   For categorical features: Apply `sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')`.
    *   The final estimator in the pipeline should be `sklearn.ensemble.HistGradientBoostingClassifier` (set `random_state=42`).
    *   Train the pipeline on `X_train`, `y_train`. Predict probabilities for the positive class (class 1) on the test set (`X_test`).
    *   Calculate and print the `sklearn.metrics.roc_auc_score` and a `sklearn.metrics.classification_report` for the test set predictions.

## Hint
Pay close attention to date filtering and handling `NULL` or 0 values from SQL aggregations and Pandas merges, especially when creating ratios. The target definition requires careful merging and filtering on the original `feedback_df` based on dates relative to `signup_date`.
