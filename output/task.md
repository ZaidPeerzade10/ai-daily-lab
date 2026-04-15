# AI Daily Lab — 2026-04-15

## Task
Develop a machine learning pipeline to predict whether a user will respond (click or purchase) to a *specific upcoming marketing campaign*, based on their historical engagement with prior campaigns and their static profile attributes.

## Focus
Predicting user response to a future marketing campaign, leveraging historical multi-campaign interaction data through SQL aggregation, and building a binary classification ML pipeline.

## Dataset
Simulate data for users, multiple marketing campaigns, and user interactions with these campaigns over time.

## Hint
1. **Generate Synthetic Data (Pandas/Numpy)**: Create three pandas DataFrames:
    *   `users_df`: With 1000-1500 rows. Columns: `user_id` (unique integers), `signup_date` (random dates over the last 3 years), `country` (e.g., 'US', 'CA', 'UK', 'DE'), `age_group` (categorical: '18-24', '25-34', '35-49', '50+'), `premium_subscriber` (binary: 0 or 1).
    *   `campaigns_df`: With 15-25 rows. Columns: `campaign_id` (unique integers), `campaign_name`, `campaign_type` (e.g., 'Email', 'In-App', 'Push Notification'), `launch_date` (random dates over the last 1.5 years, ensuring a range of dates).
    *   `campaign_events_df`: With 30000-50000 rows. Columns: `event_id` (unique integers), `user_id` (randomly sampled from `users_df` IDs), `campaign_id` (randomly sampled from `campaigns_df` IDs), `event_timestamp` (random timestamps), `event_type` (e.g., 'impression', 'open', 'click', 'purchase'), `value` (random floats 0.0-100.0, primarily for 'purchase' events).
    *   **Simulate realistic patterns**: Ensure `event_timestamp` is always after the user's `signup_date` AND after the `campaign_id`'s `launch_date`. Bias data such that: `premium_subscriber` users have higher `open`/`click`/`purchase` rates. 'Click' events should usually follow 'open' events, and 'open' events follow 'impression'. Ensure some users interact with multiple campaigns and some don't respond to any. Designate one recent campaign from `campaigns_df` as the `TARGET_CAMPAIGN` for prediction (e.g., the one with the latest `launch_date`).
    *   Sort `campaign_events_df` by `user_id`, then `event_timestamp`.

2. **Load into SQLite & SQL Feature Engineering (Pre-Campaign Behavior)**: Create an in-memory SQLite database. Load `users_df`, `campaigns_df`, and `campaign_events_df` into tables `users`, `campaigns`, and `events` respectively. Select the `TARGET_CAMPAIGN_ID` and its `launch_date` from `campaigns_df`.
    Write a single SQL query that performs the following for *each user*:
    *   **Joins** `users` with an aggregated subquery from `events` (filtered to include only events *before* the `TARGET_CAMPAIGN`'s `launch_date`).
    *   **Aggregates features based on activities *prior to the TARGET_CAMPAIGN launch date***:
        *   `num_prior_impressions` (count of 'impression' `event_type`)
        *   `num_prior_opens` (count of 'open' `event_type`)
        *   `num_prior_clicks` (count of 'click' `event_type`)
        *   `total_prior_purchase_value` (sum of `value` where `event_type = 'purchase'`)
        *   `days_since_last_prior_interaction` (calculate `TARGET_CAMPAIGN.launch_date` - `MAX(event_timestamp)` for prior events; handle users with no prior events).
    *   **Includes static user attributes**: `user_id`, `signup_date`, `country`, `age_group`, `premium_subscriber`.
    *   **Ensures** all users are included (using `LEFT JOIN`), showing 0 for counts/sums and appropriate values (e.g., a large number for `days_since_last_prior_interaction`) if no prior activity.
    *   **Hint**: Use `julianday()` for date comparisons. Aggregate event types using `SUM(CASE WHEN event_type = '...' THEN 1 ELSE 0 END)`. Use a CTE for the target campaign's launch date.

3. **Pandas Feature Engineering & Binary Target Creation**: Fetch the SQL results into `user_pre_campaign_features_df`.
    *   Handle `NaN` values: Fill `num_prior_impressions`, `num_prior_opens`, `num_prior_clicks`, `total_prior_purchase_value` with 0. Fill `days_since_last_prior_interaction` with a large number (e.g., 9999) if `NaN` (indicating no prior interaction).
    *   Convert `signup_date` to datetime. Calculate `account_age_at_campaign_launch`: Days between `signup_date` and the `TARGET_CAMPAIGN`'s `launch_date`.
    *   Calculate `prior_open_rate`: `num_prior_opens` / (`num_prior_impressions` + 1). Fill `NaN`/`inf` with 0.
    *   Calculate `prior_click_rate`: `num_prior_clicks` / (`num_prior_opens` + 1). Fill `NaN`/`inf` with 0.
    *   **Create the Binary Target `responded_to_target_campaign`**: For *each user*, determine if they had *any* 'click' or 'purchase' `event_type` related to the `TARGET_CAMPAIGN_ID` within 7 days *after* its `launch_date`. Merge this binary (0/1) target onto `user_pre_campaign_features_df`, filling `NaN` with 0.
    *   Define features `X` (numerical: all aggregated counts/sums, rates, `days_since_last_prior_interaction`, `account_age_at_campaign_launch`; categorical: `country`, `age_group`, `premium_subscriber`) and target `y` (`responded_to_target_campaign`). Split into training and testing sets (e.g., 70/30 split) using `sklearn.model_selection.train_test_split` (set `random_state=42`, `stratify` on `y`).

4. **Data Visualization**: Create two plots to inspect relationships with `responded_to_target_campaign`:
    *   A violin plot (or box plot) showing the distribution of `account_age_at_campaign_launch` for non-responders (0) vs. responders (1). Ensure labels and titles.
    *   A stacked bar chart showing the proportion of `responded_to_target_campaign` across different `age_group` values. Ensure labels and titles.

5. **ML Pipeline & Evaluation (Binary Classification)**:
    *   Create an `sklearn.pipeline.Pipeline` with a `sklearn.compose.ColumnTransformer` for preprocessing:
        *   Numerical features: `sklearn.preprocessing.SimpleImputer(strategy='mean')` followed by `sklearn.preprocessing.StandardScaler`.
        *   Categorical features: `sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')`.
    *   The final estimator should be `sklearn.ensemble.HistGradientBoostingClassifier` (set `random_state=42`).
    *   Train the pipeline on `X_train`, `y_train`. Predict probabilities for the positive class (class 1) on `X_test`.
    *   Calculate and print `sklearn.metrics.roc_auc_score` and a `sklearn.metrics.classification_report` for the test set predictions.
