# AI Daily Lab — 2026-02-12

## Task
1. **Generate Synthetic Data (Pandas/Numpy)**: Create four pandas DataFrames:
    *   `users_df`: With 500-700 rows. Columns: `user_id` (unique integers), `signup_date` (random dates over the last 5 years), `age` (random integers 18-70), `region` (e.g., 'North', 'South', 'East', 'West'), `user_segment` (e.g., 'New', 'Regular', 'Power_User').
    *   `pre_campaign_activity_df`: With 3000-5000 rows. Columns: `activity_id` (unique integers), `user_id` (randomly sampled from `users_df` IDs), `activity_date` (random dates before the simulated campaign launch), `activity_type` (e.g., 'login', 'view_dashboard', 'edit_profile', 'access_report'), `duration_seconds` (random floats 10-600).
    *   `campaign_exposure_df`: With 300-500 rows. Columns: `exposure_id` (unique integers), `user_id` (a subset of `users_df` IDs, representing those exposed to a campaign), `exposure_date` (random dates within a 30-day window, defining the campaign launch), `campaign_variant` (e.g., 'Control', 'Variant_A', 'Variant_B').
    *   `post_campaign_feature_usage_df`: With 800-1200 rows. Columns: `usage_id` (unique integers), `user_id` (randomly sampled from *exposed* `users_df` IDs), `usage_date` (random dates *after* respective `exposure_date`), `feature_name` (e.g., 'New_Dashboard_Analytics', 'Improved_Search', 'AI_Assistant').
    *   **Simulate realistic patterns**: Ensure `activity_date` and `usage_date` are chronologically consistent with `signup_date` and `exposure_date`. Simulate that users in `Variant_A` or `Variant_B` have a higher probability of using `post_campaign_feature_usage_df` compared to 'Control', and that `user_segment` or `pre_campaign_activity` also influence future feature adoption. Some users in `campaign_exposure_df` should not adopt the feature.

2. **Load into SQLite & SQL Feature Engineering**: Create an in-memory SQLite database using `sqlite3`. Load `users_df`, `pre_campaign_activity_df`, and `campaign_exposure_df` into tables named `users`, `activity`, and `exposure` respectively. Determine a `campaign_launch_date` (e.g., `min(exposure_date)` from `campaign_exposure_df`, using pandas).
    Write a single SQL query that performs the following for *each user* in the `campaign_exposure_df`:
    *   **Joins** `users`, `activity`, and `exposure` tables.
    *   **Aggregates features based on user activity *before* their `exposure_date`**:
        *   `num_pre_campaign_logins` (count of 'login' `activity_type`)
        *   `total_pre_campaign_duration` (sum of `duration_seconds`)
        *   `days_since_last_pre_campaign_activity`: Number of days between the user's `exposure_date` and their `MAX(activity_date)` (only considering activities before `exposure_date`).
    *   **Includes static user and campaign attributes**: `user_id`, `age`, `region`, `user_segment`, `signup_date`, `exposure_date`, `campaign_variant`.
    *   **Ensures** all users from `campaign_exposure_df` are included (using `LEFT JOIN`), showing 0 for counts/sums and `NULL` for `days_since_last_pre_campaign_activity` if no activity before exposure.
    *   The query should return `user_id`, `age`, `region`, `user_segment`, `signup_date`, `exposure_date`, `campaign_variant`, `num_pre_campaign_logins`, `total_pre_campaign_duration`, `days_since_last_pre_campaign_activity`.

3. **Pandas Feature Engineering & Binary Target Creation**: Fetch the SQL query results into a pandas DataFrame (`campaign_features_df`).
    *   Handle `NaN` values: Fill `num_pre_campaign_logins`, `total_pre_campaign_duration` with 0. For `days_since_last_pre_campaign_activity` (for users with no activity before exposure), fill with a large sentinel value (e.g., 9999 days).
    *   Convert `signup_date` and `exposure_date` to datetime objects. Calculate `account_age_at_exposure_days`: Days between `signup_date` and `exposure_date`.
    *   **Create the Binary Target `adopted_new_feature`**: A user is considered to have `adopted_new_feature` (1) if they have *any* entry in the original `post_campaign_feature_usage_df` with `usage_date` occurring *between* their `exposure_date` and `exposure_date + 60 days`. Otherwise, 0.
    *   Define features `X` (all numerical: `age`, `account_age_at_exposure_days`, `num_pre_campaign_logins`, `total_pre_campaign_duration`, `days_since_last_pre_campaign_activity`; categorical: `region`, `user_segment`, `campaign_variant`) and target `y` (`adopted_new_feature`). Split into training and testing sets (e.g., 70/30 split) using `sklearn.model_selection.train_test_split` (set `random_state=42`, `stratify` on `y` for class balance).

4. **Data Visualization**: Create two separate plots to visually inspect relationships with `adopted_new_feature`:
    *   A stacked bar chart showing the proportion of `adopted_new_feature` (0 or 1) across different `campaign_variant`s. Include a clear title indicating the adoption rate by variant.
    *   A violin plot (or box plot) showing the distribution of `account_age_at_exposure_days` for users who `adopted_new_feature=0` vs. `adopted_new_feature=1`.
    Ensure plots have appropriate labels and titles.

5. **ML Pipeline & Evaluation (Binary Classification)**: 
    *   Create an `sklearn.pipeline.Pipeline` with a `sklearn.compose.ColumnTransformer` for preprocessing:
        *   For numerical features (e.g., `age`, `account_age_at_exposure_days`, `num_pre_campaign_logins`, `total_pre_campaign_duration`, `days_since_last_pre_campaign_activity`): Apply `sklearn.preprocessing.SimpleImputer(strategy='mean')` followed by `sklearn.preprocessing.StandardScaler`.
        *   For categorical features (`region`, `user_segment`, `campaign_variant`): Apply `sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')`.
    *   The final estimator in the pipeline should be `sklearn.ensemble.GradientBoostingClassifier` (set `random_state=42`, `n_estimators=100`, `learning_rate=0.1`).
    *   Train the pipeline on `X_train`, `y_train`. Predict probabilities for the positive class (class 1) on the test set (`X_test`).
    *   Calculate and print the `sklearn.metrics.roc_auc_score` and a `sklearn.metrics.classification_report` for the test set predictions.

## Focus
Predicting user feature adoption based on pre-campaign activity and A/B test exposure.

## Dataset
Synthetic data: User demographics, historical activity logs, campaign exposure details, and post-campaign feature usage.

## Hint
When calculating `adopted_new_feature`, ensure you're filtering `post_campaign_feature_usage_df` for the correct time window *relative to each user's specific `exposure_date`* and then merging this aggregated adoption status back to your main feature DataFrame. For SQL, remember to use `LEFT JOIN` with `campaign_exposure_df` as the base to ensure all exposed users are included, even those with no prior activity.
