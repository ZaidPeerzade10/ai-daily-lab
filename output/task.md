# AI Daily Lab — 2026-01-24

## Task
1. **Generate Synthetic Data**: Create two pandas DataFrames:
    *   `users_df`: With 500 rows. Columns: `user_id` (unique integers), `signup_date` (random dates over the last 5 years), `country` (e.g., 'USA', 'Canada', 'UK', 'Germany' with random distribution), `subscription_type` (e.g., 'Free', 'Basic', 'Premium').
    *   `activities_df`: With 3000-5000 rows. Columns: `activity_id` (unique integers), `user_id` (randomly sampled from `users_df` IDs), `activity_date` (random dates *after* `signup_date`), `activity_type` (e.g., 'login', 'view_item', 'post_comment', 'upgrade_plan' with varying frequencies).

2. **Load into SQLite & SQL Analytics**: Create an in-memory SQLite database. Load `users_df` into a table named `users` and `activities_df` into a table named `activities`. Determine an `analysis_date` (e.g., `max(activity_date)` from `activities_df` + 30 days). Write a single SQL query that performs the following for *each user*:
    *   **Joins** `users` and `activities` tables.
    *   **Aggregates** user-level features: `total_activities` (count of all activities), `num_logins` (count of 'login' events), `num_comments_posted` (count of 'post_comment' events).
    *   Calculates `days_since_last_activity`: Days between `analysis_date` (passed as a parameter or calculated within SQL if possible via date functions) and `MAX(activity_date)`.
    *   **Ensures** all users are included, showing 0 for counts and `NULL` for `days_since_last_activity` if no activities.
    *   Returns `user_id`, `country`, `subscription_type`, `signup_date`, `total_activities`, `num_logins`, `num_comments_posted`, `days_since_last_activity`.

3. **Pandas Feature Engineering & Multi-Class Target Creation**: Fetch the SQL query results into a pandas DataFrame. 
    *   Merge with `users_df` (if necessary, to ensure all original user features like `country`, `subscription_type` are present even for users with no activities) using a left join on `user_id`.
    *   Handle `NaN` values: Fill `total_activities`, `num_logins`, `num_comments_posted` with 0. For `days_since_last_activity` (for users with no activities), fill with a large sentinel value (e.g., `365 * 5` or 1825 days).
    *   Calculate `account_age_days`: Days between `signup_date` and the `analysis_date` (from step 2).
    *   **Create the multi-class target `activity_segment`**: Based on `total_activities` and `days_since_last_activity`. First, calculate quantiles for `total_activities`. Then, define segments:
        *   'High_Activity': `total_activities` is in the top 33% *and* `days_since_last_activity` < 30 days.
        *   'Medium_Activity': `total_activities` is in the middle 34-66% *or* `days_since_last_activity` between 30 and 90 days (excluding those already classified as 'High_Activity').
        *   'Low_Activity': All remaining users (including those with 0 activities or `days_since_last_activity` >= 90 days).
    *   Define features `X` (numerical: `account_age_days`, `total_activities`, `num_logins`, `num_comments_posted`, `days_since_last_activity`; categorical: `country`, `subscription_type`) and target `y` (`activity_segment`). Split into training and testing sets (e.g., 70/30 split) using `sklearn.model_selection.train_test_split` (set `random_state=42`).

4. **Data Visualization**: Create two separate plots to visually inspect relationships with `activity_segment`:
    *   A violin plot or box plot showing the distribution of `days_since_last_activity` for each `activity_segment`.
    *   A bar plot or count plot showing the distribution of `activity_segment` across different `subscription_type`s.
    Ensure plots have appropriate labels and titles.

5. **ML Pipeline & Evaluation (Multi-Class)**: 
    *   Create an `sklearn.pipeline.Pipeline` with a `ColumnTransformer` for preprocessing:
        *   For numerical features: Apply `StandardScaler`.
        *   For categorical features: Apply `OneHotEncoder(handle_unknown='ignore')`.
    *   The final estimator should be `sklearn.ensemble.RandomForestClassifier` (set `random_state=42`, `n_estimators=100`, `class_weight='balanced'` for potential class imbalance).
    *   Train the pipeline on `X_train`, `y_train`. Predict `activity_segment` for `X_test`.
    *   Calculate and print the `sklearn.metrics.accuracy_score` and `sklearn.metrics.classification_report` for the test set predictions.

## Focus
SQL for complex user activity features, deriving a multi-class target from aggregated data, advanced Pandas feature engineering, ColumnTransformer with RandomForestClassifier for multi-class prediction, and insightful visualizations.

## Dataset
Synthetic customer profiles and activity logs across different subscription types and countries.

## Hint
When creating the `activity_segment` target, use `pd.qcut` or `df['col'].quantile()` to define activity tiers based on `total_activities`. Remember to handle users with no activities (NaNs) correctly after SQL aggregation and for the `days_since_last_activity` sentinel value. For `RandomForestClassifier` and multi-class problems, `class_weight='balanced'` can be helpful if segments are unevenly distributed.
