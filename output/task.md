# AI Daily Lab — 2026-02-10

## Task
1. **Generate Synthetic Data (Pandas/Numpy)**: Create three pandas DataFrames:
    *   `users_df`: With 500-700 rows. Columns: `user_id` (unique integers), `signup_date` (random dates over the last 3 years), `age` (random integers 18-70), `gender` (e.g., 'Male', 'Female', 'Other'), `region` (e.g., 'North', 'South', 'East', 'West').
    *   `ads_df`: With 100-150 rows. Columns: `ad_id` (unique integers), `ad_category` (e.g., 'Fashion', 'Tech', 'Travel', 'Food', 'Finance'), `ad_type` (e.g., 'Banner', 'Video', 'Text'), `target_audience_age_group` (e.g., '18-24', '25-34', '35-44', '45+').
    *   `impressions_df`: With 5000-8000 rows. Columns: `impression_id` (unique integers), `user_id` (randomly sampled from `users_df` IDs), `ad_id` (randomly sampled from `ads_df` IDs), `impression_date` (random dates occurring *after* `signup_date`), `device_type` (e.g., 'Mobile', 'Desktop', 'Tablet'), `was_clicked` (binary, 0 or 1).
    *   **Simulate realistic CTR patterns**: Ensure `was_clicked` has an approximate 10-15% overall click rate. Bias clicks: higher click rates if `user_id`'s `age` matches `ad_id`'s `target_audience_age_group`, for certain `ad_category`s with specific `regions`, or for specific `device_type`s. Sort `impressions_df` by `user_id` then `impression_date`.

2. **Load into SQLite & SQL Feature Engineering (Prior Interaction History)**: Create an in-memory SQLite database using `sqlite3`. Load `users_df`, `ads_df`, and `impressions_df` into tables named `users`, `ads`, and `impressions` respectively.
    Write a single SQL query that performs the following for *each impression*, calculating features based on the user's *prior impressions* and the ad's *prior impressions*:
    *   **Joins** `users`, `ads`, and `impressions` tables.
    *   **Calculates sequential features for each impression based on *all prior impressions* (excluding the current one) for the same user and same ad, respectively**:
        *   `user_past_total_impressions`: Count of user's previous impressions.
        *   `user_past_total_clicks`: Count of user's previous clicks.
        *   `user_past_ctr`: `user_past_total_clicks` / `user_past_total_impressions` (0 if no prior impressions).
        *   `days_since_last_user_impression`: Number of days between the current `impression_date` and the user's most recent prior `impression_date`. If it's the user's first impression, use `NULL`.
        *   `ad_past_total_impressions`: Count of ad's previous impressions.
        *   `ad_past_total_clicks`: Count of ad's previous clicks.
        *   `ad_past_ctr`: `ad_past_total_clicks` / `ad_past_total_impressions` (0 if no prior impressions).
    *   **Includes static user and ad attributes**: `age`, `gender`, `region`, `ad_category`, `ad_type`, `target_audience_age_group`, `signup_date`.
    *   The query should return `impression_id`, `user_id`, `ad_id`, `impression_date`, `device_type`, `was_clicked`, and all engineered features.
    *   **Hint**: Use window functions with `OVER (PARTITION BY ... ORDER BY ... ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)` for prior aggregates, and `LAG()` for `days_since_last_user_impression`.

3. **Pandas Feature Engineering & Binary Target Creation**: Fetch the SQL query results into a pandas DataFrame (`impression_features_df`).
    *   Handle `NaN` values: Fill `user_past_total_impressions`, `user_past_total_clicks`, `ad_past_total_impressions`, `ad_past_total_clicks` with 0. Fill `user_past_ctr` and `ad_past_ctr` with 0.0. For `days_since_last_user_impression` (for a user's first impression), fill with a large sentinel value (e.g., 9999 days).
    *   Convert `signup_date` and `impression_date` to datetime objects. Calculate `user_account_age_at_impression_days`: Days between `signup_date` and `impression_date`.
    *   **Create `user_ad_age_match`**: A binary feature (1 if the user's `age` falls within the `target_audience_age_group` for the ad, 0 otherwise). You'll need to parse the age group string (e.g., '18-24' to min/max age).
    *   Define features `X` (all numerical: `age`, `user_account_age_at_impression_days`, `user_past_total_impressions`, `user_past_total_clicks`, `user_past_ctr`, `days_since_last_user_impression`, `ad_past_total_impressions`, `ad_past_total_clicks`, `ad_past_ctr`; categorical: `gender`, `region`, `ad_category`, `ad_type`, `device_type`, `target_audience_age_group`, `user_ad_age_match`) and target `y` (`was_clicked`). Split into training and testing sets (e.g., 70/30 split) using `sklearn.model_selection.train_test_split` (set `random_state=42`, `stratify` on `y` for class balance).

4. **Data Visualization**: Create two separate plots to visually inspect relationships with `was_clicked`:
    *   A bar plot comparing the Click-Through Rate (mean of `was_clicked`) across different `device_type`s. Include a title like 'CTR by Device Type'.
    *   A stacked bar chart showing the distribution of `was_clicked` (0 or 1) across different `ad_category` values. Ensure appropriate labels and titles.

5. **ML Pipeline & Evaluation (Binary Classification)**: 
    *   Create an `sklearn.pipeline.Pipeline` with a `sklearn.compose.ColumnTransformer` for preprocessing:
        *   For numerical features (e.g., `age`, `user_account_age_at_impression_days`, `user_past_total_impressions`, `user_past_total_clicks`, `user_past_ctr`, `days_since_last_user_impression`, `ad_past_total_impressions`, `ad_past_total_clicks`, `ad_past_ctr`): Apply `sklearn.preprocessing.SimpleImputer(strategy='mean')` followed by `sklearn.preprocessing.StandardScaler`.
        *   For categorical features (`gender`, `region`, `ad_category`, `ad_type`, `device_type`, `target_audience_age_group`, `user_ad_age_match`): Apply `sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')`.
    *   The final estimator in the pipeline should be `sklearn.ensemble.HistGradientBoostingClassifier` (set `random_state=42`).
    *   Train the pipeline on `X_train`, `y_train`. Predict probabilities for the positive class (class 1) on the test set (`X_test`).
    *   Calculate and print the `sklearn.metrics.roc_auc_score` and a `sklearn.metrics.classification_report` for the test set predictions.

## Focus
Predicting Ad Click-Through Rate (CTR) using historical user and ad interaction features with sequential SQL analytics.

## Dataset
Synthetic user demographics, ad characteristics, and ad impression logs with click outcomes.

## Hint
Pay close attention to generating the sequential features in SQL using window functions like `COUNT(...) OVER (PARTITION BY ... ORDER BY ... ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)` for cumulative sums/counts and `LAG()` for time differences between consecutive events for each user. Also, ensure the synthetic `was_clicked` target reflects some logical biases (e.g., age-matching) to make the classification task meaningful.
