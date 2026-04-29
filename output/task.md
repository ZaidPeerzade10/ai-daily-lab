# AI Daily Lab — 2026-04-29

## Task
Develop a machine learning pipeline to predict whether a user will click on a given online advertisement (Click-Through Rate prediction), based on user profile, ad characteristics, and impression context.

## Focus
Binary Classification (Click/No-Click) for Online Ad Impressions

## Dataset
1. **Generate Synthetic Data (Pandas/Numpy)**: Create three pandas DataFrames:
    *   `users_df`: With 1000-1500 rows. Columns: `user_id` (unique integers), `signup_date` (random dates over the last 3 years), `age` (random integers 18-65), `gender` (e.g., 'Male', 'Female', 'Other'), `income_level` (e.g., 'Low', 'Medium', 'High'), `past_overall_ctr` (random floats 0.01-0.15, representing user's general historical click-through rate).
    *   `ads_df`: With 100-200 rows. Columns: `ad_id` (unique integers), `ad_category` (e.g., 'Electronics', 'Fashion', 'Travel', 'Finance', 'Health'), `ad_format` (e.g., 'Banner', 'Video', 'Text'), `base_cpc` (random floats 0.1-5.0), `ad_creation_date` (random dates over the last 2 years).
    *   `impressions_df`: With 20000-30000 rows. Columns: `impression_id` (unique integers), `user_id` (randomly sampled from `users_df` IDs), `ad_id` (randomly sampled from `ads_df` IDs), `timestamp` (random timestamps occurring *after* `signup_date` for user and *after* `ad_creation_date` for ad), `platform` (e.g., 'Web', 'MobileApp'), `location_country` (e.g., 'USA', 'CAN', 'GBR', 'AUS').
    *   **Simulate realistic click patterns**: Calculate `is_clicked` (binary, 0 or 1) for each impression. Bias `is_clicked` such that:
        *   Users with higher `past_overall_ctr` have a higher chance of clicking.
        *   Ads with higher `base_cpc` might generally have a slightly higher click rate (as they often represent more premium/targeted ads).
        *   Certain `ad_category` + `platform` combinations might perform better (e.g., 'Fashion' on 'MobileApp').
        *   Introduce some random noise to ensure variability. Overall click rate should be 2-8%.
    *   Sort `impressions_df` by `user_id` then `timestamp`.

2. **Load into SQLite & SQL Feature Engineering (Impression-Level Attributes)**: Create an in-memory SQLite database using `sqlite3`. Load `users_df`, `ads_df`, and `impressions_df` into tables named `users`, `ads`, and `impressions` respectively.
    Write a single SQL query that performs the following for *each impression*:
    *   **Joins** `impressions`, `users`, and `ads` data.
    *   **Extracts/Calculates features**:
        *   `impression_hour_of_day`: Hour when the impression occurred (0-23). (SQLite `strftime('%H', ...)`).
        *   `impression_day_of_week`: Day of the week when the impression occurred (0=Sunday, 6=Saturday). (SQLite `strftime('%w', ...)`).
        *   `signup_date` (from `users`).
        *   `ad_creation_date` (from `ads`).
        *   `age`, `gender`, `income_level`, `past_overall_ctr` (from `users`).
        *   `ad_category`, `ad_format`, `base_cpc` (from `ads`).
        *   `platform`, `location_country` (from `impressions`).
    *   **Includes static impression details**: `impression_id`, `user_id`, `ad_id`, `timestamp`, `is_clicked` (the target).
    *   The query should return all these attributes and engineered features.

3. **Pandas Feature Engineering & Binary Target Creation**: Fetch the SQL query results into a pandas DataFrame (`impression_features_df`).
    *   Convert `timestamp`, `signup_date`, and `ad_creation_date` to datetime objects.
    *   Calculate `days_since_user_signup_at_impression`: Number of days between `signup_date` and `timestamp`.
    *   Calculate `days_since_ad_creation_at_impression`: Number of days between `ad_creation_date` and `timestamp`.
    *   Calculate `ctr_x_cpc_interaction`: `past_overall_ctr` * `base_cpc`.
    *   Define features `X` (numerical: `age`, `past_overall_ctr`, `base_cpc`, `impression_hour_of_day`, `impression_day_of_week`, `days_since_user_signup_at_impression`, `days_since_ad_creation_at_impression`, `ctr_x_cpc_interaction`; categorical: `gender`, `income_level`, `ad_category`, `ad_format`, `platform`, `location_country`) and target `y` (`is_clicked`). Split into training and testing sets (e.g., 70/30 split) using `sklearn.model_selection.train_test_split` (set `random_state=42`, `stratify` on `y` for class balance).

4. **Data Visualization**: Create two separate plots to visually inspect relationships with `is_clicked`:
    *   A violin plot (or box plot) showing the distribution of `past_overall_ctr` for non-clicked (0) vs. clicked (1) impressions. Ensure appropriate labels and titles.
    *   A stacked bar chart showing the proportion of `is_clicked` (0 or 1) across different `ad_category` values. Ensure appropriate labels and titles.

5. **ML Pipeline & Evaluation (Binary Classification)**: 
    *   Create an `sklearn.pipeline.Pipeline` with a `sklearn.compose.ColumnTransformer` for preprocessing:
        *   For numerical features: Apply `sklearn.preprocessing.SimpleImputer(strategy='mean')` followed by `sklearn.preprocessing.StandardScaler`.
        *   For categorical features: Apply `sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')`.
    *   The final estimator in the pipeline should be `sklearn.ensemble.HistGradientBoostingClassifier` (set `random_state=42`).
    *   Train the pipeline on `X_train`, `y_train`. Predict probabilities for the positive class (class 1) on the test set (`X_test`).
    *   Calculate and print the `sklearn.metrics.roc_auc_score` and a `sklearn.metrics.classification_report` for the test set predictions.

## Hint
When simulating `is_clicked`, consider creating a baseline probability and then adjusting it based on user, ad, and context features with some added random noise. For date difference calculations in Pandas, ensure both columns are of `datetime` type, then subtract and extract days (e.g., `(df['end_date'] - df['start_date']).dt.days`). Remember that `HistGradientBoostingClassifier` can handle missing values internally, but the `SimpleImputer` is included for robustness and demonstration.
