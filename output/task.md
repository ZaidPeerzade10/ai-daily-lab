# AI Daily Lab — 2026-06-12

## Task
Develop a machine learning pipeline to predict if a newly sent marketing email will achieve **'High Open Rate'** (binary classification) within 24 hours of sending, based on campaign attributes, recipient segment demographics, and historical email performance up to a specific cutoff date.

1.  **Generate Synthetic Data (Pandas/Numpy)**: Create three pandas DataFrames:
    *   `emails_df`: With 10000-15000 rows. Columns: `email_id` (unique integers), `campaign_id` (randomly sampled from `campaigns_df` IDs), `segment_id` (randomly sampled from `segments_df` IDs), `send_date` (random datetimes over the last 2 years), `subject_line_length` (random integers 20-120), `_actual_24h_open_rate` (target creation: random floats 0.05-0.60).
    *   `campaigns_df`: With 1000-1500 rows. Columns: `campaign_id` (unique integers), `sender_name` (e.g., 'Support Team', 'Marketing Dept', 'Product Updates' - 5-7 distinct names), `campaign_type` (e.g., 'Promotional', 'Newsletter', 'Transactional'), `send_hour_of_day` (random integers 0-23).
    *   `segments_df`: With 100-200 rows. Columns: `segment_id` (unique integers), `avg_recipient_age` (random integers 18-70), `region` (e.g., 'North', 'South', 'East', 'West'), `past_avg_segment_engagement_score` (random floats 0.1-0.9, simulating overall engagement history).
    *   **Simulate realistic patterns**: Ensure `_actual_24h_open_rate` correlates positively with `past_avg_segment_engagement_score` and negatively with very long `subject_line_length`. Introduce significant open rate differences between `sender_name`s and `campaign_type`s. Simulate a slight upward trend in open rates over time, but with some volatility. Ensure `send_hour_of_day` is realistic.
    *   Sort `emails_df` by `send_date`.

2.  **Load into SQLite & SQL Feature Engineering (Time-Windowed Aggregations)**: Create an in-memory SQLite database using `sqlite3`. Load `emails_df`, `campaigns_df`, and `segments_df` into tables named `emails`, `campaigns`, and `segments` respectively.
    *   Define `GLOBAL_PREDICTION_CUTOFF_DATE` as 2 weeks prior to the latest `send_date` in your generated `emails_df`.
    *   Write a single SQL query that performs the following for *each email sent AFTER `GLOBAL_PREDICTION_CUTOFF_DATE`*:
        *   Joins the `emails` (filtered for events after cutoff) with the `campaigns` and `segments` tables.
        *   Aggregates historical features based on sales *within the same `segment_id` and `campaign_type` in the 90 days preceding `GLOBAL_PREDICTION_CUTOFF_DATE`*:
            *   `avg_segment_campaign_open_rate_prev_90d` (average `_actual_24h_open_rate`).
            *   `num_segment_campaigns_prev_90d` (count of emails).
        *   Aggregates historical features based on sales *within the same `sender_name` in the 90 days preceding `GLOBAL_PREDICTION_CUTOFF_DATE`*:
            *   `avg_sender_open_rate_prev_90d`.
            *   `num_sender_campaigns_prev_90d`.
        *   Extracts time-based features from the `send_date` of the *current* email: `send_day_of_week` (0-6), `send_month_of_year` (1-12).
        *   Includes static attributes: `email_id`, `send_date`, `subject_line_length`, `sender_name`, `campaign_type`, `send_hour_of_day`, `avg_recipient_age`, `region`, `past_avg_segment_engagement_score`, and the target `_actual_24h_open_rate` for the *current* email.
    *   **Ensures** all emails *after* the cutoff are included. Handle `NULL`s for historical aggregates (e.g., 0.0 for averages, 0 for counts if no prior activity).
    *   The query should return all mentioned fields.
    *   **Hint**: Use CTEs for pre-calculating historical aggregates up to `GLOBAL_PREDICTION_CUTOFF_DATE`, then join these with the future emails. Use `julianday()` for date comparisons.

3.  **Pandas Feature Engineering & Binary Target Creation**: Fetch the SQL query results into a pandas DataFrame (`email_features_df`).
    *   Convert `send_date` to datetime objects.
    *   Handle `NaN` values: Fill numerical historical aggregated features (e.g., averages, counts) with 0.0 or 0 as appropriate. Fill `avg_recipient_age` with its mean, `past_avg_segment_engagement_score` with its mean.
    *   **Create the Binary Target `is_high_open_rate`**: Based on `_actual_24h_open_rate`:
        *   'High Open Rate' (1): If `_actual_24h_open_rate` > (75th percentile of `_actual_24h_open_rate`)
        *   'Not High Open Rate' (0): Otherwise
        (Adjust the percentile threshold dynamically based on your synthetic data distribution to ensure a reasonable class balance, e.g., 20-30% positive class).
    *   Define features `X` (numerical: `subject_line_length`, `send_hour_of_day`, `avg_recipient_age`, `past_avg_segment_engagement_score`, `avg_segment_campaign_open_rate_prev_90d`, `num_segment_campaigns_prev_90d`, `avg_sender_open_rate_prev_90d`, `num_sender_campaigns_prev_90d`, `send_day_of_week`, `send_month_of_year`; categorical: `sender_name`, `campaign_type`, `region`) and target `y` (`is_high_open_rate`).
    *   Split into training and testing sets (e.g., 70/30 split) using `sklearn.model_selection.train_test_split` (set `random_state=42`, `stratify` on `y` for class balance).

4.  **Data Visualization (Matplotlib/Seaborn)**: Create two separate plots to visually inspect relationships with `is_high_open_rate`:
    *   A violin plot (or box plot) showing the distribution of `subject_line_length` for 'Not High Open Rate' (0) vs. 'High Open Rate' (1) articles. Ensure appropriate labels and titles.
    *   A stacked bar chart showing the proportion of `is_high_open_rate` (0 or 1) across different `campaign_type` values. Ensure appropriate labels and titles.

5.  **ML Pipeline & Evaluation (Binary Classification)**:
    *   Create an `sklearn.pipeline.Pipeline` with a `sklearn.compose.ColumnTransformer` for preprocessing:
        *   For numerical features: Apply `sklearn.preprocessing.SimpleImputer(strategy='mean')` followed by `sklearn.preprocessing.StandardScaler`.
        *   For categorical features: Apply `sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')`.
    *   The final estimator in the pipeline should be `sklearn.ensemble.HistGradientBoostingClassifier` (set `random_state=42`, and consider `class_weight='balanced'` due to potential target imbalance).
    *   Train the pipeline on `X_train`, `y_train`. Predict probabilities for the positive class (class 1) on the test set (`X_test`).
    *   Calculate and print the `sklearn.metrics.roc_auc_score` and a `sklearn.metrics.classification_report` for the test set predictions.

## Focus
Binary classification of marketing email success, leveraging time-windowed historical performance and campaign/recipient features.

## Dataset
Synthetic marketing email campaigns, recipient segments, and historical email engagement data.

## Hint
Carefully manage date comparisons and filtering in SQL for the `GLOBAL_PREDICTION_CUTOFF_DATE` to ensure no data leakage. When creating the binary target `is_high_open_rate`, calculate the percentile dynamically from `_actual_24h_open_rate` to ensure a reasonable class balance for classification.
