# AI Daily Lab — 2026-05-24

## Task
Develop a machine learning pipeline to predict if a streaming service customer will **churn** in the next 30 days, based on their profile, subscription details, and recent content viewing behavior.

1.  **Generate Synthetic Data (Pandas/Numpy)**: Create three pandas DataFrames:
    *   `customers_df`: With 1000-1500 rows. Columns: `customer_id` (unique integers), `signup_date` (random dates over the last 3-5 years), `subscription_plan` (e.g., 'Basic', 'Premium', 'Family'), `region` (e.g., 'North', 'South', 'East', 'West'), `age_group` (e.g., '18-24', '25-44', '45+'), `churn_date` (random dates, for ~15-20% of customers, occurring *after* `signup_date` and within the last 12 months, `NaT` otherwise).
    *   `content_df`: With 100-200 rows. Columns: `content_id` (unique integers), `title` (unique strings), `genre` (e.g., 'Action', 'Comedy', 'Drama', 'Sci-Fi', 'Documentary'), `avg_rating` (random floats 1.0-5.0).
    *   `viewing_history_df`: With 20000-30000 rows. Columns: `view_id` (unique integers), `customer_id` (randomly sampled from `customers_df` IDs), `content_id` (randomly sampled from `content_df` IDs), `view_date` (random datetimes occurring *after* respective `signup_date` and *before* `churn_date` if applicable), `duration_minutes` (random floats 5.0-180.0).
    *   **Simulate realistic patterns**: Ensure `view_date` is always after `signup_date` and before `churn_date`. Simulate varying viewing habits: 'Premium' plan users should generally have higher `duration_minutes` and more frequent views. For customers with a `churn_date`, their `duration_minutes` and `num_views` should show a noticeable drop-off in the 1-2 months leading up to their `churn_date` compared to their earlier activity. Sort `viewing_history_df` by `customer_id` then `view_date`.

2.  **Load into SQLite & SQL Feature Engineering (Recent Viewing Behavior)**: Create an in-memory SQLite database using `sqlite3`. Load `customers_df`, `content_df`, and `viewing_history_df` into tables named `customers`, `content`, and `viewing_history` respectively.
    *   Define `GLOBAL_PREDICTION_CUTOFF_DATE` as 1 month prior to the latest `view_date` in your generated `viewing_history_df` (e.g., `viewing_history_df['view_date'].max() - pd.Timedelta(months=1)`).
    *   Write a single SQL query that performs the following for *each customer*, aggregating their content viewing behavior *within the 30 days immediately preceding `GLOBAL_PREDICTION_CUTOFF_DATE`*:
        *   `current_cutoff_date` (the `GLOBAL_PREDICTION_CUTOFF_DATE` itself, for consistency).
        *   `total_view_duration_prev_30d` (sum of `duration_minutes`).
        *   `num_views_prev_30d` (count of `view_id`s).
        *   `num_unique_content_prev_30d` (count of distinct `content_id`s).
        *   `num_unique_genres_prev_30d` (count of distinct `genre`s viewed).
        *   `days_since_last_view_at_cutoff`: Number of days between `current_cutoff_date` and the most recent `view_date` *before or on* the cutoff. Return a large number (e.g., 9999) if no views before cutoff.
    *   **Includes static customer attributes**: `customer_id`, `signup_date`, `subscription_plan`, `region`, `age_group`.
    *   **Ensures** all customers are included (using `LEFT JOIN`), showing 0 for counts/sums and 0.0 for averages if no activity in the 30-day window. Handle `NULL`s appropriately.
    *   The query should return all mentioned fields.

3.  **Pandas Feature Engineering & Binary Target Creation**: Fetch the SQL query results into a pandas DataFrame (`customer_features_df`).
    *   Convert all relevant date columns (`signup_date`, `current_cutoff_date`) to datetime objects.
    *   Handle `NaN` values: Fill numerical aggregated features (`total_view_duration_prev_30d`, etc.) with 0 or 0.0 as appropriate. Fill `days_since_last_view_at_cutoff` with 9999.
    *   Calculate `customer_tenure_at_cutoff_days`: Number of days between `signup_date` and `current_cutoff_date`.
    *   Calculate `avg_view_duration_per_view_prev_30d`: `total_view_duration_prev_30d` / `num_views_prev_30d`. Fill any `NaN` or `inf` with 0.
    *   **Create the Binary Target `will_churn_in_next_30_days`**: For *each customer*, determine if their simulated `churn_date` (from the original `customers_df` merged back) falls within the 30-day period *immediately following* their `current_cutoff_date`. Merge this target (1 if yes, 0 if no) with `customer_features_df`, filling `NaN`s with 0 for customers who did not churn or whose churn date falls outside the window.
    *   Define features `X` (numerical: `total_view_duration_prev_30d`, `num_views_prev_30d`, `num_unique_content_prev_30d`, `num_unique_genres_prev_30d`, `days_since_last_view_at_cutoff`, `customer_tenure_at_cutoff_days`, `avg_view_duration_per_view_prev_30d`; categorical: `subscription_plan`, `region`, `age_group`) and target `y` (`will_churn_in_next_30_days`). Split into training and testing sets (e.g., 70/30 split) using `sklearn.model_selection.train_test_split` (set `random_state=42`, `stratify` on `y` for class balance).

4.  **Data Visualization (Matplotlib/Seaborn)**: Create two separate plots to visually inspect relationships with `will_churn_in_next_30_days`:
    *   A violin plot (or box plot) showing the distribution of `total_view_duration_prev_30d` for non-churners (0) vs. churners (1). Ensure appropriate labels and titles.
    *   A stacked bar chart showing the proportion of `will_churn_in_next_30_days` (0 or 1) across different `subscription_plan` values. Ensure appropriate labels and titles.

5.  **ML Pipeline & Evaluation (Binary Classification)**:
    *   Create an `sklearn.pipeline.Pipeline` with a `sklearn.compose.ColumnTransformer` for preprocessing:
        *   For numerical features: Apply `sklearn.preprocessing.SimpleImputer(strategy='mean')` followed by `sklearn.preprocessing.StandardScaler`.
        *   For categorical features: Apply `sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')`.
    *   The final estimator in the pipeline should be `sklearn.ensemble.HistGradientBoostingClassifier` (set `random_state=42`, and consider `class_weight='balanced'` due to potential target imbalance).
    *   Train the pipeline on `X_train`, `y_train`. Predict probabilities for the positive class (class 1) on the test set (`X_test`).
    *   Calculate and print the `sklearn.metrics.roc_auc_score` and a `sklearn.metrics.classification_report` for the test set predictions.

## Focus
Time-series feature engineering from user activity logs, binary classification for customer churn prediction, handling class imbalance.

## Dataset
Synthetic customer profiles, content metadata, and granular viewing history for a streaming service.

## Hint
When performing SQL aggregations for time-windowed features, remember to join `viewing_history` with `content` to access `genre` information. Use `julianday()` for date arithmetic in SQL, and `COALESCE` to manage `NULL` values gracefully when no activity is found for a customer in a specific time window.
