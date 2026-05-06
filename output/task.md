# AI Daily Lab — 2026-05-06

## Task
Develop a machine learning pipeline to predict employee turnover (binary classification) within the next 90 days, based on employee profile and recent performance review data.

1.  **Generate Synthetic Data (Pandas/Numpy)**: Create two pandas DataFrames:
    *   `employees_df`: With 1000-1500 rows. Columns: `employee_id` (unique integers), `department` (e.g., 'Sales', 'Engineering', 'HR', 'Marketing'), `years_at_company` (random floats 0.5-20.0), `salary_level` (e.g., 'Junior', 'Mid', 'Senior', 'Lead'), `satisfaction_score` (random floats 0.1-1.0), `last_promotion_date` (random dates over the last 5 years, or `NaT` if never promoted), `hire_date` (random dates over the last 20 years), `exit_date` (random dates, for ~15-20% of employees, occurring *after* `hire_date` and within the last 18 months, `NaT` otherwise).
    *   `performance_reviews_df`: With 3000-5000 rows. Columns: `review_id` (unique integers), `employee_id` (randomly sampled from `employees_df` IDs), `review_date` (random dates *after* respective `hire_date` and before `exit_date` if applicable), `performance_score` (random integers 1-5), `feedback_sentiment` (e.g., 'Positive', 'Neutral', 'Negative').
    *   **Simulate realistic patterns**: Ensure `review_date` is always after `hire_date` and before `exit_date`. Simulate lower `satisfaction_score` for employees with `exit_date`. Employees with higher `years_at_company` should generally have higher `salary_level`. Employees with an `exit_date` should, for their last 1-3 reviews before exit, have slightly lower `performance_score` or 'Negative' `feedback_sentiment` 10-20% of the time. `performance_reviews_df` should have multiple reviews per employee over time.
    *   Sort `performance_reviews_df` by `employee_id` then `review_date`.

2.  **Load into SQLite & SQL Feature Engineering (Recent Performance & Tenure)**: Create an in-memory SQLite database using `sqlite3`. Load `employees_df` and `performance_reviews_df` into tables named `employees` and `performance_reviews` respectively.
    *   Define `GLOBAL_PREDICTION_CUTOFF_DATE` as 3 months prior to the latest `review_date` in your generated `performance_reviews_df` (e.g., `performance_reviews_df['review_date'].max() - pd.Timedelta(months=3)`).
    *   Write a single SQL query that performs the following for *each employee*, aggregating their performance review behavior and tenure *up to `GLOBAL_PREDICTION_CUTOFF_DATE`*:
        *   `current_cutoff_date` (the `GLOBAL_PREDICTION_CUTOFF_DATE` itself, for consistency).
        *   `avg_performance_prev_12m` (average of `performance_score` for reviews in the 12 months ending at `current_cutoff_date`).
        *   `last_review_score_at_cutoff` (most recent `performance_score` before or on `current_cutoff_date`).
        *   `days_since_last_review_at_cutoff`: Number of days between `current_cutoff_date` and the `review_date` of `last_review_score_at_cutoff`. Return a large number (e.g., 9999) if no reviews before cutoff.
        *   `num_reviews_prev_12m` (count of `review_id`s in the 12 months ending at `current_cutoff_date`).
        *   `days_since_last_promotion_at_cutoff`: Number of days between `current_cutoff_date` and `last_promotion_date`. Return a large number (e.g., 9999) if no promotion date or promotion after cutoff.
    *   **Includes static employee attributes**: `employee_id`, `department`, `years_at_company`, `salary_level`, `satisfaction_score`, `hire_date`.
    *   **Ensures** all employees are included (using `LEFT JOIN`), showing 0 for counts/sums/averages if no activity in the relevant windows.
    *   The query should return `employee_id`, `department`, `years_at_company`, `salary_level`, `satisfaction_score`, `hire_date`, `current_cutoff_date`, and all aggregated features.
    *   **Hint**: Use `julianday()` for date comparisons. Use `COALESCE` to handle `NULL`s from `LEFT JOIN`s or when no matching history.

3.  **Pandas Feature Engineering & Binary Target Creation**: Fetch the SQL query results into a pandas DataFrame (`employee_features_df`).
    *   Convert all relevant date columns (`hire_date`, `current_cutoff_date`, `last_promotion_date`) to datetime objects.
    *   Handle `NaN` values: Fill numerical aggregated features (`avg_performance_prev_12m`, etc.) with 0 or 0.0 as appropriate. Fill `days_since_last_review_at_cutoff` and `days_since_last_promotion_at_cutoff` with 9999.
    *   Calculate `employee_tenure_at_cutoff_days`: Number of days between `hire_date` and `current_cutoff_date`.
    *   Calculate `review_frequency_prev_12m`: `num_reviews_prev_12m` / 365.0. Fill any `NaN` or `inf` with 0.
    *   **Create the Binary Target `will_churn_in_next_90_days`**: For *each employee*, determine if their simulated `exit_date` (from the original `employees_df` merged back) falls within the 90-day period *immediately following* their `current_cutoff_date`. Merge this aggregate (1 if yes, 0 if no) with `employee_features_df`, filling `NaN`s with 0 for employees who did not exit or whose exit date falls outside the window.
    *   Define features `X` (numerical: `years_at_company`, `satisfaction_score`, `avg_performance_prev_12m`, `last_review_score_at_cutoff`, `days_since_last_review_at_cutoff`, `num_reviews_prev_12m`, `days_since_last_promotion_at_cutoff`, `employee_tenure_at_cutoff_days`, `review_frequency_prev_12m`; categorical: `department`, `salary_level`) and target `y` (`will_churn_in_next_90_days`). Split into training and testing sets (e.g., 70/30 split) using `sklearn.model_selection.train_test_split` (set `random_state=42`, `stratify` on `y` for class balance).

4.  **Data Visualization**: Create two separate plots to visually inspect relationships with `will_churn_in_next_90_days`:
    *   A violin plot (or box plot) showing the distribution of `satisfaction_score` for non-churners (0) vs. churners (1). Ensure appropriate labels and titles.
    *   A stacked bar chart showing the proportion of `will_churn_in_next_90_days` (0 or 1) across different `department` values. Ensure appropriate labels and titles.

5.  **ML Pipeline & Evaluation (Binary Classification)**: 
    *   Create an `sklearn.pipeline.Pipeline` with a `sklearn.compose.ColumnTransformer` for preprocessing:
        *   For numerical features: Apply `sklearn.preprocessing.SimpleImputer(strategy='mean')` followed by `sklearn.preprocessing.StandardScaler`.
        *   For categorical features: Apply `sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')`.
    *   The final estimator in the pipeline should be `sklearn.ensemble.HistGradientBoostingClassifier` (set `random_state=42`).
    *   Train the pipeline on `X_train`, `y_train`. Predict probabilities for the positive class (class 1) on the test set (`X_test`).
    *   Calculate and print the `sklearn.metrics.roc_auc_score` and a `sklearn.metrics.classification_report` for the test set predictions.

## Focus
Employee Turnover Prediction (Binary Classification)

## Dataset
Synthetic employee profiles and performance review history.

## Hint
Ensure your date comparisons in SQL correctly handle `NaT` (NULL) values for `last_promotion_date` and `exit_date`. When defining the target, carefully filter `performance_reviews` to ensure only reviews *before or on* the `GLOBAL_PREDICTION_CUTOFF_DATE` are used for features. For visualization, consider handling imbalanced classes if churn is rare.
