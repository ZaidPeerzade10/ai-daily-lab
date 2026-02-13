# AI Daily Lab — 2026-02-13

## Task
1. **Generate Synthetic Data (Pandas/Numpy)**: Create three pandas DataFrames:
    *   `employees_df`: With 500-700 rows. Columns: `employee_id` (unique integers), `hire_date` (random dates over the last 5-10 years), `department` (e.g., 'HR', 'Engineering', 'Sales', 'Marketing'), `job_role` (e.g., 'Analyst', 'Manager', 'Senior Developer', 'Associate'), `salary` (random floats 50000-200000, higher for certain roles/departments), `gender` (e.g., 'Male', 'Female', 'Other'), `is_churned` (binary target, 0 or 1, with an approximate 15-25% churn rate).
    *   `performance_reviews_df`: With 1000-1500 rows. Columns: `review_id` (unique integers), `employee_id` (randomly sampled from `employees_df` IDs), `review_date` (random dates occurring *after* `hire_date`), `rating` (random integers 1-5, biased towards 3-5).
    *   `training_completion_df`: With 800-1200 rows. Columns: `completion_id` (unique integers), `employee_id` (randomly sampled from `employees_df` IDs), `completion_date` (random dates occurring *after* `hire_date`), `course_category` (e.g., 'Technical', 'Soft_Skills', 'Compliance', 'Leadership').
    *   **Simulate churn behavior**: Churned employees (`is_churned=1`) should generally have lower `rating`s, fewer `performance_reviews` and `training_completion` records, and their `review_date`s and `completion_date`s should be concentrated earlier in their tenure, with few or no records in the last 6-12 months before an `analysis_date`. Conversely, non-churned employees should show more consistent activity.

2. **Load into SQLite & SQL Feature Engineering**: Create an in-memory SQLite database using `sqlite3`. Load `employees_df`, `performance_reviews_df`, and `training_completion_df` into tables named `employees`, `reviews`, and `training` respectively. Determine a `global_analysis_date` (e.g., `max(review_date, completion_date)` from all available data + 90 days, using pandas) and a `feature_cutoff_date` (`global_analysis_date` - 180 days).
    Write a single SQL query that performs the following for *each employee*, aggregating their activity *before* the `feature_cutoff_date`:
    *   **Joins** `employees`, `reviews`, and `training` tables.
    *   **Aggregates features based on activity *before* `feature_cutoff_date`**: 
        *   `avg_performance_rating_pre_cutoff` (average of `rating`)
        *   `num_reviews_pre_cutoff` (count of `review_id`s)
        *   `days_since_last_review_pre_cutoff`: Number of days between `feature_cutoff_date` and `MAX(review_date)` for the employee.
        *   `num_trainings_pre_cutoff` (count of `completion_id`s)
        *   `days_since_last_training_pre_cutoff`: Number of days between `feature_cutoff_date` and `MAX(completion_date)` for the employee.
    *   **Includes static employee attributes**: `employee_id`, `department`, `job_role`, `salary`, `gender`, `hire_date`, `is_churned`.
    *   **Ensures** all employees are included (using `LEFT JOIN`), showing 0 for counts, 0.0 for averages, and `NULL` for `days_since_last_review_pre_cutoff`/`days_since_last_training_pre_cutoff` if no activity before cutoff.
    *   The query should return `employee_id`, `department`, `job_role`, `salary`, `gender`, `hire_date`, `is_churned`, and all the aggregated features.

3. **Pandas Feature Engineering & Data Preparation**: Fetch the SQL query results into a pandas DataFrame (`employee_features_df`).
    *   Handle `NaN` values: Fill `num_reviews_pre_cutoff`, `num_trainings_pre_cutoff` with 0. Fill `avg_performance_rating_pre_cutoff` with a neutral value (e.g., 3.0) for employees with no reviews. For `days_since_last_review_pre_cutoff` and `days_since_last_training_pre_cutoff`, fill with a large sentinel value (e.g., `365 * 10` or 3650 days).
    *   Convert `hire_date` to datetime objects. Calculate `tenure_at_cutoff_days`: The number of days between `hire_date` and the `feature_cutoff_date`.
    *   Define features `X` (all numerical: `salary`, `tenure_at_cutoff_days`, `avg_performance_rating_pre_cutoff`, `num_reviews_pre_cutoff`, `days_since_last_review_pre_cutoff`, `num_trainings_pre_cutoff`, `days_since_last_training_pre_cutoff`; categorical: `department`, `job_role`, `gender`) and target `y` (`is_churned`). Split into training and testing sets (e.g., 70/30 split) using `sklearn.model_selection.train_test_split` (set `random_state=42`, `stratify` on `y` for class balance).

4. **Data Visualization**: Create two separate plots to visually inspect relationships with `is_churned`:
    *   A violin plot (or box plot) showing the distribution of `salary` for employees with `is_churned=0` vs. `is_churned=1`.
    *   A stacked bar chart showing the proportion of `is_churned` (0 or 1) across different `department` values.
    Ensure plots have appropriate labels and titles.

5. **ML Pipeline & Evaluation (Binary Classification)**: 
    *   Create an `sklearn.pipeline.Pipeline` with a `sklearn.compose.ColumnTransformer` for preprocessing:
        *   For numerical features (e.g., `salary`, `tenure_at_cutoff_days`, `avg_performance_rating_pre_cutoff`, `num_reviews_pre_cutoff`, `days_since_last_review_pre_cutoff`, `num_trainings_pre_cutoff`, `days_since_last_training_pre_cutoff`): Apply `sklearn.preprocessing.SimpleImputer(strategy='mean')` followed by `sklearn.preprocessing.StandardScaler`.
        *   For categorical features (`department`, `job_role`, `gender`): Apply `sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')`.
    *   The final estimator in the pipeline should be `sklearn.ensemble.GradientBoostingClassifier` (set `random_state=42`, `n_estimators=100`, `learning_rate=0.1`).
    *   Train the pipeline on `X_train`, `y_train`. Predict probabilities for the positive class (class 1) on the test set (`X_test`).
    *   Calculate and print the `sklearn.metrics.roc_auc_score` and a `sklearn.metrics.classification_report` for the test set predictions.

## Focus
Employee Attrition Prediction based on historical performance and training data.

## Dataset
Synthetic employee records, performance reviews, and training completions.

## Hint
Pay close attention to the `feature_cutoff_date` and `global_analysis_date` in SQL for creating time-aware features. For handling `NULL`s for `days_since_last_review_pre_cutoff` or `days_since_last_training_pre_cutoff` in SQL, use `IFNULL` or `COALESCE` with a large default before calculating days to `feature_cutoff_date`. In Pandas, ensure you convert dates to datetime objects before calculating differences.
