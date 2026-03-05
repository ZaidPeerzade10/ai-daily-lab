# AI Daily Lab — 2026-03-05

## Task
1. **Generate Synthetic Data (Pandas/Numpy)**: Create four pandas DataFrames:
    *   `employees_df`: With 500-700 rows. Columns: `employee_id` (unique integers), `hire_date` (random dates over the last 5 years), `department` (e.g., 'Engineering', 'Marketing', 'Sales', 'HR'), `role_level` (e.g., 'Associate', 'Senior', 'Lead', 'Manager'), `is_remote` (binary, 0 or 1).
    *   `work_logs_df`: With 3000-5000 rows. Columns: `log_id` (unique integers), `employee_id` (randomly sampled from `employees_df` IDs), `log_date` (random dates occurring *after* their respective `hire_date`), `projects_contributed` (random integers 0-3 per log event), `training_hours` (random floats 0-8 per log event), `sick_days_taken` (random floats 0-1 per log event, mostly 0).
    *   `satisfaction_surveys_df`: With 800-1200 rows. Columns: `survey_id` (unique integers), `employee_id` (randomly sampled from `employees_df` IDs), `survey_date` (random dates occurring *after* their respective `hire_date`), `job_satisfaction_score` (random integers 1-10), `work_life_balance_score` (random integers 1-10).
    *   `termination_events_df`: With 50-100 rows. Columns: `termination_id` (unique integers), `employee_id` (randomly sampled from `employees_df` IDs), `termination_date` (random dates *after* `hire_date`). This table represents employees who have churned.
    *   **Simulate realistic churn patterns**: Ensure `log_date` and `survey_date` are after `hire_date`, and `termination_date` is after `hire_date`. Introduce a correlation: employees who eventually churn (i.e., appear in `termination_events_df`) should, *before their termination_date or a general cutoff date*, exhibit patterns like: lower average `job_satisfaction_score` and `work_life_balance_score`, fewer `projects_contributed` or `training_hours`, and potentially more `sick_days_taken`. Some departments/role levels might have higher churn. Sort `work_logs_df` and `satisfaction_surveys_df` by `employee_id` then `date`.

2. **Load into SQLite & SQL Feature Engineering (Employee Pre-Churn Behavior)**: Create an in-memory SQLite database using `sqlite3`. Load `employees_df`, `work_logs_df`, and `satisfaction_surveys_df` into tables named `employees`, `work_logs`, and `satisfaction_surveys` respectively. Determine a `global_analysis_date` (e.g., `max(log_date)` from `work_logs_df` + 60 days, using pandas) and a `feature_cutoff_date` (`global_analysis_date` - 90 days).
    Write a single SQL query that performs the following for *each employee*, aggregating their work activity and survey responses *before* the `feature_cutoff_date`:
    *   **Joins** `employees` with aggregated subqueries for `work_logs` and `satisfaction_surveys`.
    *   **Aggregates features based on activity *before* `feature_cutoff_date`**:
        *   `total_projects_completed_pre_cutoff` (sum of `projects_contributed`)
        *   `total_training_hours_pre_cutoff` (sum of `training_hours`)
        *   `total_sick_leave_days_pre_cutoff` (sum of `sick_days_taken`)
        *   `num_work_log_entries_pre_cutoff` (count of `log_id`s)
        *   `avg_job_satisfaction_pre_cutoff` (average `job_satisfaction_score`)
        *   `avg_work_life_balance_pre_cutoff` (average `work_life_balance_score`)
        *   `num_survey_responses_pre_cutoff` (count of `survey_id`s)
        *   `days_since_last_activity_pre_cutoff`: Number of days between `feature_cutoff_date` and the maximum of `MAX(log_date)` and `MAX(survey_date)` for the employee (only considering activities before `feature_cutoff_date`).
    *   **Includes static employee attributes**: `employee_id`, `hire_date`, `department`, `role_level`, `is_remote`.
    *   **Ensures** all employees are included (using `LEFT JOIN`), showing 0 for counts/sums, 0.0 for averages, and `NULL` for `days_since_last_activity_pre_cutoff` if no activity before cutoff.
    *   The query should return `employee_id`, `hire_date`, `department`, `role_level`, `is_remote`, and all the aggregated features.

3. **Pandas Feature Engineering & Binary Target Creation**: Fetch the SQL query results into a pandas DataFrame (`employee_churn_features_df`).
    *   Handle `NaN` values: Fill `total_projects_completed_pre_cutoff`, `total_training_hours_pre_cutoff`, `total_sick_leave_days_pre_cutoff`, `num_work_log_entries_pre_cutoff`, `num_survey_responses_pre_cutoff` with 0. Fill `avg_job_satisfaction_pre_cutoff` and `avg_work_life_balance_pre_cutoff` with 0.0 (or a neutral score like 5.0). For `days_since_last_activity_pre_cutoff` (for employees with no activities before cutoff), fill with a large sentinel value (e.g., 9999 days).
    *   Convert `hire_date` to datetime objects. Calculate `tenure_at_cutoff_days`: The number of days between `hire_date` and the `feature_cutoff_date`.
    *   Calculate `work_activity_frequency_pre_cutoff`: `num_work_log_entries_pre_cutoff` / (`tenure_at_cutoff_days` + 1). Use `+1` to prevent division by zero for very new employees.
    *   Calculate `avg_sick_days_per_year_pre_cutoff`: (`total_sick_leave_days_pre_cutoff` * 365) / (`tenure_at_cutoff_days` + 1). Handles very new employees.
    *   **Create the Binary Target `will_churn_in_next_90_days`**: For each employee, check the `termination_events_df` for any `termination_date` that occurs *between* `feature_cutoff_date` and `feature_cutoff_date + timedelta(days=90)`. Create a binary column (1 if churned, 0 otherwise). Perform a left merge, filling `NaN`s from the merge with 0.
    *   Define features `X` (all numerical and categorical features engineered) and target `y` (`will_churn_in_next_90_days`). Split into training and testing sets (e.g., 70/30 split) using `sklearn.model_selection.train_test_split` (set `random_state=42`, `stratify` on `y` for class balance).

4. **Data Visualization**: Create two separate plots to visually inspect relationships with `will_churn_in_next_90_days`:
    *   A violin plot (or box plot) showing the distribution of `total_sick_leave_days_pre_cutoff` for churned (1) vs. non-churned (0) employees.
    *   A stacked bar chart showing the proportion of `will_churn_in_next_90_days` (0 or 1) across different `department` values.
    Ensure plots have appropriate labels and titles.

5. **ML Pipeline & Evaluation (Binary Classification)**: 
    *   Create an `sklearn.pipeline.Pipeline` with a `sklearn.compose.ColumnTransformer` for preprocessing:
        *   For numerical features (e.g., `tenure_at_cutoff_days`, `total_projects_completed_pre_cutoff`, `total_training_hours_pre_cutoff`, `total_sick_leave_days_pre_cutoff`, `num_work_log_entries_pre_cutoff`, `avg_job_satisfaction_pre_cutoff`, `avg_work_life_balance_pre_cutoff`, `num_survey_responses_pre_cutoff`, `days_since_last_activity_pre_cutoff`, `work_activity_frequency_pre_cutoff`, `avg_sick_days_per_year_pre_cutoff`): Apply `sklearn.preprocessing.SimpleImputer(strategy='mean')` followed by `sklearn.preprocessing.StandardScaler`.
        *   For categorical features (`department`, `role_level`, `is_remote`): Apply `sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')`.
    *   The final estimator in the pipeline should be `sklearn.ensemble.HistGradientBoostingClassifier` (set `random_state=42`).
    *   Train the pipeline on `X_train`, `y_train`. Predict probabilities for the positive class (class 1) on the test set (`X_test`).
    *   Calculate and print the `sklearn.metrics.roc_auc_score` and a `sklearn.metrics.classification_report` for the test set predictions.

## Focus
Predicting Employee Turnover (Churn) based on historical work activity and satisfaction survey data, using time-based feature engineering and a binary classification ML pipeline.

## Dataset
Synthetic employee activity logs, satisfaction surveys, and termination records.

## Hint
When simulating churn patterns, consider creating 'churn-prone' employees with specific characteristics (e.g., lower satisfaction scores, fewer contributions) and ensuring their termination dates fall within the target prediction window. For the SQL `days_since_last_activity_pre_cutoff`, you'll need to use `MAX()` across both `log_date` and `survey_date` within a subquery filtered by the `feature_cutoff_date`, and `julianday()` for date differences in days.
