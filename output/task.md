# AI Daily Lab — 2026-02-17

## Task
1. **Generate Synthetic Data (Pandas/Numpy)**: Create four pandas DataFrames:
    *   `students_df`: With 500-700 rows. Columns: `student_id` (unique integers), `signup_date` (random dates over the last 3 years), `education_level` (e.g., 'High School', 'Undergrad', 'Postgrad'), `country` (e.g., 'USA', 'Canada', 'UK', 'India', 'Australia').
    *   `courses_df`: With 50-100 rows. Columns: `course_id` (unique integers), `course_name` (e.g., 'Python Fundamentals', 'Advanced SQL'), `difficulty` (e.g., 'Beginner', 'Intermediate', 'Advanced'), `category` (e.g., 'Programming', 'Data Science', 'Marketing', 'Design'), `expected_duration_days` (random integers 30-180).
    *   `enrollments_df`: With 800-1200 rows (representing a student enrolling in a specific course). Columns: `enrollment_id` (unique integers), `student_id` (randomly sampled from `students_df` IDs), `course_id` (randomly sampled from `courses_df` IDs), `enrollment_date` (random dates *after* `signup_date`), `is_completed_course` (binary, 0 or 1, with a bias for easier courses/higher education levels to be completed).
    *   `activity_logs_df`: With 5000-8000 rows. Columns: `activity_log_id` (unique integers), `enrollment_id` (randomly sampled from `enrollments_df` IDs), `activity_date` (random dates *after* respective `enrollment_date`), `activity_type` (e.g., 'lecture_view', 'quiz_attempt', 'assignment_submit', 'forum_post'), `time_spent_minutes` (random floats 5-90).
    *   **Simulate Realistic Behavior**: Ensure `activity_date` is always after `enrollment_date`. For `is_completed_course=1`, generate activities that are more frequent, spread out over the `expected_duration_days`, and include more 'assignment_submit'/'quiz_attempt' types. For `is_completed_course=0`, activities should be less frequent, stop earlier than `expected_duration_days`, and have fewer 'submission' types.

2. **Load into SQLite & SQL Feature Engineering (Early Engagement)**: Create an in-memory SQLite database using `sqlite3`. Load `students_df`, `courses_df`, `enrollments_df`, and `activity_logs_df` into tables named `students`, `courses`, `enrollments`, and `activity_logs` respectively. Determine a `global_analysis_date` (e.g., `max(activity_date)` from `activity_logs_df` + 60 days, using pandas) and define an `early_engagement_window_days` (e.g., 30 days).
    Write a single SQL query that performs the following for *each student-course enrollment* (from `enrollments` table), aggregating their activity *within the first `early_engagement_window_days`* after their `enrollment_date` for that specific course:
    *   **Joins** `students`, `courses`, `enrollments`, and `activity_logs` tables.
    *   **Aggregates features based on early activities**: 
        *   `early_total_activities`: Count of all activities within the window.
        *   `early_total_time_spent`: Sum of `time_spent_minutes` within the window.
        *   `early_num_quiz_attempts`: Count of 'quiz_attempt' activities within the window.
        *   `early_num_assignments_submitted`: Count of 'assignment_submit' activities within the window.
        *   `days_from_enroll_to_first_activity`: Number of days between `enrollment_date` and `MIN(activity_date)` (only considering activities within the window). `NULL` if no activity.
        *   `early_activity_frequency`: `early_total_activities` / `early_engagement_window_days` (calculated as `CAST(COUNT(al.activity_log_id) AS REAL) / {early_engagement_window_days}`).
    *   **Includes static enrollment, student, and course attributes**: `enrollment_id`, `student_id`, `course_id`, `enrollment_date`, `education_level`, `country`, `difficulty`, `category`, `expected_duration_days`, `is_completed_course` (the raw target from `enrollments_df`).
    *   **Ensures** all enrollments are included (using a `LEFT JOIN` from `enrollments` to `activity_logs` subquery), showing 0 for counts/sums, 0.0 for averages/frequencies, and `NULL` for `days_from_enroll_to_first_activity` if no early activities.
    *   The query should return `enrollment_id`, `student_id`, `course_id`, `education_level`, `country`, `difficulty`, `category`, `expected_duration_days`, `enrollment_date`, `is_completed_course`, and all the aggregated features.

3. **Pandas Feature Engineering & Binary Target Creation**: Fetch the SQL query results into a pandas DataFrame (`enrollment_features_df`).
    *   Handle `NaN` values: Fill `early_total_activities`, `early_total_time_spent`, `early_num_quiz_attempts`, `early_num_assignments_submitted` with 0. Fill `early_activity_frequency` with 0.0. For `days_from_enroll_to_first_activity` (for enrollments with no early activities), fill with a large sentinel value (e.g., `early_engagement_window_days` + 10 or 40 days).
    *   Convert `enrollment_date` to datetime objects. Calculate `enrollment_age_at_cutoff_days`: The number of days between `enrollment_date` and (`enrollment_date` + `early_engagement_window_days` as a datetime object).
    *   Define features `X` (all numerical and categorical features engineered from early engagement and static info) and target `y` (`is_completed_course`). Split into training and testing sets (e.g., 70/30 split) using `sklearn.model_selection.train_test_split` (set `random_state=42`, `stratify` on `y` for class balance).

4. **Data Visualization**: Create two separate plots to visually inspect relationships with `is_completed_course`:
    *   A violin plot (or box plot) showing the distribution of `early_total_time_spent` for `is_completed_course=0` vs. `is_completed_course=1`.
    *   A stacked bar chart showing the proportion of `is_completed_course` (0 or 1) across different `difficulty` levels.
    Ensure plots have appropriate labels and titles.

5. **ML Pipeline & Evaluation (Binary Classification)**: 
    *   Create an `sklearn.pipeline.Pipeline` with a `sklearn.compose.ColumnTransformer` for preprocessing:
        *   For numerical features (e.g., `expected_duration_days`, `enrollment_age_at_cutoff_days`, `early_total_activities`, `early_total_time_spent`, `early_num_quiz_attempts`, `early_num_assignments_submitted`, `days_from_enroll_to_first_activity`, `early_activity_frequency`): Apply `sklearn.preprocessing.SimpleImputer(strategy='mean')` followed by `sklearn.preprocessing.StandardScaler`.
        *   For categorical features (`education_level`, `country`, `difficulty`, `category`): Apply `sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')`.
    *   The final estimator in the pipeline should be `sklearn.ensemble.HistGradientBoostingClassifier` (set `random_state=42`).
    *   Train the pipeline on `X_train`, `y_train`. Predict probabilities for the positive class (class 1) on the test set (`X_test`).
    *   Calculate and print the `sklearn.metrics.roc_auc_score` and a `sklearn.metrics.classification_report` for the test set predictions.

## Focus
Predicting course completion based on early student engagement features.

## Dataset
Synthetic student enrollment and activity data.

## Hint
In SQL Step 2, ensure the `LEFT JOIN` correctly preserves all `enrollments` records even if they have no early activities. The `WHERE` clause for `activity_logs` should filter `activity_date` to be between `e.enrollment_date` and `DATE(e.enrollment_date, '+' || {early_engagement_window_days} || ' days')`. For calculating `days_from_enroll_to_first_activity` and `early_activity_frequency`, `JULIANDAY` can be useful for date arithmetic. Remember to cast counts to `REAL` for correct float division when calculating frequency.
