# AI Daily Lab — 2026-06-20

## Task
Develop a machine learning pipeline to predict if a patient will **no-show** for their scheduled appointment (binary classification), based on their demographic profile, appointment details, and historical attendance patterns up to a specific cutoff date.

## Focus
Predicting patient no-show rates for healthcare appointment optimization.

## Dataset
1.  **Generate Synthetic Data (Pandas/Numpy)**: Create three pandas DataFrames:
    *   `patients_df`: With 1000-1500 rows. Columns: `patient_id` (unique integers), `age` (random integers 18-90), `gender` (e.g., 'Male', 'Female', 'Other'), `existing_condition` (e.g., 'None', 'Chronic', 'Acute'), `signup_date` (random dates over the last 5-10 years).
    *   `doctors_df`: With 50-100 rows. Columns: `doctor_id` (unique integers), `specialty` (e.g., 'Cardiology', 'Pediatrics', 'Dermatology', 'General', 'Orthopedics'), `doctor_experience_years` (random integers 3-30), `doctor_rating` (random floats 3.0-5.0).
    *   `appointments_df`: With 30000-50000 rows. Columns: `appointment_id` (unique integers), `patient_id` (randomly sampled from `patients_df` IDs), `doctor_id` (randomly sampled from `doctors_df` IDs), `appointment_datetime` (random datetimes, *after* respective `signup_date` for the `patient_id`, up to `pd.Timestamp.now()`), `_attended` (binary: 1 for attended, 0 for no-show - simulate ~15-25% no-shows, with higher rates for patients with past no-shows, lower `doctor_rating`, or certain `existing_condition`).
    *   **Simulate realistic patterns**: Ensure `appointment_datetime` is after `signup_date`. Patients with `Chronic` conditions might have more appointments. Doctors with higher `doctor_rating` might have fewer no-shows. Patients who have no-showed in the past are more likely to no-show again. Ensure a good mix of specialties and patient conditions. Sort `appointments_df` by `patient_id` then `appointment_datetime`.

2.  **Load into SQLite & SQL Feature Engineering (Time-Windowed Aggregations)**: Create an in-memory SQLite database using `sqlite3`. Load `patients_df`, `doctors_df`, and `appointments_df` into tables named `patients`, `doctors`, and `appointments` respectively.
    *   Define `GLOBAL_PREDICTION_CUTOFF_DATE` as 7 days prior to the latest `appointment_datetime` in your generated `appointments_df`.
    *   Write a single SQL query that performs the following for *each appointment scheduled AFTER `GLOBAL_PREDICTION_CUTOFF_DATE`*:
        *   Joins `appointments` (filtered for events after cutoff) with `patients` and `doctors` tables.
        *   Aggregates historical features based on *appointments up to and including `GLOBAL_PREDICTION_CUTOFF_DATE`*:
            *   `num_past_appointments_patient_prev_90d`: Count of appointments for the specific `patient_id` in the 90 days ending at `GLOBAL_PREDICTION_CUTOFF_DATE`.
            *   `patient_no_show_rate_prev_90d`: Average of `(1 - _attended)` for the `patient_id` in the 90 days ending at `GLOBAL_PREDICTION_CUTOFF_DATE` (i.e., proportion of no-shows).
            *   `days_since_last_appointment_at_cutoff`: Number of days between `GLOBAL_PREDICTION_CUTOFF_DATE` and the most recent `appointment_datetime` for this `patient_id` *before or on* the cutoff. Return a large number (e.g., 9999) if no prior appointments.
            *   `doctor_avg_no_show_rate_prev_90d`: Average of `(1 - _attended)` for the specific `doctor_id` in the 90 days ending at `GLOBAL_PREDICTION_CUTOFF_DATE`.
        *   Extracts time-based features from the `appointment_datetime` of the *current* future appointment: `appointment_day_of_week` (0-6), `appointment_hour_of_day` (0-23).
        *   Includes static attributes: `appointment_id`, `patient_id`, `age`, `gender`, `existing_condition`, `doctor_id`, `specialty`, `doctor_experience_years`, `doctor_rating`, and the target `_attended` for the *current* future appointment.
    *   **Ensures** all appointments *after* the cutoff are included. Handle `NULL`s for historical aggregates (e.g., 0.0 for averages, 0 for counts if no prior activity, 9999 for days since last activity).
    *   The query should return all mentioned fields.

3.  **Pandas Feature Engineering & Binary Target Creation**: Fetch the SQL query results into a pandas DataFrame (`appointment_features_df`).
    *   Convert `appointment_datetime` and `signup_date` (from original `patients_df` if needed for tenure) to datetime objects. You'll also need the `GLOBAL_PREDICTION_CUTOFF_DATE` in pandas for tenure calculation.
    *   Handle `NaN` values: Fill numerical historical aggregated features (e.g., rates, counts) with 0.0 or 0 as appropriate. Fill `days_since_last_appointment_at_cutoff` with 9999. Fill `age`, `doctor_experience_years`, `doctor_rating` with their respective means.
    *   Calculate `patient_tenure_at_cutoff_days`: Number of days between `signup_date` and `GLOBAL_PREDICTION_CUTOFF_DATE`.
    *   **Create the Binary Target `will_no_show`**: Based on the `_attended` column for the *current* appointment: `1` if `_attended` is 0 (no-show), `0` if `_attended` is 1 (attended).
    *   Define features `X` (numerical: `age`, `doctor_experience_years`, `doctor_rating`, `num_past_appointments_patient_prev_90d`, `patient_no_show_rate_prev_90d`, `days_since_last_appointment_at_cutoff`, `doctor_avg_no_show_rate_prev_90d`, `appointment_day_of_week`, `appointment_hour_of_day`, `patient_tenure_at_cutoff_days`; categorical: `gender`, `existing_condition`, `specialty`) and target `y` (`will_no_show`).
    *   Split into training and testing sets (e.g., 70/30 split) using `sklearn.model_selection.train_test_split` (set `random_state=42`, `stratify` on `y` for class balance).

4.  **Data Visualization (Matplotlib/Seaborn)**: Create two separate plots to visually inspect relationships with `will_no_show`:
    *   A violin plot (or box plot) showing the distribution of `days_since_last_appointment_at_cutoff` for 'Attended' (0) vs. 'No-Show' (1) appointments. Ensure appropriate labels and titles.
    *   A stacked bar chart showing the proportion of `will_no_show` (0 or 1) across different `specialty` values. Ensure appropriate labels and titles.

5.  **ML Pipeline & Evaluation (Binary Classification)**:
    *   Create an `sklearn.pipeline.Pipeline` with a `sklearn.compose.ColumnTransformer` for preprocessing:
        *   For numerical features: Apply `sklearn.preprocessing.SimpleImputer(strategy='mean')` followed by `sklearn.preprocessing.StandardScaler`.
        *   For categorical features: Apply `sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')`.
    *   The final estimator in the pipeline should be `sklearn.ensemble.HistGradientBoostingClassifier` (set `random_state=42`, and consider `class_weight='balanced'` due to potential target imbalance).
    *   Train the pipeline on `X_train`, `y_train`. Predict probabilities for the positive class (class 1) on the test set (`X_test`).
    *   Calculate and print the `sklearn.metrics.roc_auc_score` and a `sklearn.metrics.classification_report` for the test set predictions.

## Hint
When simulating `_attended` in `appointments_df`, make no-show probability higher if the patient has a 'Chronic' condition, a lower `doctor_rating`, or has a history of no-shows (e.g., using a cumulative count of past no-shows). For SQL date comparisons, use `julianday()` and `CAST` date strings carefully. Remember to handle potential division by zero when calculating rates (e.g., using `CASE WHEN COUNT(...) > 0 THEN AVG(...) ELSE 0 END`).
