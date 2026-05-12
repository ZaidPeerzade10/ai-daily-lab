# AI Daily Lab — 2026-05-12

## Task
Develop a machine learning pipeline to predict if a sales lead will convert (make a purchase/sign-up) within the next 30 days, based on their demographic information and recent interaction history.

## Focus
Predictive modeling for sales lead conversion using historical interaction data and lead attributes, involving SQL-based time-windowed feature engineering, synthetic data generation, data visualization, and standard ML evaluation.

## Dataset
Synthetic data: `leads_df` (lead profiles including `conversion_date`) and `interactions_df` (lead interaction timestamps, types, and duration).

## Hint
1.  **Synthetic Data Generation (Pandas/Numpy)**: Create `leads_df` (1000-1500 rows) with `lead_id`, `signup_date` (random dates over last 2-4 years), `source` (e.g., 'Webinar', 'Referral', 'Paid Ad', 'Cold Call'), `industry` (e.g., 'Tech', 'Finance', 'Retail'), `lead_score_initial` (random integers 1-100), and `conversion_date` (for ~15-20% of leads, occurring after `signup_date` and within the last 18 months, `NaT` otherwise). Create `interactions_df` (3000-5000 rows) with `interaction_id`, `lead_id` (randomly sampled), `interaction_date` (after respective `signup_date` and before `conversion_date` if applicable), `interaction_type` (e.g., 'Email Open', 'Demo Request', 'Website Visit', 'Call', 'Content Download'), `duration_minutes` (random floats 1-60, mostly for 'Call' or 'Demo Request'). Simulate realistic patterns: higher `lead_score_initial` or 'Demo Request' interactions indicate higher conversion probability. Sort `interactions_df` by `lead_id` then `interaction_date`.
2.  **Load into SQLite & SQL Feature Engineering**: Create an in-memory SQLite database. Load `leads_df` and `interactions_df` into tables named `leads` and `interactions`. Define `GLOBAL_PREDICTION_CUTOFF_DATE` as 30 days prior to the latest `interaction_date` in your generated data. Write a single SQL query that, for *each lead*, aggregates their interaction history *within the 30 days immediately preceding `GLOBAL_PREDICTION_CUTOFF_DATE`*:
    *   `current_cutoff_date`
    *   `num_interactions_prev_30d` (count of interactions).
    *   `days_since_last_interaction_at_cutoff`: Days between `GLOBAL_PREDICTION_CUTOFF_DATE` and most recent `interaction_date` before or on cutoff (use 9999 if no interactions).
    *   `avg_interaction_duration_prev_30d` (average `duration_minutes`).
    *   `num_unique_interaction_types_prev_30d`.
    *   Include static lead attributes: `lead_id`, `signup_date`, `source`, `industry`, `lead_score_initial`. Use `LEFT JOIN` to ensure all leads are included, handling `NULL`s for leads with no interactions in the window (e.g., `COALESCE(SUM(...), 0)`).
3.  **Pandas Feature Engineering & Binary Target Creation**: Fetch SQL results into a pandas DataFrame (`lead_features_df`). Convert date columns to datetime. Handle `NaN` values: fill numerical aggregated features (e.g., `num_interactions_prev_30d`) with 0 or 0.0. Fill `days_since_last_interaction_at_cutoff` with 9999. Calculate `lead_age_at_cutoff_days` (days between `signup_date` and `current_cutoff_date`). Create the binary target `will_convert_in_next_30_days` by checking if a lead's `conversion_date` (merged back from original `leads_df`) falls within the 30-day period *after* `GLOBAL_PREDICTION_CUTOFF_DATE`. Define features `X` (numerical: `lead_score_initial`, `num_interactions_prev_30d`, `days_since_last_interaction_at_cutoff`, `avg_interaction_duration_prev_30d`, `num_unique_interaction_types_prev_30d`, `lead_age_at_cutoff_days`; categorical: `source`, `industry`) and target `y`. Split into training and testing sets with `stratify=y`.
4.  **Data Visualization**: Create a violin plot showing `days_since_last_interaction_at_cutoff` distributions for converting vs. non-converting leads. Create a stacked bar chart showing the proportion of conversion across different `source` values.
5.  **ML Pipeline & Evaluation (Binary Classification)**: Build an `sklearn.pipeline.Pipeline` with a `ColumnTransformer` for preprocessing (numerical: `SimpleImputer(strategy='mean')` + `StandardScaler`; categorical: `OneHotEncoder(handle_unknown='ignore')`). The final estimator should be `sklearn.ensemble.HistGradientBoostingClassifier(random_state=42)`. Train on `X_train`, `y_train`. Predict probabilities for the positive class on `X_test`. Evaluate performance using `roc_auc_score` and `classification_report`.
