# AI Daily Lab — 2026-05-29

## Task
Develop a machine learning pipeline to predict the **post-interaction customer satisfaction score category** ('Low', 'Medium', 'High') for customer support interactions, based on interaction metadata, agent performance, and the customer's historical satisfaction and interaction patterns up to a specific cutoff date.

1.  **Generate Synthetic Data (Pandas/Numpy)**: Create three pandas DataFrames:
    *   `customers_df`: With 1000-1500 rows. Columns: `customer_id` (unique integers), `signup_date` (random dates over the last 3-5 years), `region` (e.g., 'North', 'South', 'East', 'West'), `customer_segment` (e.g., 'New', 'Regular', 'VIP', 'Churn Risk').
    *   `agents_df`: With 100-200 rows. Columns: `agent_id` (unique integers), `department` (e.g., 'Billing', 'Tech Support', 'Sales'), `agent_seniority` (e.g., 'Junior', 'Mid', 'Senior'), `average_handle_time_minutes` (random floats 10.0-60.0), `past_satisfaction_rating` (random floats 1.0-5.0).
    *   `interactions_df`: With 20000-30000 rows. Columns: `interaction_id` (unique integers), `customer_id` (randomly sampled from `customers_df` IDs), `agent_id` (randomly sampled from `agents_df` IDs), `interaction_date` (random datetimes occurring *after* their respective `signup_date`), `channel` (e.g., 'Phone', 'Chat', 'Email'), `issue_type` (e.g., 'Billing Inquiry', 'Technical Issue', 'Account Update', 'Product Info'), `interaction_duration_minutes` (random floats 5.0-90.0), `post_interaction_satisfaction_score` (target variable: random integers 1-5).
    *   **Simulate realistic patterns**: Ensure `interaction_date` is always after `signup_date`. Higher `agent_seniority` and `past_satisfaction_rating` should correlate with higher `post_interaction_satisfaction_score`. 'VIP' `customer_segment` might have higher scores. 'Technical Issue' `issue_type` might correlate with slightly lower scores. Introduce a slight imbalance in `post_interaction_satisfaction_score` (e.g., more 4s and 5s, fewer 1s and 2s). `interaction_duration_minutes` might vary by `channel` or `issue_type`.
    *   Sort `interactions_df` by `interaction_date`.

2.  **Load into SQLite & SQL Feature Engineering (Time-Windowed Aggregations)**: Create an in-memory SQLite database using `sqlite3`. Load `customers_df`, `agents_df`, and `interactions_df` into tables named `customers`, `agents`, and `interactions` respectively.
    *   Define `GLOBAL_PREDICTION_CUTOFF_DATE` as 2 weeks prior to the latest `interaction_date` in your generated `interactions_df` (e.g., `interactions_df['interaction_date'].max() - pd.Timedelta(weeks=2)`).
    *   Write a single SQL query that performs the following for *each interaction that occurred AFTER `GLOBAL_PREDICTION_CUTOFF_DATE`*:
        *   Joins the `interactions` (filtered for events after cutoff) with the `customers` and `agents` tables.
        *   Aggregates historical features *for the respective `customer_id` up to and including `GLOBAL_PREDICTION_CUTOFF_DATE`* (NOT including the current interaction):
            *   `avg_satisfaction_customer_prev_90d`: Average `post_interaction_satisfaction_score` for this `customer_id` in the 90 days *prior to or on* `GLOBAL_PREDICTION_CUTOFF_DATE`.
            *   `num_interactions_customer_prev_90d`: Count of interactions for this `customer_id` in the 90 days *prior to or on* `GLOBAL_PREDICTION_CUTOFF_DATE`.
            *   `days_since_last_interaction_customer_at_cutoff`: Number of days between `GLOBAL_PREDICTION_CUTOFF_DATE` and the most recent `interaction_date` for this `customer_id` *before or on* the cutoff. Return a large number (e.g., 9999) if no prior interactions.
        *   Extracts time-based features from the `interaction_date` of the *current* interaction (e.g., `day_of_week`, `hour_of_day`, `month_of_year`).
        *   Includes static attributes: `interaction_id`, `customer_id`, `agent_id`, `interaction_date`, `channel`, `issue_type`, `interaction_duration_minutes`, `signup_date`, `region`, `customer_segment`, `department`, `agent_seniority`, `average_handle_time_minutes`, `past_satisfaction_rating`, and the target `post_interaction_satisfaction_score` for the *current* interaction.
    *   **Ensures** all interactions *after* the cutoff are included. Handle `NULL`s for historical aggregates (e.g., 0.0 for averages, 0 for counts if no prior activity).
    *   The query should return all mentioned fields.
    *   **Hint**: Use CTEs for pre-calculating customer-level historical aggregates up to `GLOBAL_PREDICTION_CUTOFF_DATE`, then join these with the future interactions. Use `julianday()` for date comparisons.

3.  **Pandas Feature Engineering & Multi-class Target Creation**: Fetch the SQL query results into a pandas DataFrame (`interaction_features_df`).
    *   Convert all relevant date/datetime columns (`signup_date`, `interaction_date`) to appropriate types.
    *   Handle `NaN` values: Fill numerical historical aggregated features (e.g., averages, counts) with 0.0 or 0 as appropriate. Fill `days_since_last_interaction_customer_at_cutoff` with 9999.
    *   Calculate `customer_tenure_at_interaction_days`: Number of days between `signup_date` and `interaction_date` (for the current interaction).
    *   **Create the Multi-class Target `satisfaction_category`**: Based on `post_interaction_satisfaction_score`:
        *   'Low': `post_interaction_satisfaction_score` <= 2
        *   'Medium': `post_interaction_satisfaction_score` == 3
        *   'High': `post_interaction_satisfaction_score` >= 4
    *   Define features `X` (numerical: `interaction_duration_minutes`, `average_handle_time_minutes`, `past_satisfaction_rating`, `avg_satisfaction_customer_prev_90d`, `num_interactions_customer_prev_90d`, `days_since_last_interaction_customer_at_cutoff`, `day_of_week`, `hour_of_day`, `month_of_year`, `customer_tenure_at_interaction_days`; categorical: `region`, `customer_segment`, `channel`, `issue_type`, `department`, `agent_seniority`) and target `y` (`satisfaction_category`).
    *   Split into training and testing sets (e.g., 70/30 split) using `sklearn.model_selection.train_test_split` (set `random_state=42`, `stratify` on `y` for class balance).

4.  **Data Visualization (Matplotlib/Seaborn)**: Create two separate plots to visually inspect relationships with `satisfaction_category`:
    *   A violin plot (or box plot) showing the distribution of `interaction_duration_minutes` for each `satisfaction_category`. Ensure appropriate labels and titles.
    *   A stacked bar chart showing the proportion of `satisfaction_category` (across 'Low', 'Medium', 'High') for different `channel` values. Ensure appropriate labels and titles.

5.  **ML Pipeline & Evaluation (Multi-class Classification)**:
    *   Create an `sklearn.pipeline.Pipeline` with a `sklearn.compose.ColumnTransformer` for preprocessing:
        *   For numerical features: Apply `sklearn.preprocessing.SimpleImputer(strategy='mean')` followed by `sklearn.preprocessing.StandardScaler`.
        *   For categorical features: Apply `sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')`.
    *   The final estimator in the pipeline should be `sklearn.ensemble.HistGradientBoostingClassifier` (set `random_state=42`).
    *   Train the pipeline on `X_train`, `y_train`. Predict the `satisfaction_category` on the test set (`X_test`).
    *   Calculate and print a `sklearn.metrics.classification_report` for the test set predictions.

## Focus
Predicting multi-class customer satisfaction based on interaction metadata, agent performance, and time-windowed historical customer behavior using SQL for feature engineering.

## Dataset
Synthetic customer, agent, and interaction data.

## Hint
Pay close attention to the `GLOBAL_PREDICTION_CUTOFF_DATE` in SQL. Ensure historical aggregates for a customer only consider interactions *before or on* the cutoff, while the target interaction is *after* the cutoff. Use `julianday()` for date arithmetic in SQLite for time differences and windowing. Handle `NULL`s diligently in both SQL with `COALESCE` and Pandas for robust features.
