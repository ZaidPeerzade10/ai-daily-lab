# AI Daily Lab — 2026-06-13

## Task
Develop a machine learning pipeline to predict if a customer will **churn** (cancel their subscription) within the next 30 days, based on their subscription plan, usage patterns, and historical activity up to a specific cutoff date.

1.  **Generate Synthetic Data (Pandas/Numpy)**: Create two pandas DataFrames:
    *   `customers_df`: With 1000-1500 rows. Columns: `customer_id` (unique integers), `signup_date` (random dates over the last 3-5 years), `region` (e.g., 'North', 'South', 'East', 'West'), `subscription_plan` (e.g., 'Basic', 'Standard', 'Premium'), `monthly_price` (random floats, e.g., $9.99-$49.99, positively correlated with `subscription_plan`), `actual_churn_date` (for ~15-25% of customers, a random date *after* `signup_date` and within the general data range; `pd.NaT` for non-churners).
    *   `usage_events_df`: With 50000-70000 rows. Columns: `event_id` (unique integers), `customer_id` (randomly sampled from `customers_df` IDs), `event_date` (random datetimes, *after* respective `signup_date` for the `customer_id`, and *before* `actual_churn_date` if applicable, up to `pd.Timestamp.now()`), `usage_minutes` (random floats 5.0-600.0), `num_logins` (random integers 1-20).
    *   **Simulate realistic patterns**: Ensure `event_date` is always after `signup_date` and before `actual_churn_date` for churned customers. `Premium` plan customers should generally have higher `usage_minutes` and `num_logins`. Customers with an `actual_churn_date` should exhibit *decreasing* `usage_minutes` and `num_logins` in the months leading up to their `actual_churn_date`. Sort `usage_events_df` by `customer_id` then `event_date`.

2.  **Load into SQLite & SQL Feature Engineering (Time-Windowed Aggregations)**: Create an in-memory SQLite database using `sqlite3`. Load `customers_df` and `usage_events_df` into tables named `customers` and `usage` respectively.
    *   Define `GLOBAL_PREDICTION_CUTOFF_DATE` as 30 days prior to the latest `event_date` in your generated `usage_events_df`.
    *   Write a single SQL query that performs the following for *each customer*, aggregating their behavior *up to and including `GLOBAL_PREDICTION_CUTOFF_DATE`*:
        *   `customer_id`, `signup_date`, `region`, `subscription_plan`, `monthly_price`.
        *   `avg_usage_prev_30d`: Average `usage_minutes` for the customer in the 30 days ending at `GLOBAL_PREDICTION_CUTOFF_DATE`.
        *   `num_logins_prev_30d`: Sum of `num_logins` for the customer in the 30 days ending at `GLOBAL_PREDICTION_CUTOFF_DATE`.
        *   `days_since_last_activity_at_cutoff`: Number of days between `GLOBAL_PREDICTION_CUTOFF_DATE` and the most recent `event_date` for this `customer_id` *before or on* the cutoff. Return a large number (e.g., 9999) if no activity before cutoff.
        *   `total_usage_since_signup`: Sum of `usage_minutes` for the customer from `signup_date` to `GLOBAL_PREDICTION_CUTOFF_DATE`.
        *   `num_activity_events_prev_90d`: Count of `event_id`s for the customer in the 90 days ending at `GLOBAL_PREDICTION_CUTOFF_DATE`.
        *   Include `actual_churn_date` from the `customers` table for target creation later.
    *   **Ensures** all customers are included (using `LEFT JOIN`), showing 0 for counts/sums and 0.0 for averages if no activity in the window. Handle `NULL`s appropriately.
    *   The query should return all mentioned fields.
    *   **Hint**: Use CTEs for pre-calculating customer-level historical aggregates up to `GLOBAL_PREDICTION_CUTOFF_DATE`. Use `julianday()` for date comparisons.

3.  **Pandas Feature Engineering & Binary Target Creation**: Fetch the SQL query results into a pandas DataFrame (`customer_features_df`).
    *   Convert all relevant date/datetime columns (`signup_date`, `actual_churn_date`) to appropriate types. Note: `GLOBAL_PREDICTION_CUTOFF_DATE` needs to be defined in Pandas for target creation, potentially by extracting it from the latest `event_date` in the original `usage_events_df` again.
    *   Handle `NaN` values: Fill numerical historical aggregated features (e.g., averages, sums, counts) with 0.0 or 0 as appropriate. Fill `days_since_last_activity_at_cutoff` with 9999.
    *   Calculate `customer_tenure_at_cutoff_days`: Number of days between `signup_date` and `GLOBAL_PREDICTION_CUTOFF_DATE`.
    *   **Create the Binary Target `will_churn_next_30d`**: For *each customer*, assign 1 if their `actual_churn_date` falls *after* `GLOBAL_PREDICTION_CUTOFF_DATE` and *on or before* `GLOBAL_PREDICTION_CUTOFF_DATE + pd.Timedelta(days=30)`. Assign 0 otherwise.
    *   Define features `X` (numerical: `monthly_price`, `avg_usage_prev_30d`, `num_logins_prev_30d`, `days_since_last_activity_at_cutoff`, `total_usage_since_signup`, `num_activity_events_prev_90d`, `customer_tenure_at_cutoff_days`; categorical: `region`, `subscription_plan`) and target `y` (`will_churn_next_30d`).
    *   Split into training and testing sets (e.g., 70/30 split) using `sklearn.model_selection.train_test_split` (set `random_state=42`, `stratify` on `y` for class balance).

4.  **Data Visualization (Matplotlib/Seaborn)**: Create two separate plots to visually inspect relationships with `will_churn_next_30d`:
    *   A violin plot (or box plot) showing the distribution of `customer_tenure_at_cutoff_days` for 'Not Churn' (0) vs. 'Will Churn' (1) customers. Ensure appropriate labels and titles.
    *   A stacked bar chart showing the proportion of `will_churn_next_30d` (0 or 1) across different `subscription_plan` values. Ensure appropriate labels and titles.

5.  **ML Pipeline & Evaluation (Binary Classification)**:
    *   Create an `sklearn.pipeline.Pipeline` with a `sklearn.compose.ColumnTransformer` for preprocessing:
        *   For numerical features: Apply `sklearn.preprocessing.SimpleImputer(strategy='mean')` followed by `sklearn.preprocessing.StandardScaler`.
        *   For categorical features: Apply `sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')`.
    *   The final estimator in the pipeline should be `sklearn.ensemble.HistGradientBoostingClassifier` (set `random_state=42`, and consider `class_weight='balanced'` due to potential target imbalance).
    *   Train the pipeline on `X_train`, `y_train`. Predict probabilities for the positive class (class 1) on the test set (`X_test`).
    *   Calculate and print the `sklearn.metrics.roc_auc_score` and a `sklearn.metrics.classification_report` for the test set predictions.

## Focus
Subscription Churn Prediction (Binary Classification) with Time-Windowed Feature Engineering

## Dataset
Synthetic customer, subscription, and usage event data.

## Hint
Pay close attention to simulating realistic churn dates for the synthetic data and defining the `will_churn_next_30d` target variable based on a future window relative to the `GLOBAL_PREDICTION_CUTOFF_DATE`. Ensure SQL aggregations are strictly time-windowed to prevent data leakage and handle potential `NULL`s from `LEFT JOIN`s for customers with no historical activity.
