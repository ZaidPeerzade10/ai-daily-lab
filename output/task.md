# AI Daily Lab — 2026-03-12

## Task
Develop a machine learning pipeline to predict whether a customer will redeem a specific marketing offer, based on their profile, past behavior, and offer characteristics.

## Focus
Predictive analytics for marketing campaign optimization using customer and offer features, sequential data processing with SQL, and binary classification.

## Dataset
1. **Generate Synthetic Data (Pandas/Numpy)**: Create three pandas DataFrames:
    *   `users_df`: With 500-700 rows. Columns: `user_id` (unique integers), `signup_date` (random dates over the last 5 years), `age` (random integers 18-70), `region` (e.g., 'North', 'South', 'East', 'West'), `email_engagement_score` (random floats 0.0-1.0), `has_premium_plan` (binary, 0 or 1).
    *   `offers_df`: With 50-100 rows. Columns: `offer_id` (unique integers), `offer_type` (e.g., 'Discount', 'Free_Shipping', 'Bonus_Points'), `discount_percent` (random floats 5.0-50.0), `min_purchase_value` (random floats 0.0-200.0), `offer_start_date` (random dates over the last 3 years), `offer_end_date` (random dates 1-30 days after `offer_start_date`).
    *   `offer_interactions_df`: With 8000-12000 rows. Columns: `interaction_id` (unique integers), `user_id` (randomly sampled from `users_df` IDs), `offer_id` (randomly sampled from `offers_df` IDs), `interaction_date` (random dates occurring *after* `signup_date` for the user and *between* `offer_start_date` and `offer_end_date` for the offer), `was_redeemed` (binary, 0 or 1, representing the target).
    *   **Simulate realistic redemption patterns**: Ensure `interaction_date` is valid. Bias `was_redeemed` (overall 5-15% redemption rate) such that:
        *   Users with higher `email_engagement_score` or `has_premium_plan=1` have a higher redemption likelihood.
        *   Offers with higher `discount_percent` or specific `offer_type`s (e.g., 'Discount') have higher redemption rates.
        *   `interaction_date` closer to `offer_start_date` or `offer_end_date` might influence redemption.
        *   Users who have previously redeemed offers are more likely to redeem again.
    *   Sort `offer_interactions_df` by `user_id` then `interaction_date` for easier sequential processing.

2. **Load into SQLite & SQL Feature Engineering (Event-Level Offer Context)**: Create an in-memory SQLite database using `sqlite3`. Load `users_df`, `offers_df`, and `offer_interactions_df` into tables named `users`, `offers`, and `offer_interactions` respectively.
    Write a single SQL query that performs the following for *each offer interaction event* in `offer_interactions`:
    *   **Joins** `users`, `offers`, and `offer_interactions` tables.
    *   **Calculates sequential features based on the user's *prior offer interactions* and the offer's *prior interactions* (across all users), relative to the current `interaction_date`**:
        *   `user_prior_offers_received`: Count of all *previous* offers (any offer_id) for the same user, before the current `interaction_date`.
        *   `user_prior_offers_redeemed`: Count of *previous* offers for the same user that `was_redeemed=1`, before the current `interaction_date`.
        *   `user_prior_redemption_rate`: `user_prior_offers_redeemed` / `user_prior_offers_received` (0.0 if no prior offers received).
        *   `days_since_last_user_redemption`: Number of days between the current `interaction_date` and the user's *most recent prior redemption date*. If no prior redemption, use the number of days between `signup_date` and the current `interaction_date`.
        *   `offer_prior_interactions_all_users`: Count of all *previous* interactions for the *same offer_id*, across all users, before the current `interaction_date`.
        *   `offer_prior_redemptions_all_users`: Count of *previous* interactions for the *same offer_id* that `was_redeemed=1`, across all users, before the current `interaction_date`.
        *   `offer_prior_redemption_rate_all_users`: `offer_prior_redemptions_all_users` / `offer_prior_interactions_all_users` (0.0 if no prior interactions for this offer).
    *   **Includes static user, offer, and current interaction attributes**: `interaction_id`, `user_id`, `offer_id`, `interaction_date`, `was_redeemed` (the target), `age`, `region`, `email_engagement_score`, `has_premium_plan`, `offer_type`, `discount_percent`, `min_purchase_value`, `signup_date`, `offer_start_date`, `offer_end_date`.
    *   The query should return all these attributes and engineered features. Missing values for prior aggregates/dates should be `NULL`.

3. **Pandas Feature Engineering & Binary Target Creation**: Fetch the SQL query results into a pandas DataFrame (`offer_prediction_df`).
    *   Handle `NaN` values: Fill `user_prior_offers_received`, `user_prior_offers_redeemed`, `offer_prior_interactions_all_users`, `offer_prior_redemptions_all_users` with 0. Fill `user_prior_redemption_rate` and `offer_prior_redemption_rate_all_users` with 0.0. For `days_since_last_user_redemption` (for first interactions or no prior redemptions), if `NaN`s remain, fill with the `days_since_signup_at_interaction`.
    *   Convert `signup_date`, `offer_start_date`, `offer_end_date`, `interaction_date` to datetime objects.
    *   Calculate `days_since_signup_at_interaction`: Days between `signup_date` and `interaction_date`.
    *   Calculate `days_into_offer_campaign`: Days between `offer_start_date` and `interaction_date`.
    *   Calculate `offer_total_duration_days`: Days between `offer_end_date` and `offer_start_date`.
    *   Define features `X` (all numerical: `age`, `email_engagement_score`, `discount_percent`, `min_purchase_value`, `user_prior_offers_received`, `user_prior_offers_redeemed`, `user_prior_redemption_rate`, `days_since_last_user_redemption`, `offer_prior_interactions_all_users`, `offer_prior_redemptions_all_users`, `offer_prior_redemption_rate_all_users`, `days_since_signup_at_interaction`, `days_into_offer_campaign`, `offer_total_duration_days`; categorical: `region`, `has_premium_plan`, `offer_type`) and target `y` (`was_redeemed`). Split into training and testing sets (e.g., 70/30 split) using `sklearn.model_selection.train_test_split` (set `random_state=42`, `stratify` on `y` for class balance due to low redemption rate).

4. **Data Visualization**: Create two separate plots to visually inspect relationships with `was_redeemed`:
    *   A violin plot (or box plot) showing the distribution of `user_prior_redemption_rate` for non-redeemed (0) vs. redeemed (1) offers.
    *   A stacked bar chart showing the proportion of `was_redeemed` (0 or 1) across different `offer_type` values.
    Ensure plots have appropriate labels and titles.

5. **ML Pipeline & Evaluation (Binary Classification)**: 
    *   Create an `sklearn.pipeline.Pipeline` with a `sklearn.compose.ColumnTransformer` for preprocessing:
        *   For numerical features: Apply `sklearn.preprocessing.SimpleImputer(strategy='mean')` followed by `sklearn.preprocessing.StandardScaler`.
        *   For categorical features: Apply `sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')`.
    *   The final estimator in the pipeline should be `sklearn.ensemble.HistGradientBoostingClassifier` (set `random_state=42`).
    *   Train the pipeline on `X_train`, `y_train`. Predict probabilities for the positive class (class 1) on the test set (`X_test`).
    *   Calculate and print the `sklearn.metrics.roc_auc_score` and a `sklearn.metrics.classification_report` for the test set predictions.

## Hint
For SQL sequential features (`user_prior_...`, `offer_prior_...`, `days_since_last_...`), use window functions with `LAG()` and `SUM(CASE WHEN ... END) OVER (PARTITION BY ... ORDER BY ... ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)`. Use `julianday()` for date differences. Be mindful of handling division by zero for rates and `NULL` values for initial events using `COALESCE` or `IFNULL` in SQL.
