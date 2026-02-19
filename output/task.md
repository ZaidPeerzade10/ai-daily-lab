# AI Daily Lab — 2026-02-19

## Task
1. **Generate Synthetic Data (Pandas/Numpy)**: Create three pandas DataFrames:
    *   `users_df`: With 500-700 rows. Columns: `user_id` (unique integers), `signup_date` (random dates over the last 5 years), `age` (random integers 18-70), `region` (e.g., 'North', 'South', 'East', 'West'), `browsing_frequency_level` (e.g., 'Low', 'Medium', 'High').
    *   `offers_df`: With 50-100 rows. Columns: `offer_id` (unique integers), `offer_type` (e.g., 'Discount_10', 'Free_Shipping', 'Bundle_Deal', 'Gift_Card'), `category_focus` (e.g., 'Electronics', 'Books', 'Clothing', 'HomeGoods', 'Services'), `discount_percentage` (random floats 5.0-30.0).
    *   `campaign_exposures_df`: With 5000-8000 rows. Columns: `exposure_id` (unique integers), `user_id` (randomly sampled from `users_df` IDs), `offer_id` (randomly sampled from `offers_df` IDs), `exposure_date` (random dates occurring *after* their respective `signup_date`), `was_converted` (binary, 0 or 1).
    *   **Simulate realistic conversion patterns**: Ensure `exposure_date` is always after `signup_date`. Bias `was_converted` (overall 5-10% conversion rate) such that:
        *   Users with 'High' `browsing_frequency_level` have a higher chance of converting.
        *   Some `offer_type`s (e.g., 'Discount_10', 'Free_Shipping') have generally higher conversion rates.
        *   A subtle correlation where younger users (`age` < 35) might convert more for 'Electronics' or 'Gaming' `category_focus` offers, and older users for 'HomeGoods' or 'Services'.
        *   Sort `campaign_exposures_df` by `user_id` then `exposure_date` for easier sequential processing.

2. **Load into SQLite & SQL Feature Engineering (Event-Level Context)**: Create an in-memory SQLite database using `sqlite3`. Load `users_df`, `offers_df`, and `campaign_exposures_df` into tables named `users`, `offers`, and `exposures` respectively.
    Write a single SQL query that performs the following for *each exposure event* in `exposures`:
    *   **Joins** `users`, `offers`, and `exposures` tables.
    *   **Calculates sequential features based on the user's *prior exposures* and the offer's *prior exposures* (excluding the current one), relative to the current `exposure_date`**:
        *   `user_prior_total_exposures`: Count of all *previous* exposures for the same user.
        *   `user_prior_converted_exposures`: Count of *previous* exposures that resulted in conversion for the same user.
        *   `user_prior_conversion_rate`: `user_prior_converted_exposures` / `user_prior_total_exposures` (0.0 if no prior exposures).
        *   `days_since_last_user_exposure`: Number of days between the current `exposure_date` and the user's *most recent prior* `exposure_date`. If it's the user's first exposure, use the number of days between `signup_date` and `exposure_date`.
        *   `offer_prior_total_exposures`: Count of all *previous* exposures for the same offer (across all users).
        *   `offer_prior_converted_exposures`: Count of *previous* exposures that resulted in conversion for the same offer.
        *   `offer_prior_conversion_rate`: `offer_prior_converted_exposures` / `offer_prior_total_exposures` (0.0 if no prior exposures).
    *   **Includes static user and offer attributes**: `user_id`, `offer_id`, `age`, `region`, `browsing_frequency_level`, `offer_type`, `category_focus`, `discount_percentage`, `signup_date`.
    *   The query should return `exposure_id`, `user_id`, `offer_id`, `exposure_date`, `was_converted`, `age`, `region`, `browsing_frequency_level`, `offer_type`, `category_focus`, `discount_percentage`, `signup_date`, and all the engineered prior features.
    *   **Hint**: Use window functions with `OVER (PARTITION BY ... ORDER BY ... ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)` for prior aggregates, and `LAG()` for `days_since_last_user_exposure` (with a `COALESCE` to `julianday(e.exposure_date) - julianday(u.signup_date)` for first exposures).

3. **Pandas Feature Engineering & Binary Target Creation**: Fetch the SQL query results into a pandas DataFrame (`campaign_features_df`).
    *   Handle `NaN` values: Fill `user_prior_total_exposures`, `user_prior_converted_exposures`, `offer_prior_total_exposures`, `offer_prior_converted_exposures` with 0. Fill `user_prior_conversion_rate` and `offer_prior_conversion_rate` with 0.0. Ensure `days_since_last_user_exposure` is filled appropriately (SQL should handle this, but verify/fill with a large sentinel like 9999 if any `NaN`s remain for first exposures).
    *   Convert `signup_date` and `exposure_date` to datetime objects. Calculate `user_account_age_at_exposure_days`: Days between `signup_date` and `exposure_date`.
    *   Create `user_had_prior_conversion`: A binary feature (1 if `user_prior_converted_exposures > 0`, else 0).
    *   Define features `X` (all numerical: `age`, `discount_percentage`, `user_account_age_at_exposure_days`, `user_prior_total_exposures`, `user_prior_converted_exposures`, `user_prior_conversion_rate`, `days_since_last_user_exposure`, `offer_prior_total_exposures`, `offer_prior_converted_exposures`, `offer_prior_conversion_rate`; categorical: `region`, `browsing_frequency_level`, `offer_type`, `category_focus`, `user_had_prior_conversion`) and target `y` (`was_converted`). Split into training and testing sets (e.g., 70/30 split) using `sklearn.model_selection.train_test_split` (set `random_state=42`, `stratify` on `y` to handle potential class imbalance).

4. **Data Visualization**: Create two separate plots to visually inspect relationships with `was_converted`:
    *   A violin plot (or box plot) showing the distribution of `discount_percentage` for `was_converted=0` vs. `was_converted=1`. Ensure appropriate labels and titles.
    *   A stacked bar chart showing the proportion of `was_converted` (0 or 1) across different `offer_type` values. Ensure appropriate labels and titles.

5. **ML Pipeline & Evaluation (Binary Classification)**: 
    *   Create an `sklearn.pipeline.Pipeline` with a `sklearn.compose.ColumnTransformer` for preprocessing:
        *   For numerical features (`age`, `discount_percentage`, `user_account_age_at_exposure_days`, `user_prior_total_exposures`, `user_prior_converted_exposures`, `user_prior_conversion_rate`, `days_since_last_user_exposure`, `offer_prior_total_exposures`, `offer_prior_converted_exposures`, `offer_prior_conversion_rate`): Apply `sklearn.preprocessing.SimpleImputer(strategy='mean')` followed by `sklearn.preprocessing.StandardScaler`.
        *   For categorical features (`region`, `browsing_frequency_level`, `offer_type`, `category_focus`, `user_had_prior_conversion`): Apply `sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')`.
    *   The final estimator in the pipeline should be `sklearn.ensemble.HistGradientBoostingClassifier` (set `random_state=42`).
    *   Train the pipeline on `X_train`, `y_train`. Predict probabilities for the positive class (class 1) on the test set (`X_test`).
    *   Calculate and print the `sklearn.metrics.roc_auc_score` and a `sklearn.metrics.classification_report` for the test set predictions.

## Focus
Predicting marketing campaign conversion at the individual exposure level, leveraging historical user and offer interaction patterns and dynamic time-series features using SQL window functions.

## Dataset
Synthetic campaign exposure data (users, offers, exposures) simulating conversion rates based on user demographics, past behavior, and offer attributes.

## Hint
For the SQL query in Step 2, carefully craft `LAG()` and `SUM() OVER (PARTITION BY ... ORDER BY ... ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)` window functions. Remember to `COALESCE` the `LAG` result for `days_since_last_user_exposure` with the user's account age for their very first exposure. This approach builds context dynamically for each exposure event.
