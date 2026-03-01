# AI Daily Lab — 2026-03-01

## Task
Develop a machine learning pipeline to predict the likelihood of a positive user-product interaction (e.g., add to cart, purchase) given historical user behavior and product attributes.

## Focus
Predicting user-product interaction outcome (binary classification) using event-level feature engineering, sequential features, and static attributes.

## Dataset
1. **Generate Synthetic Data (Pandas/Numpy)**: Create three pandas DataFrames:
    *   `users_df`: With 500-700 rows. Columns: `user_id` (unique integers), `signup_date` (random dates over the last 5 years), `region` (e.g., 'North', 'South', 'East', 'West'), `subscription_level` (e.g., 'Free', 'Basic', 'Premium').
    *   `products_df`: With 100-150 rows. Columns: `product_id` (unique integers), `category` (e.g., 'Electronics', 'Books', 'Apparel', 'HomeGoods'), `price` (random floats 10.0-1000.0), `avg_rating` (random floats 2.5-5.0), `launch_date` (random dates over the last 3 years).
    *   `interactions_df`: With 5000-8000 rows. Columns: `interaction_id` (unique integers), `user_id` (randomly sampled from `users_df` IDs), `product_id` (randomly sampled from `products_df` IDs), `interaction_date` (random dates occurring *after* respective `signup_date` and `launch_date`), `interaction_type` (e.g., 'view', 'add_to_cart', 'purchase', 'review'), `is_positive_interaction` (binary target, 0 for 'view', 1 for 'add_to_cart'/'purchase'/'review').
    *   **Simulate realistic patterns**: Ensure `interaction_date` is always after `signup_date` and `launch_date`. Bias `is_positive_interaction` (overall 10-20% positive rate) such that:
        *   Users with 'Premium' `subscription_level` have a higher chance of positive interactions.
        *   Products with higher `avg_rating` or lower `price` tend to have more positive interactions.
        *   A user's `is_positive_interaction` on a product should correlate with their past overall positive interaction rate.
        *   Sort `interactions_df` by `user_id` then `interaction_date` for easier sequential processing.

2. **Load into SQLite & SQL Feature Engineering (Event-Level Context)**: Create an in-memory SQLite database using `sqlite3`. Load `users_df`, `products_df`, and `interactions_df` into tables named `users`, `products`, and `interactions` respectively.
    Write a single SQL query that performs the following for *each interaction event* in `interactions`:
    *   **Joins** `users`, `products`, and `interactions` tables.
    *   **Calculates sequential features based on the user's *prior interactions* and the product's *prior interactions* (excluding the current one), relative to the current `interaction_date`**:
        *   `user_prior_total_interactions`: Count of all *previous* interactions for the same user.
        *   `user_prior_positive_interactions`: Count of *previous* interactions that were positive (`is_positive_interaction=1`) for the same user.
        *   `user_prior_positive_interaction_rate`: `user_prior_positive_interactions` / `user_prior_total_interactions` (0.0 if no prior interactions).
        *   `days_since_last_user_interaction`: Number of days between the current `interaction_date` and the user's *most recent prior* `interaction_date`. If it's the user's first interaction, use the number of days between `signup_date` and `interaction_date`.
        *   `product_prior_total_interactions`: Count of all *previous* interactions for the same product (across all users).
        *   `product_prior_positive_interactions`: Count of *previous* interactions that were positive (`is_positive_interaction=1`) for the same product.
        *   `product_prior_positive_interaction_rate`: `product_prior_positive_interactions` / `product_prior_total_interactions` (0.0 if no prior interactions).
    *   **Includes static user and product attributes**: `interaction_id`, `user_id`, `product_id`, `interaction_date`, `is_positive_interaction` (the target), `region`, `subscription_level`, `category`, `price`, `avg_rating`, `signup_date`, `launch_date`.
    *   The query should return all these attributes and engineered features.

3. **Pandas Feature Engineering & Binary Target Creation**: Fetch the SQL query results into a pandas DataFrame (`user_product_features_df`).
    *   Handle `NaN` values: Fill `user_prior_total_interactions`, `user_prior_positive_interactions`, `product_prior_total_interactions`, `product_prior_positive_interactions` with 0. Fill `user_prior_positive_interaction_rate` and `product_prior_positive_interaction_rate` with 0.0. Ensure `days_since_last_user_interaction` is handled appropriately (SQL should mostly do this; if `NaN`s remain for first interactions, fill with a large sentinel like 9999 days).
    *   Convert `signup_date`, `launch_date`, and `interaction_date` to datetime objects. Calculate `user_account_age_at_interaction_days`: Days between `signup_date` and `interaction_date`. Calculate `product_age_at_interaction_days`: Days between `launch_date` and `interaction_date`.
    *   Create `user_had_prior_positive_interaction`: A binary feature (1 if `user_prior_positive_interactions > 0`, else 0).
    *   Define features `X` (all numerical: `price`, `avg_rating`, `user_account_age_at_interaction_days`, `product_age_at_interaction_days`, `user_prior_total_interactions`, `user_prior_positive_interactions`, `user_prior_positive_interaction_rate`, `days_since_last_user_interaction`, `product_prior_total_interactions`, `product_prior_positive_interactions`, `product_prior_positive_interaction_rate`; categorical: `region`, `subscription_level`, `category`, `user_had_prior_positive_interaction`) and target `y` (`is_positive_interaction`). Split into training and testing sets (e.g., 70/30 split) using `sklearn.model_selection.train_test_split` (set `random_state=42`, `stratify` on `y` to handle class imbalance).

4. **Data Visualization**: Create two separate plots to visually inspect relationships with `is_positive_interaction`:
    *   A violin plot (or box plot) showing the distribution of `avg_rating` for `is_positive_interaction=0` vs. `is_positive_interaction=1`. Ensure appropriate labels and titles.
    *   A stacked bar chart showing the proportion of `is_positive_interaction` (0 or 1) across different `category` values. Ensure appropriate labels and titles.

5. **ML Pipeline & Evaluation (Binary Classification)**: 
    *   Create an `sklearn.pipeline.Pipeline` with a `sklearn.compose.ColumnTransformer` for preprocessing:
        *   For numerical features: Apply `sklearn.preprocessing.SimpleImputer(strategy='mean')` followed by `sklearn.preprocessing.StandardScaler`.
        *   For categorical features: Apply `sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')`.
    *   The final estimator in the pipeline should be `sklearn.ensemble.HistGradientBoostingClassifier` (set `random_state=42`).
    *   Train the pipeline on `X_train`, `y_train`. Predict probabilities for the positive class (class 1) on the test set (`X_test`).
    *   Calculate and print the `sklearn.metrics.roc_auc_score` and a `sklearn.metrics.classification_report` for the test set predictions.

## Hint
For sequential SQL features, carefully use `LAG()` for previous dates and `SUM() OVER (PARTITION BY ... ORDER BY ... ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)` for cumulative sums/counts. Remember to handle division by zero for rates and `NULL` values for `LAG()` results. Convert dates to Julian days for subtraction in SQLite to get day differences.
