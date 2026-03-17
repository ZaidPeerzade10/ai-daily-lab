# AI Daily Lab — 2026-03-17

## Task
Develop a machine learning pipeline to predict the success tier of new product launches based on early user interaction metrics and product attributes.

## Focus
Product launch success prediction (multi-class classification) using initial user engagement and product characteristics.

## Dataset
1. **Generate Synthetic Data (Pandas/Numpy)**: Create three pandas DataFrames:
    *   `products_df`: With 100-200 rows. Columns: `product_id` (unique integers), `launch_date` (random dates over the last 3 years), `category` (e.g., 'Electronics', 'Software', 'Apparel', 'HomeGoods'), `initial_price` (random floats 20.0-5000.0), `marketing_spend_at_launch` (random floats 1000.0-50000.0).
    *   `user_interactions_df`: With 10000-15000 rows. Columns: `interaction_id` (unique integers), `user_id` (random integers, representing different users for different products), `product_id` (randomly sampled from `products_df` IDs), `interaction_date` (random dates occurring *after* their respective `launch_date` and *within the first 14 days* of product launch), `interaction_type` (e.g., 'View', 'Add_to_Cart', 'Wishlist', 'Share'), `duration_seconds` (random floats 5-600, primarily for 'View' interactions, 0 for others).
    *   `sales_df`: With 1000-2000 rows. Columns: `sale_id` (unique integers), `product_id` (randomly sampled from `products_df` IDs), `sale_date` (random dates occurring *after* `launch_date` and *within the first 60 days* of product launch), `quantity_sold` (random integers 1-10).
    *   **Simulate realistic patterns**: Ensure `interaction_date` is after `launch_date` and within the 14-day window. `sale_date` is after `launch_date` and within the 60-day window. Bias data such that:
        *   Products with higher `marketing_spend_at_launch` tend to have more `View` interactions.
        *   Products with more early `Add_to_Cart` or `Wishlist` interactions are more likely to have higher `quantity_sold` in the first 60 days.
        *   Some `category` types might naturally have higher or lower sales. Longer `duration_seconds` for 'View' can indicate higher user interest.

2. **Load into SQLite & SQL Feature Engineering (Product-Level Early Performance)**: Create an in-memory SQLite database using `sqlite3`. Load `products_df` and `user_interactions_df` into tables named `products` and `user_interactions` respectively. For each product, define its `initial_interaction_cutoff_date` as `launch_date + 14 days`.
    Write a single SQL query that performs the following for *each product*, aggregating its user interaction behavior *within its first 14 days post-launch* (i.e., `interaction_date` before or on `initial_interaction_cutoff_date`):
    *   **Joins** `products` with aggregated subqueries for `user_interactions`.
    *   **Aggregates features based on interactions *within the first 14 days* post-launch**:
        *   `total_views_first_14d` (count of `interaction_type = 'View'`)
        *   `total_add_to_cart_first_14d` (count of `interaction_type = 'Add_to_Cart'`)
        *   `total_wishlist_first_14d` (count of `interaction_type = 'Wishlist'`)
        *   `avg_view_duration_first_14d` (average `duration_seconds` for 'View' interactions)
        *   `num_unique_users_interacting_first_14d` (count of distinct `user_id`s)
        *   `days_from_launch_to_first_interaction`: Number of days between `launch_date` and `MIN(interaction_date)` for the product. `NULL` if no interactions.
    *   **Includes static product attributes**: `product_id`, `launch_date`, `category`, `initial_price`, `marketing_spend_at_launch`.
    *   **Ensures** all products are included (using `LEFT JOIN`s to aggregated subqueries), showing 0 for counts/sums, 0.0 for averages, and `NULL` for `days_from_launch_to_first_interaction` if no interactions in the first 14 days.
    *   The query should return `product_id`, `launch_date`, `category`, `initial_price`, `marketing_spend_at_launch`, and all the aggregated features.
    *   **Hint**: Use `julianday()` for date differences for filtering. Filter interactions based on `i.interaction_date BETWEEN p.launch_date AND DATE(p.launch_date, '+14 days')`.

3. **Pandas Feature Engineering & Multi-Class Target Creation (Product Success Tier)**: Fetch the SQL query results into a pandas DataFrame (`product_features_df`).
    *   Handle `NaN` values: Fill `total_views_first_14d`, `total_add_to_cart_first_14d`, `total_wishlist_first_14d`, `num_unique_users_interacting_first_14d` with 0. Fill `avg_view_duration_first_14d` with 0.0. For `days_from_launch_to_first_interaction` (for products with no interactions in the first 14 days), fill with 14 (representing activity started on day 14, or no activity).
    *   Convert `launch_date` to datetime objects.
    *   Calculate `total_interactions_first_14d`: Sum of all interaction counts.
    *   Calculate `interaction_frequency_per_day_first_14d`: `total_interactions_first_14d` / 14.0. Fill any `NaN`s with 0.
    *   **Create the Multi-Class Target `product_success_tier`**: Calculate `total_sales_in_first_60d` (sum of `quantity_sold`) for each `product_id` from the *original* `sales_df` for sales occurring *after* `launch_date` and *within 60 days* of `launch_date`. Merge this aggregate with `product_features_df` (left join), filling `NaN`s with 0 for products with no sales.
        *   Calculate the 33rd and 66th percentiles for *non-zero* `total_sales_in_first_60d`.
        *   Define segments:
            *   'Low_Success': `total_sales_in_first_60d` == 0.
            *   'Medium_Success': `total_sales_in_first_60d` > 0 AND `total_sales_in_first_60d` <= 33rd percentile.
            *   'High_Success': `total_sales_in_first_60d` > 33rd percentile AND `total_sales_in_first_60d` <= 66th percentile.
            *   'Very_High_Success': `total_sales_in_first_60d` > 66th percentile.
    *   Define features `X` (all numerical: `initial_price`, `marketing_spend_at_launch`, `total_views_first_14d`, `total_add_to_cart_first_14d`, `total_wishlist_first_14d`, `avg_view_duration_first_14d`, `num_unique_users_interacting_first_14d`, `days_from_launch_to_first_interaction`, `total_interactions_first_14d`, `interaction_frequency_per_day_first_14d`; categorical: `category`) and target `y` (`product_success_tier`). Split into training and testing sets (e.g., 70/30 split) using `sklearn.model_selection.train_test_split` (set `random_state=42`, `stratify` on `y` for class balance).

4. **Data Visualization**: Create two separate plots to visually inspect relationships with `product_success_tier`:
    *   A violin plot (or box plot) showing the distribution of `total_add_to_cart_first_14d` for each `product_success_tier`. Ensure appropriate labels and titles.
    *   A stacked bar chart showing the proportion of `product_success_tier` across different `category` values. Ensure appropriate labels and titles.

5. **ML Pipeline & Evaluation (Multi-Class)**: 
    *   Create an `sklearn.pipeline.Pipeline` with a `sklearn.compose.ColumnTransformer` for preprocessing:
        *   For numerical features: Apply `sklearn.preprocessing.SimpleImputer(strategy='mean')` followed by `sklearn.preprocessing.StandardScaler`.
        *   For categorical features: Apply `sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')`.
    *   The final estimator in the pipeline should be `sklearn.ensemble.HistGradientBoostingClassifier` (set `random_state=42`).
    *   Train the pipeline on `X_train`, `y_train`. Predict `product_success_tier` for `X_test`.
    *   Calculate and print the `sklearn.metrics.accuracy_score` and a `sklearn.metrics.classification_report` for the test set predictions.

## Hint
When simulating data, create clear patterns between early interactions (e.g., 'Add_to_Cart') and later sales to make the target predictable. For SQL, ensure `LEFT JOIN`s are used to retain all products, even those with no early interactions. When defining the target tiers in Pandas, remember to calculate percentiles only on *non-zero* sales values to avoid skewing low-engagement products.
