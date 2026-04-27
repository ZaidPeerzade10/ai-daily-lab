# AI Daily Lab — 2026-04-27

## Task
Develop a machine learning pipeline to predict the **overall average customer rating** for new products, based on their early review sentiment and product metadata.

## Focus
Regression, Feature Engineering from Time-Series/Event Data, SQL Aggregations, Model Evaluation.

## Dataset
1. **Generate Synthetic Data (Pandas/Numpy)**: Create two pandas DataFrames:
    *   `products_df`: With 500-800 rows. Columns: `product_id` (unique integers), `category` (e.g., 'Electronics', 'Books', 'Apparel', 'Home Goods'), `launch_date` (random dates over the last 3 years), `initial_marketing_spend` (random floats 100.0-5000.0).
    *   `reviews_df`: With 10000-15000 rows. Columns: `review_id` (unique integers), `product_id` (randomly sampled from `products_df` IDs), `review_date` (random dates occurring *after* their respective `launch_date`), `rating` (random integers 1-5).
    *   **Simulate realistic patterns**: Ensure `review_date` is always after `launch_date`. Products with higher `initial_marketing_spend` should generally attract more reviews. Products in certain categories ('Electronics') might have a slightly higher average rating bias.
    *   Sort `reviews_df` by `product_id` then `review_date`.

2. **Load into SQLite & SQL Feature Engineering (Early Review Metrics)**: Create an in-memory SQLite database using `sqlite3`. Load `products_df` and `reviews_df` into tables named `products` and `reviews` respectively. Define `early_review_cutoff_days = 30`.
    Write a single SQL query that performs the following for *each product*:
    *   **Joins** `products` with an aggregated subquery for `reviews`.
    *   **Aggregates features based on reviews *within the first 30 days* post-launch** (i.e., `review_date` within `launch_date` and `launch_date + 30 days`):
        *   `num_initial_reviews_30d` (count of `review_id`s)
        *   `avg_rating_initial_30d` (average of `rating`)
        *   `std_rating_initial_30d` (standard deviation of `rating` - if not directly available, approximate with `(SUM(rating*rating) - SUM(rating)*SUM(rating)/COUNT(rating)) / COUNT(rating)`) or just calculate `MAX(rating) - MIN(rating)` as a proxy for spread.
    *   **Includes static product attributes**: `product_id`, `category`, `launch_date`, `initial_marketing_spend`.
    *   **Ensures** all products are included (using `LEFT JOIN`), showing 0 for counts/sums/std if no reviews in the first 30 days.
    *   The query should return `product_id`, `category`, `launch_date`, `initial_marketing_spend`, and all the aggregated features.
    *   **Hint**: Use `julianday()` for date comparisons. Use `COALESCE` to handle `NULL`s from `LEFT JOIN`s for numerical aggregates.

3. **Pandas Feature Engineering & Regression Target Creation**: Fetch the SQL query results into a pandas DataFrame (`product_features_df`).
    *   Handle `NaN` values: Fill `avg_rating_initial_30d`, `std_rating_initial_30d` with a neutral value (e.g., 3.0) and counts/spends with 0.0 as appropriate.
    *   Convert `launch_date` to datetime objects. Assume a `CURRENT_DATE = pd.to_datetime('2024-03-01')`.
    *   Calculate `days_since_launch_at_current_date`: Number of days between `launch_date` and `CURRENT_DATE`. Fill `NaN` or `inf` with 0.
    *   Calculate `marketing_spend_per_initial_review`: `initial_marketing_spend` / (`num_initial_reviews_30d` + 1). Fill `NaN` or `inf` with 0.
    *   **Create the Regression Target `overall_average_rating`**: For *each product*, calculate its overall average rating from *all* reviews in the *original* `reviews_df`.
        *   Merge this aggregate with `product_features_df` (left join).
    *   Define features `X` (numerical: `num_initial_reviews_30d`, `avg_rating_initial_30d`, `std_rating_initial_30d`, `initial_marketing_spend`, `days_since_launch_at_current_date`, `marketing_spend_per_initial_review`; categorical: `category`) and target `y` (`overall_average_rating`). Split into training and testing sets (e.g., 70/30 split) using `sklearn.model_selection.train_test_split` (set `random_state=42`).

4. **Data Visualization**: Create two separate plots to visually inspect relationships with `overall_average_rating`:
    *   A scatter plot showing `avg_rating_initial_30d` vs. `overall_average_rating`. Add a regression line if possible. Ensure appropriate labels and titles.
    *   A box plot showing the distribution of `overall_average_rating` across different `category` values. Ensure appropriate labels and titles.

5. **ML Pipeline & Evaluation (Regression)**: 
    *   Create an `sklearn.pipeline.Pipeline` with a `sklearn.compose.ColumnTransformer` for preprocessing:
        *   For numerical features: Apply `sklearn.preprocessing.SimpleImputer(strategy='mean')` followed by `sklearn.preprocessing.StandardScaler`.
        *   For categorical features: Apply `sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')`.
    *   The final estimator in the pipeline should be `sklearn.ensemble.HistGradientBoostingRegressor` (set `random_state=42`).
    *   Train the pipeline on `X_train`, `y_train`. Predict `overall_average_rating` on the test set (`X_test`).
    *   Calculate and print the `sklearn.metrics.mean_absolute_error` and `sklearn.metrics.r2_score` for the test set predictions.

## Hint
When simulating `std_rating_initial_30d` in SQL, if SQLite's `STDEV` isn't easily accessible, `(AVG(rating*rating) - AVG(rating)*AVG(rating)) * COUNT(rating) / (COUNT(rating)-1)` or a simpler range `MAX(rating) - MIN(rating)` can work as a proxy for variability. Ensure your target `overall_average_rating` is calculated from *all* reviews for a product, not just the initial 30 days used for features. Use `np.random.normal` or similar to introduce slight biases in ratings for specific categories or marketing spend levels during data generation.
