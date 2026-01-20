# AI Daily Lab — 2026-01-20

## Task
1. **Generate Synthetic Data**: Create three pandas DataFrames:
    *   `products_df`: 100-150 rows. Columns: `product_id` (unique int), `category` (e.g., 'Electronics', 'books', 'Home Goods', 'CLOTHING', 'elec' with inconsistent casing/formats), `base_price` (random floats 50-500).
    *   `orders_df`: 800-1000 rows. Columns: `order_id` (unique int), `product_id` (randomly sampled from `products_df` IDs), `order_date` (random dates over last 2 years), `quantity` (random int 1-5).
    *   `reviews_df`: 700-900 rows. Columns: `review_id` (unique int), `order_id` (randomly sampled from `orders_df` IDs, ensuring some orders have no review and some products have multiple reviews), `rating` (random int 1-5, this will be our target), `review_text` (short text containing keywords like 'good', 'excellent', 'fast', 'slow', 'defective', 'broken' randomly, mixed with generic words).

2. **Load into SQLite & SQL Analytics**: Create an in-memory SQLite database and load `products_df`, `orders_df`, and `reviews_df` into tables. Write a single SQL query that joins these three tables. For each review, calculate the `line_item_total` (`orders.quantity * products.base_price`). The query should return `review_id`, `rating`, `review_text`, `order_date`, `category`, and `line_item_total`.

3. **Feature Engineering (Pandas)**: Fetch the SQL query results into a pandas DataFrame. Create the following new features:
    *   `clean_category`: Standardize `category` names (e.g., 'Electronics', 'Books', 'Home Goods', 'Clothing') by converting to title case and handling variations (e.g., 'elec' -> 'Electronics'). (Hint: Use string methods like `.str.title()` and `.str.replace()` or a custom mapping function).
    *   `has_positive_feedback`: Binary (1 if 'good' or 'excellent' (case-insensitive) is found in `review_text`, 0 otherwise).
    *   `has_negative_feedback`: Binary (1 if 'slow', 'defective', or 'broken' (case-insensitive) is found in `review_text`, 0 otherwise).
    *   `days_since_order`: Calculate the number of days between `order_date` and a fixed `analysis_date` (e.g., `max(order_date)` + 30 days from your generated data, after converting `order_date` to datetime objects).

4. **Data Visualization**: Create two visualizations:
    *   A histogram showing the distribution of the `rating` column.
    *   A box plot or violin plot showing the distribution of `rating` for each `clean_category`.

5. **ML Pipeline & Evaluation**: 
    *   Define features `X` (`line_item_total`, `days_since_order`, `clean_category`, `has_positive_feedback`, `has_negative_feedback`) and target `y` (`rating`).
    *   Split the data into training and testing sets (e.g., 80/20 split) using `sklearn.model_selection.train_test_split` (set `random_state=42`).
    *   Create an `sklearn.pipeline.Pipeline` with a `ColumnTransformer` for preprocessing:
        *   Numerical features (`line_item_total`, `days_since_order`): Apply `SimpleImputer(strategy='mean')` followed by `StandardScaler`.
        *   Categorical features (`clean_category`): Apply `OneHotEncoder(handle_unknown='ignore')`.
        *   Binary features (`has_positive_feedback`, `has_negative_feedback`): Use `Passthrough`.
    *   The final estimator in the pipeline should be `sklearn.ensemble.RandomForestRegressor` (set `random_state=42`, `n_estimators=100`).
    *   Train the pipeline on the training data. Predict `rating` for the test set.
    *   Calculate and print the `sklearn.metrics.mean_absolute_error` (MAE) and `sklearn.metrics.r2_score` for the test set predictions.

## Focus
Multi-Source Data Integration (SQL), Advanced Feature Engineering (Categorical Cleaning, Text Keywords, Datetime), Regression ML Pipeline, Data Visualization, Model Evaluation.

## Dataset
Synthetic transactional data (products, orders, reviews) with messy categorical and text features.

## Hint
When generating `review_text`, use `np.random.choice` with a list of common words and a few keywords. For `clean_category`, consider using `df['category'].str.lower().replace({'elec': 'electronics', 'books': 'books', ...}).str.title()` or a custom mapping function. Ensure `order_date` is parsed as datetime objects before calculating `days_since_order`. Remember to handle potential `NaN` values after merging/feature engineering, especially for `days_since_order` if any `order_date` is missing.
