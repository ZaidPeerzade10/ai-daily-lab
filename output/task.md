# AI Daily Lab — 2026-05-28

## Task
Develop a machine learning pipeline to predict the **conversion rate category** ('Low', 'Medium', 'High') for an individual e-commerce product listing in its first 30 days, based on product attributes, its listed price, and historical category/brand performance up to a specific cutoff date.

1.  **Generate Synthetic Data (Pandas/Numpy)**: Create two pandas DataFrames:
    *   `products_df`: With 1000-1500 rows. Columns: `product_id` (unique integers), `product_name` (unique strings), `category` (e.g., 'Electronics', 'Fashion', 'Home Goods'), `brand` (e.g., 'BrandA', 'BrandB', 'BrandC'), `base_cost` (random floats 10-500), `release_date` (random dates over the last 3 years).
    *   `historical_listings_df`: With 20000-30000 rows. Columns: `listing_id` (unique integers), `product_id` (randomly sampled from `products_df` IDs), `listing_date` (random datetimes, *after* respective `release_date` for the `product_id`), `listed_price` (random floats, generally higher than `base_cost`), `impressions` (random integers 100-5000), `conversions` (random integers 0-200).
    *   **Simulate realistic patterns**: Ensure `listing_date` is always after `release_date`. `conversions` should be positively correlated with `impressions` and inversely correlated with `listed_price`. Some `category`/`brand` combinations should inherently have higher conversion rates (`conversions`/`impressions`). Simulate a slight positive trend in overall conversions over time. Ensure a range of conversion rates to enable multi-class categorization. Sort `historical_listings_df` by `listing_date`.

2.  **Load into SQLite & SQL Feature Engineering (Time-Windowed Aggregations)**: Create an in-memory SQLite database using `sqlite3`. Load `products_df` and `historical_listings_df` into tables named `products` and `listings` respectively.
    *   Define `GLOBAL_PREDICTION_CUTOFF_DATE` as 2 months prior to the latest `listing_date` in your generated `historical_listings_df` (e.g., `historical_listings_df['listing_date'].max() - pd.Timedelta(months=2)`).
    *   Write a single SQL query that performs the following for *each listing event that occurs AFTER `GLOBAL_PREDICTION_CUTOFF_DATE`*:
        *   Joins the `listings` (filtered for events after cutoff) with the `products` table.
        *   Aggregates historical features based on *other listings of the same `category` and `brand` within the 60 days immediately preceding `GLOBAL_PREDICTION_CUTOFF_DATE`*:
            *   `avg_category_cr_prev_60d`: Average conversion rate (`conversions` / `impressions`) for the current listing's `category`.
            *   `num_listings_category_prev_60d`: Count of listings for the current listing's `category`.
            *   `avg_brand_cr_prev_60d`: Average conversion rate for the current listing's `brand`.
            *   `num_listings_brand_prev_60d`: Count of listings for the current listing's `brand`.
            *   `avg_listed_price_category_prev_60d`: Average `listed_price` for the current listing's `category`.
        *   Extracts time-based features from the `listing_date` of the *current* listing (e.g., `day_of_week`, `hour_of_day`, `month_of_year`).
        *   Includes static product and listing attributes: `listing_id`, `product_id`, `category`, `brand`, `base_cost`, `release_date`, `listed_price`, `listing_date`, `impressions`, `conversions` (the actual values for *this specific future listing* to calculate the target later).
    *   **Ensures** all listings *after* the cutoff are included. Handle `NULL`s for historical aggregates (e.g., 0.0 for averages, 0 for counts if no prior activity).
    *   The query should return all mentioned fields.
    *   **Hint**: Use CTEs for pre-calculating category/brand historical aggregates up to the `GLOBAL_PREDICTION_CUTOFF_DATE`, then join these with the future listings. Remember to calculate `conversion_rate = CAST(conversions AS REAL) / impressions` within your CTEs for averages.

3.  **Pandas Feature Engineering & Multi-class Target Creation**: Fetch the SQL query results into a pandas DataFrame (`listing_features_df`).
    *   Convert all relevant date/datetime columns (`release_date`, `listing_date`) to appropriate types.
    *   Handle `NaN` values: Fill numerical historical aggregates (e.g., averages, counts) with 0.0 or 0 as appropriate. Fill `base_cost` NaNs with mean/median. For `impressions` and `conversions` in the target calculation, ensure they are not NaN or zero to avoid division by zero or NaN target. Consider `impressions`=1 if 0 to avoid division by zero for target calculation, or fill `conversions` with 0 if `impressions` is 0.
    *   Calculate `listing_age_days`: Number of days between `release_date` and `listing_date`.
    *   Calculate `price_to_cost_ratio`: `listed_price` / (`base_cost` + 1e-6). Fill any `NaN` or `inf` with 0.
    *   **Create the Multi-class Target `conversion_category`**: First, calculate `conversion_rate = conversions / impressions`. Handle cases where `impressions` is 0 (e.g., `conversion_rate = 0`). Then categorize `conversion_rate`:
        *   'Low': `conversion_rate` <= 0.03
        *   'Medium': 0.03 < `conversion_rate` <= 0.10
        *   'High': `conversion_rate` > 0.10
        (Adjust these thresholds based on the actual distribution of your synthetic data to ensure a reasonable class balance).
    *   Define features `X` (numerical: `base_cost`, `listed_price`, `avg_category_cr_prev_60d`, `num_listings_category_prev_60d`, `avg_brand_cr_prev_60d`, `num_listings_brand_prev_60d`, `avg_listed_price_category_prev_60d`, `day_of_week`, `hour_of_day`, `month_of_year`, `listing_age_days`, `price_to_cost_ratio`; categorical: `category`, `brand`) and target `y` (`conversion_category`).
    *   Split into training and testing sets (e.g., 70/30 split) using `sklearn.model_selection.train_test_split` (set `random_state=42`, `stratify` on `y` for class balance).

4.  **Data Visualization (Matplotlib/Seaborn)**: Create two separate plots to visually inspect relationships with `conversion_category`:
    *   A violin plot (or box plot) showing the distribution of `listed_price` for each `conversion_category`. Ensure appropriate labels and titles.
    *   A stacked bar chart showing the proportion of `conversion_category` (across 'Low', 'Medium', 'High') for different `category` values. Ensure appropriate labels and titles.

5.  **ML Pipeline & Evaluation (Multi-class Classification)**:
    *   Create an `sklearn.pipeline.Pipeline` with a `sklearn.compose.ColumnTransformer` for preprocessing:
        *   For numerical features: Apply `sklearn.preprocessing.SimpleImputer(strategy='mean')` followed by `sklearn.preprocessing.StandardScaler`.
        *   For categorical features: Apply `sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')`.
    *   The final estimator in the pipeline should be `sklearn.ensemble.HistGradientBoostingClassifier` (set `random_state=42`).
    *   Train the pipeline on `X_train`, `y_train`. Predict the `conversion_category` on the test set (`X_test`).
    *   Calculate and print a `sklearn.metrics.classification_report` for the test set predictions.

## Focus
Predicting product listing success category (multi-class classification) with time-windowed historical aggregates.

## Dataset
Synthetic e-commerce product and listing data.

## Hint
Pay close attention to the time-based filtering in SQL for historical features (prior to cutoff) vs. the target data (after cutoff). Handle division by zero for conversion rate calculation carefully in both SQL and Pandas. Ensure your synthetic data creates a meaningful distribution for the multi-class target.
