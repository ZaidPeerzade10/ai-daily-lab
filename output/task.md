# AI Daily Lab — 2026-05-21

## Task
Develop a machine learning pipeline to predict the **likelihood of an individual product item within an order being returned** within 30 days of its purchase, based on customer profile, product attributes, and historical return patterns.

## Focus
Binary Classification for Product Returns, covering Synthetic Data Generation, SQL Feature Engineering with time-windowed aggregates, Pandas Feature Engineering, Data Visualization, and an Scikit-learn ML Pipeline.

## Dataset
Simulated E-commerce Data: `customers_df` (customer demographics), `products_df` (product details), `orders_df` (order metadata), `order_items_df` (items within orders), `returns_df` (simulated returns for order items).

## Hint
1. **Synthetic Data**: Simulate return dates such that some items are returned within 30 days, some later, and most not at all. Ensure `return_date` is after `order_date`. Certain product categories (e.g., 'Apparel') or newer customers might have higher return rates. 
2. **SQL Feature Engineering**: Define `GLOBAL_PREDICTION_CUTOFF_DATE`. For each `order_item` that occurs *after* the cutoff, calculate historical return rates (e.g., `customer_avg_return_rate_prev_6m`, `product_avg_return_rate_all_time_at_cutoff`, `category_avg_return_rate_all_time_at_cutoff`) based on data *up to and including* the cutoff date. Use `LEFT JOIN`s and `COALESCE` to handle cases with no prior history (e.g., 0.0 for averages, 0 for counts). Use `julianday()` for date comparisons.
3. **Pandas Feature Engineering & Target Creation**: The target `will_be_returned_in_next_30_days` is 1 if an `order_item`'s `return_date` falls between its `order_date` and `order_date + 30 days`. Otherwise, 0. Create features like `days_since_customer_signup_at_order` and `item_value_percentage_of_order`.
4. **Data Visualization**: Explore relationships between numerical features (e.g., `item_price`, `customer_avg_return_rate_prev_6m`) and the binary target using violin/box plots, and categorical features (e.g., `product_category`, `loyalty_status`) using stacked bar charts.
5. **ML Pipeline**: Use a `HistGradientBoostingClassifier` with `ColumnTransformer` for numerical (imputation, scaling) and categorical (one-hot encoding) preprocessing. Evaluate with `roc_auc_score` and `classification_report`, stratifying the train-test split on the target.
