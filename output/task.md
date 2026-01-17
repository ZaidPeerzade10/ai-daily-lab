# AI Daily Lab — 2026-01-17

## Task
1. **Generate Synthetic Transactional Data**: Create a pandas DataFrame `transactions_df` with 800-1000 rows. Columns should include:
    *   `transaction_id` (unique integers)
    *   `transaction_date` (daily dates spanning 2-3 years, starting from '2022-01-01')
    *   `product_category` (e.g., 4-5 distinct string categories like 'Electronics', 'Books', 'Groceries', 'Clothing', but with inconsistent casing and minor variations, e.g., 'electronics', 'Book', 'groceries item', 'clothes').
    *   `description` (short text descriptions that sometimes contain keywords like 'discount', 'premium', 'sale').
    *   `amount` (random float values between 10.0 and 500.0).
    Ensure the DataFrame is sorted by `product_category` and then `transaction_date`.
2. **Advanced Feature Engineering**: 
    *   **Clean Categorical**: Create a new column `clean_category` by standardizing the `product_category` names (e.g., 'electronics', 'Electronics', 'ELEC' all map to 'Electronics').
    *   **Datetime Features**: Extract `month` and `day_of_week` (integer 0-6) from `transaction_date`.
    *   **Text Feature**: Create `is_discount` (binary: 1 if 'discount' or 'sale' is found in `description` case-insensitively, 0 otherwise).
    *   **Grouped Lag Feature**: For each `clean_category`, calculate `lagged_amount_1d`, which is the `amount` from the previous day for that *specific category*. Fill `NaN` values (e.g., with 0, or by propagating the last valid observation forward and then filling remaining with 0 if needed for initial values).
3. **Chronological Data Split**: Define features `X` (all engineered features created in step 2) and target `y` (the original `amount` column). Split the data chronologically, using the last 60 days of data for the test set. Ensure `X_train`, `X_test`, `y_train`, `y_test` are correctly defined.
4. **ML Pipeline with ColumnTransformer**: Create an `sklearn.pipeline.Pipeline` that uses `sklearn.compose.ColumnTransformer` for preprocessing and `sklearn.ensemble.RandomForestRegressor` as the final estimator (set `random_state=42`, `n_estimators=100`).
    *   **Inside the `ColumnTransformer`**:
        *   For numerical features (`lagged_amount_1d`): Apply `SimpleImputer(strategy='mean')` followed by `StandardScaler`.
        *   For categorical features (`clean_category`, `month`, `day_of_week`): Apply `OneHotEncoder(handle_unknown='ignore')`.
        *   For the binary feature (`is_discount`): Use `Passthrough`.
5. **Train, Predict, and Evaluate**: Train the pipeline on the training data (`X_train`, `y_train`). Predict `amount` for the test set (`X_test`). Calculate and report the `sklearn.metrics.mean_absolute_error` (MAE) and `sklearn.metrics.r2_score`.
6. **Visualize Forecast**: Create a single line plot showing the actual `amount` values for the test period and the model's predicted `amount` values for the same period. Label the axes, add a title like 'Actual vs. Predicted Amounts for Test Set', and include a legend.

## Focus
Feature Engineering (datetime, string, grouped lags), ML Pipelines (ColumnTransformer), Regression, Model Evaluation, Data Visualization

## Dataset
Synthetic transactional data with date, category, description, and amount.

## Hint
Pay close attention to sorting before calculating grouped lag features. For the ColumnTransformer, ensure you correctly map feature names to their respective transformers and remember that `amount` itself is the target `y`, not a feature in `X`.
