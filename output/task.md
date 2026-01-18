# AI Daily Lab — 2026-01-18

## Task
1. **Generate Synthetic Data (pandas/numpy)**: Create two pandas DataFrames:
    *   `customer_profiles_df`: With 500 rows. Columns: `customer_id` (unique integers), `age` (random ints 18-70), `income` (random floats 20000-150000), `is_churn` (binary target, 0 or 1, with a slight imbalance, e.g., 80/20 split).
    *   `transactions_df`: With 2000-3000 rows. Columns: `transaction_id` (unique integers), `customer_id` (randomly sampled from `customer_profiles_df` IDs, ensuring some customers have no transactions and others have many), `transaction_date` (random dates over the last 3 years), `amount` (random floats between 10.0 and 1000.0).
2. **Load into SQLite**: Create an in-memory SQLite database using `sqlite3` and load `customer_profiles_df` into a table named `customers` and `transactions_df` into `transactions`.
3. **SQL Feature Engineering**: Write a single SQL query that performs the following:
    *   **Joins** the `customers` and `transactions` tables.
    *   **Aggregates** to calculate `total_spend`, `avg_spend_per_transaction`, and `num_transactions` for each customer.
    *   **Ensures** that customers with no transactions are still included in the result, showing 0 for their aggregated spend/transaction counts.
    *   Returns `customer_id`, `total_spend`, `avg_spend_per_transaction`, `num_transactions`.
4. **Retrieve and Merge (pandas)**: Fetch the results of the SQL query into a pandas DataFrame. Then, merge this aggregated DataFrame with the original `customer_profiles_df` on `customer_id`.
5. **Data Visualization**: Create a histogram or kernel density plot of `total_spend`, differentiated by `is_churn` (e.g., using `hue` in seaborn or `plt.hist` with different colors) to visually inspect the relationship.
6. **ML Pipeline & Evaluation**: 
    *   Define features `X` (`age`, `income`, `total_spend`, `avg_spend_per_transaction`, `num_transactions`) and target `y` (`is_churn`) from the merged DataFrame.
    *   Split the data into training and testing sets (e.g., 70/30 split) using `sklearn.model_selection.train_test_split`.
    *   Create an `sklearn.pipeline.Pipeline` consisting of `sklearn.preprocessing.StandardScaler` followed by `sklearn.linear_model.LogisticRegression` (set `random_state=42`, `solver='liblinear'` for reproducibility).
    *   Train the pipeline on the training data. Predict probabilities for the positive class (class 1) on the test set.
    *   Calculate and print the `sklearn.metrics.roc_auc_score` for the test set predictions.
    *   Generate and display an ROC curve for the model using `sklearn.metrics.RocCurveDisplay.from_estimator` with the trained pipeline and test data.

## Focus
SQL-driven Feature Engineering for ML, ML Pipelines, Model Evaluation, Data Visualization

## Dataset
Synthetic customer profiles and transactional data

## Hint
For the SQL query in step 3, consider using a `LEFT JOIN` to ensure all customers are included, even those without transactions. For aggregation, `IFNULL` or `COALESCE` can help handle `NULL` values gracefully when counting or summing. Remember to import `sqlite3` and `pandas` for data handling, and `sklearn` for ML components and evaluation, `matplotlib`/`seaborn` for visualization.
