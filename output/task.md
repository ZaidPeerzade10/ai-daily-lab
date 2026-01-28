# AI Daily Lab — 2026-01-28

## Task
1. **Generate Synthetic Data**: Create three pandas DataFrames:
    *   `users_df`: 500-700 rows. Columns: `user_id` (unique int), `age` (random ints 18-70), `region` (e.g., 'North', 'South', 'East', 'West'), `signup_date` (random dates over last 5 years).
    *   `products_df`: 100-150 rows. Columns: `product_id` (unique int), `category` (e.g., 'Electronics', 'Books', 'Clothing', 'HomeGoods'), `price` (random floats 50-500).
    *   `interactions_df`: 3000-5000 rows. Columns: `interaction_id` (unique int), `user_id` (randomly sampled from `users_df` IDs), `product_id` (randomly sampled from `products_df` IDs), `interaction_date` (random dates *after* respective `signup_date`), `interaction_type` (e.g., 'view', 'add_to_cart', 'purchase'), `feedback_text` (short text containing keywords like 'good', 'excellent', 'slow', 'broken', 'loved it', 'ok' randomly, mixed with generic words). Ensure users show a preference for certain categories.

2. **Load into SQLite & SQL Feature Engineering**: Create an in-memory SQLite database. Load `users_df`, `products_df`, and `interactions_df` into tables. Determine an `analysis_date` (e.g., `max(interaction_date)` from `interactions_df` + 30 days). Write a single SQL query that performs the following for *each user*:
    *   **Joins** `users`, `products`, and `interactions` tables.
    *   **Aggregates** user-level features:
        *   `total_interactions` (count of all interactions).
        *   `total_purchases` (count of 'purchase' interactions).
        *   `num_electronics_purchases`, `num_books_purchases`, etc. (count of 'purchase' interactions for each `category`).
        *   `concatenated_feedback`: A single string containing all `feedback_text` provided by the user (concatenate with a space).
        *   `days_since_last_interaction`: Days between `analysis_date` and `MAX(interaction_date)`.
    *   **Ensures** all users are included, showing 0 for counts, empty string for `concatenated_feedback`, and `NULL` for `days_since_last_interaction` if no interactions.
    *   Returns `user_id`, `age`, `region`, `signup_date`, `total_interactions`, `total_purchases`, `num_electronics_purchases`, `num_books_purchases`, `num_clothing_purchases`, `num_homegoods_purchases`, `concatenated_feedback`, `days_since_last_interaction`.

3. **Pandas Feature Engineering & Multi-Class Target Creation**: Fetch the SQL query results into a pandas DataFrame. 
    *   Handle `NaN` values: Fill `total_interactions`, `total_purchases`, and category-specific purchase counts with 0. Fill `concatenated_feedback` with an empty string. For `days_since_last_interaction` (for users with no interactions), fill with a large sentinel value (e.g., `365 * 5` or 1825 days).
    *   Calculate `account_age_days`: Days between `signup_date` and the `analysis_date` (from step 2).
    *   **Create the multi-class target `preferred_category_segment`**: For each user, determine the category they purchased from most. If a tie, pick one (e.g., alphabetically). If no purchases, assign a 'General' segment. (Hint: Use `idxmax()` on category purchase columns after handling zeros).
    *   Define features `X` (all numerical, `region`, `concatenated_feedback`) and target `y` (`preferred_category_segment`). Split into training and testing sets (e.g., 70/30 split) using `sklearn.model_selection.train_test_split` (set `random_state=42`).

4. **Data Visualization**: Create two separate plots to visually inspect relationships with `preferred_category_segment`:
    *   A stacked bar plot showing the distribution of `region` for each `preferred_category_segment`.
    *   A violin plot or box plot showing the distribution of `account_age_days` for each `preferred_category_segment`.
    Ensure plots have appropriate labels and titles.

5. **ML Pipeline & Evaluation (Multi-Class with Text Features and Basic Neural Network)**: 
    *   Create an `sklearn.pipeline.Pipeline` with a `ColumnTransformer` for preprocessing:
        *   For numerical features (e.g., `age`, `account_age_days`, `total_interactions`, `total_purchases`, `num_electronics_purchases`, etc., `days_since_last_interaction`): Apply `sklearn.preprocessing.StandardScaler`.
        *   For the categorical feature (`region`): Apply `sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')`.
        *   For the text feature (`concatenated_feedback`): Apply `sklearn.feature_extraction.text.TfidfVectorizer(max_features=1000)`.
    *   The final estimator in the pipeline should be `sklearn.neural_network.MLPClassifier` (set `random_state=42`, `hidden_layer_sizes=(100, 50)`, `max_iter=300`, `early_stopping=True`).
    *   Train the pipeline on `X_train`, `y_train`. Predict `preferred_category_segment` for `X_test`.
    *   Calculate and print the `sklearn.metrics.accuracy_score` and `sklearn.metrics.classification_report` for the test set predictions.

## Focus
Multi-class classification, SQL aggregation, Text Feature Engineering (TF-IDF), ColumnTransformer with mixed data types, Basic Neural Network (MLPClassifier) for AI experimentation.

## Dataset
Synthetic user demographics, product details, and user interaction data including free-text feedback.

## Hint
When performing SQL aggregation for category-specific counts, consider using `SUM(CASE WHEN ... END)` or subqueries. For the `concatenated_feedback`, SQLite's `GROUP_CONCAT` function is useful. Integrating `TfidfVectorizer` directly into `ColumnTransformer` is key for handling text features in the pipeline. `MLPClassifier` is a good choice for 'basic AI experimentation' in a scikit-learn context.
