# AI Daily Lab — 2026-05-05

## Task
Develop a machine learning pipeline to predict the **sentiment of customer product reviews** (multi-class: 'Positive', 'Neutral', 'Negative') based on the raw review text and associated product attributes.

## Focus
Text processing with traditional ML, multi-class classification, feature engineering from structured and unstructured data.

## Dataset
Synthetic E-commerce Data: 
1.  `products_df`: `product_id`, `category`, `brand`, `base_price`.
2.  `reviews_df`: `review_id`, `product_id`, `review_text`, `rating` (1-5 stars), `review_date`.

**Simulate realistic patterns**: 
*   Generate `review_text` based on `rating`: 4-5 stars for 'Positive' text, 3 stars for 'Neutral', 1-2 stars for 'Negative'. Vary text length. 
*   Ensure some categorical products (e.g., 'Books') might have a higher proportion of 'Positive' reviews, while others ('Electronics') might have more 'Neutral' or 'Negative' if issues are common. 
*   Derive the true `sentiment` target: 'Positive' (4-5 stars), 'Neutral' (3 stars), 'Negative' (1-2 stars).

**SQL Analytics (SQLite)**:
*   Create an in-memory SQLite database. Load `products_df` and `reviews_df` into `products` and `reviews` tables.
*   Write a single SQL query that joins `reviews` with `products`. Calculate `review_text_length` for each review (e.g., `LENGTH(review_text)`). Include `review_id`, `review_text`, `category`, `brand`, `base_price`, and the derived `sentiment` (from the synthetic data generation step).

**Pandas Feature Engineering & Multi-Class Target Creation**:
*   Fetch SQL results into a pandas DataFrame (`reviews_features_df`).
*   Calculate `word_count` from `review_text`.
*   Define features `X` (raw text: `review_text`; numerical: `review_text_length`, `word_count`, `base_price`; categorical: `category`, `brand`) and target `y` (`sentiment`).
*   Split `X` and `y` into training and testing sets (e.g., 70/30 split) using `sklearn.model_selection.train_test_split` with `random_state=42` and `stratify` on `y` for class balance.

**Data Visualization (Matplotlib/Seaborn)**:
*   A violin plot (or box plot) showing the distribution of `review_text_length` for each `sentiment` category. 
*   A stacked bar chart showing the proportion of `sentiment` categories across different `category` values.

**ML Pipeline & Evaluation (Multi-Class Classification)**:
*   Create an `sklearn.pipeline.Pipeline` with a `sklearn.compose.ColumnTransformer` for preprocessing:
    *   For `review_text`: Apply `sklearn.feature_extraction.text.TfidfVectorizer`.
    *   For numerical features: Apply `sklearn.preprocessing.SimpleImputer(strategy='mean')` followed by `sklearn.preprocessing.StandardScaler`.
    *   For categorical features: Apply `sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')`.
*   The final estimator should be `sklearn.ensemble.HistGradientBoostingClassifier` (set `random_state=42`).
*   Train the pipeline on `X_train`, `y_train`. Predict `sentiment` for `X_test`.
*   Calculate and print the `sklearn.metrics.accuracy_score` and `sklearn.metrics.classification_report` for the test set predictions.

## Hint
When using `ColumnTransformer` with `TfidfVectorizer`, remember that `TfidfVectorizer` expects a single string input, so ensure your `review_text` column is passed correctly. The `TfidfVectorizer` can be put directly into the `ColumnTransformer`'s transformers list. You might need to handle empty review texts during synthetic data generation or preprocessing to avoid `NaN` issues for the vectorizer.
