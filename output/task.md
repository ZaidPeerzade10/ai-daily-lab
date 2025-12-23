# AI Daily Lab — 2025-12-23

## Task
1. Generate a synthetic dataset of 500-1000 short text documents (e.g., short sentences or phrases) belonging to 3 distinct categories. Ensure some keywords are strongly associated with each category.
2. Split the dataset into training and testing sets (e.g., 70/30 split).
3. Construct an `sklearn.pipeline.Pipeline` that first applies `sklearn.feature_extraction.text.TfidfVectorizer` to convert text into numerical features, and then trains an `sklearn.linear_model.LogisticRegression` model (set `random_state` and `solver='liblinear'` for reproducibility).
4. Train the pipeline on the training data and make predictions on the test data.
5. Print the `sklearn.metrics.classification_report` for the test set predictions.
6. From the *trained* pipeline, extract the `TfidfVectorizer` and `LogisticRegression` steps. Identify and print the top 5 most important features (words) for *each class* based on the `LogisticRegression` coefficients (e.g., highest positive coefficients for each class). Briefly interpret what these features tell you about each class.

## Focus
ML pipelines, feature engineering (text), model evaluation, basic model interpretability

## Dataset
Synthetic text data (e.g., product reviews, news snippets) with 3 categories.

## Hint
When generating synthetic text, create a pool of category-specific keywords and generic words. To extract feature importance from a `Pipeline` for a `LogisticRegression` model, you'll need to access the `named_steps` of the trained pipeline to get the `TfidfVectorizer` and `LogisticRegression` objects. Use `TfidfVectorizer.get_feature_names_out()` and `LogisticRegression.coef_` (paying attention to its shape for multi-class classification) to map coefficients to words. For multi-class (OvR), `coef_[i]` represents coefficients for class `i` vs. all others.
