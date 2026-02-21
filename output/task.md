# AI Daily Lab — 2026-02-21

## Task
Develop a machine learning pipeline to predict customer sentiment for individual interactions, leveraging both historical user behavior and the content of the interaction itself.

## Focus
Text Feature Engineering, Sequential Aggregations with SQL, Multi-Class Classification, ML Pipeline Integration.

## Dataset
1. **Generate Synthetic Data (Pandas/Numpy)**: Create two pandas DataFrames:
    *   `users_df`: With 500-700 rows. Columns: `user_id` (unique integers), `signup_date` (random dates over the last 5 years), `age` (random integers 18-70), `region` (e.g., 'North', 'South', 'East', 'West'), `subscription_tier` (e.g., 'Free', 'Basic', 'Premium').
    *   `interactions_df`: With 5000-8000 rows. Columns: `interaction_id` (unique integers), `user_id` (randomly sampled from `users_df` IDs), `interaction_date` (random dates occurring *after* their respective `signup_date`), `channel` (e.g., 'Chat', 'Email', 'Survey', 'Social_Media'), `interaction_text` (short text strings), `sentiment_label` (multi-class target: 'Positive', 'Neutral', 'Negative').
    *   **Simulate Realistic Sentiment Patterns**: Ensure `interaction_date` is always after `signup_date`. Generate `interaction_text` such that it **reflects the `sentiment_label`** (e.g., 'Positive' texts include words like 'excellent', 'happy', 'resolved'; 'Negative' texts include 'frustrated', 'issue', 'slow'; 'Neutral' texts include 'ok', 'question', 'feedback'). Some users should have more positive interactions, others more negative. Sort `interactions_df` by `user_id` then `interaction_date`.

2. **Load into SQLite & SQL Feature Engineering (Prior User Sentiment History)**: Create an in-memory SQLite database using `sqlite3`. Load `users_df` and `interactions_df` into tables named `users` and `interactions` respectively.
    Write a single SQL query that performs the following for *each interaction* in `interactions`:
    *   **Joins** `interactions` with `users` to get user attributes.
    *   **Calculates sequential features based on the user's *prior interactions* (excluding the current one)**:
        *   `user_prior_total_interactions`: Count of all *previous* interactions by the same user.
        *   `user_prior_positive_interactions`: Count of *previous* interactions with `sentiment_label = 'Positive'` for the same user.
        *   `user_prior_negative_interactions`: Count of *previous* interactions with `sentiment_label = 'Negative'` for the same user.
        *   `user_prior_sentiment_ratio_pos_neg`: (`user_prior_positive_interactions` + 1) / (`user_prior_negative_interactions` + 1) (add 1 for Laplace smoothing).
        *   `days_since_last_user_interaction`: Number of days between the current `interaction_date` and the user's *most recent prior* `interaction_date`. If it's the user's first interaction, use the number of days between `signup_date` and `interaction_date`.
    *   **Includes static user and interaction attributes**: `interaction_id`, `user_id`, `interaction_date`, `channel`, `interaction_text`, `sentiment_label`, `age`, `region`, `subscription_tier`, `signup_date`.
    *   The query should return all these columns.
    *   **Hint**: Use window functions with `OVER (PARTITION BY user_id ORDER BY interaction_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)` for prior aggregates, and `LAG()` for `days_since_last_user_interaction` with a `COALESCE` to `JULIANDAY(i.interaction_date) - JULIANDAY(u.signup_date)` for first interactions.

3. **Pandas Feature Engineering & Multi-Class Target Creation**: Fetch the SQL query results into a pandas DataFrame (`interaction_features_df`).
    *   Handle `NaN` values: Fill `user_prior_total_interactions`, `user_prior_positive_interactions`, `user_prior_negative_interactions` with 0. Fill `user_prior_sentiment_ratio_pos_neg` with 1.0. Ensure `days_since_last_user_interaction` is filled appropriately (SQL should handle this, but verify/fill with a large sentinel like 9999 if any `NaN`s remain for first interactions).
    *   Convert `signup_date` and `interaction_date` to datetime objects. Calculate `user_account_age_at_interaction_days`: Days between `signup_date` and `interaction_date`.
    *   **Extract Text Features**: Use `sklearn.feature_extraction.text.TfidfVectorizer` to convert `interaction_text` into a sparse TF-IDF feature matrix. Use `max_features=500` for a manageable size.
    *   Define features `X` (all numerical: `age`, `user_account_age_at_interaction_days`, `user_prior_total_interactions`, `user_prior_positive_interactions`, `user_prior_negative_interactions`, `user_prior_sentiment_ratio_pos_neg`, `days_since_last_user_interaction`; categorical: `region`, `subscription_tier`, `channel`; and the TF-IDF features from `interaction_text`) and target `y` (`sentiment_label`). Split `X` and `y` into training and testing sets (e.g., 70/30 split) using `sklearn.model_selection.train_test_split` (set `random_state=42`, `stratify` on `y` for class balance).

4. **Data Visualization**: Create two separate plots to visually inspect relationships with `sentiment_label`:
    *   A stacked bar chart showing the distribution of `sentiment_label` across different `channel` values. Ensure appropriate labels and titles.
    *   A violin plot (or box plot) showing the distribution of `user_prior_sentiment_ratio_pos_neg` for each `sentiment_label`. Ensure appropriate labels and titles.

5. **ML Pipeline & Evaluation (Multi-Class Classification)**: 
    *   Create an `sklearn.pipeline.Pipeline` with a `sklearn.compose.ColumnTransformer` for preprocessing:
        *   For numerical features (e.g., `age`, `user_account_age_at_interaction_days`, `user_prior_total_interactions`, `user_prior_positive_interactions`, `user_prior_negative_interactions`, `user_prior_sentiment_ratio_pos_neg`, `days_since_last_user_interaction`): Apply `sklearn.preprocessing.SimpleImputer(strategy='mean')` followed by `sklearn.preprocessing.StandardScaler`.
        *   For categorical features (`region`, `subscription_tier`, `channel`): Apply `sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')`.
        *   **Note**: The TF-IDF features should be concatenated with the output of the ColumnTransformer *before* passing to the final estimator. Treat them as already processed numerical features.
    *   The final estimator in the pipeline should be `sklearn.ensemble.RandomForestClassifier` (set `random_state=42`, `n_estimators=100`, `class_weight='balanced'` for potential class imbalance).
    *   Train the pipeline on `X_train` (including TF-IDF features), `y_train`. Predict `sentiment_label` for `X_test`.
    *   Calculate and print the `sklearn.metrics.accuracy_score` and a `sklearn.metrics.classification_report` for the test set predictions.

## Hint
When combining TF-IDF features with `ColumnTransformer` output, remember that TF-IDF generates a sparse matrix. You'll need to convert `X_train` and `X_test` into the full feature sets (numerical, categorical, and TF-IDF) before training. `scipy.sparse.hstack` can be useful after `TfidfVectorizer` and `ColumnTransformer` outputs. For the `days_since_last_user_interaction` calculation in SQL, be careful with `LAG`'s default value and date calculations (e.g., `JULIANDAY` in SQLite to get days difference).
