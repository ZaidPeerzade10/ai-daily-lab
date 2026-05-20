# AI Daily Lab — 2026-05-20

## Task
Develop a machine learning pipeline to predict the **box office success category** ('Flop', 'Moderate', 'Hit') of a movie at its pre-release stage, based on its budget, genre, director/lead actor's historical performance, and aggregated early pre-release buzz up to a fixed cutoff date.

1.  **Generate Synthetic Data (Pandas/Numpy)**: Create three pandas DataFrames:
    *   `movies_df`: With 1000-1500 rows. Columns: `movie_id` (unique integers), `title` (unique strings), `genre` (e.g., 'Action', 'Comedy', 'Drama', 'Sci-Fi'), `budget_usd` (random floats 5M-300M), `director_id`, `lead_actor_id` (randomly sampled from `personnel_df` IDs), `release_date` (random dates over the last 5 years), `box_office_revenue_usd` (random floats 1M-1B - this will be used for target creation).
    *   `personnel_df`: With 500-700 rows. Columns: `person_id` (unique integers), `name`, `role` (e.g., 'Director', 'Actor' - ensure a mix), `past_avg_rating` (random floats 2.0-4.5, for their past projects), `past_total_revenue` (random floats 100M-5B, for their past projects).
    *   `pre_release_buzz_df`: With 3000-5000 rows. Columns: `buzz_id` (unique integers), `movie_id` (randomly sampled from `movies_df` IDs), `buzz_date` (random dates occurring *before* their respective `release_date` for each movie, e.g., 3-12 months prior), `trailer_views` (random integers 100K-50M), `social_mentions` (random integers 10K-1M).
    *   **Simulate realistic patterns**: Ensure `buzz_date` is always before `release_date`. Higher `budget_usd` should correlate with higher `box_office_revenue_usd`. Higher `past_avg_rating` and `past_total_revenue` for director/actor should positively influence `box_office_revenue_usd`. Higher `trailer_views` and `social_mentions` should also positively influence `box_office_revenue_usd`. Create a class imbalance for success categories (e.g., more 'Moderate', fewer 'Hit').
    *   Sort `pre_release_buzz_df` by `movie_id` then `buzz_date`.

2.  **Load into SQLite & SQL Feature Engineering (Pre-release Aggregations)**: Create an in-memory SQLite database using `sqlite3`. Load `movies_df`, `personnel_df`, and `pre_release_buzz_df` into tables named `movies`, `personnel`, and `pre_release_buzz` respectively.
    *   Define `GLOBAL_PREDICTION_CUTOFF_DATE` as 4 months prior to the latest `release_date` in your generated `movies_df` (e.g., `movies_df['release_date'].max() - pd.Timedelta(months=4)`). This represents the fixed point in time at which we are making predictions.
    *   Write a single SQL query that performs the following for *each movie* that has its latest pre-release buzz *before or on* `GLOBAL_PREDICTION_CUTOFF_DATE`, and whose `release_date` is *after* `GLOBAL_PREDICTION_CUTOFF_DATE` (i.e., we are predicting for movies that haven't been released yet at the cutoff):
        *   `movie_id`, `genre`, `budget_usd`.
        *   `director_past_avg_rating`, `director_past_total_revenue` (from `personnel` table, assuming `role='Director'`).
        *   `actor_past_avg_rating`, `actor_past_total_revenue` (from `personnel` table, assuming `role='Actor'`).
        *   `total_trailer_views_at_cutoff`: Sum of `trailer_views` for that movie for all `buzz_date`s *on or before* `GLOBAL_PREDICTION_CUTOFF_DATE`.
        *   `total_social_mentions_at_cutoff`: Sum of `social_mentions` for that movie for all `buzz_date`s *on or before* `GLOBAL_PREDICTION_CUTOFF_DATE`.
        *   `days_from_cutoff_to_release`: Number of days between `GLOBAL_PREDICTION_CUTOFF_DATE` and `release_date`.
        *   Include the actual `box_office_revenue_usd` for target creation later.
    *   **Ensures** only movies with relevant pre-release buzz up to the cutoff are included. Handle `NULL`s for aggregated buzz features (e.g., 0 for sums).
    *   The query should return all mentioned fields.
    *   **Hint**: Use CTEs for aggregating buzz and for handling director/actor joins. Use `julianday()` for date comparisons.

3.  **Pandas Feature Engineering & Multi-class Target Creation**: Fetch the SQL query results into a pandas DataFrame (`movie_features_df`).
    *   Convert `release_date` to datetime objects.
    *   Handle `NaN` values: Fill numerical aggregated buzz features with 0. Fill director/actor `NaN`s with reasonable values (e.g., mean for ratings, 0 for revenue if person not found/data missing).
    *   Create `budget_per_director_revenue`: `budget_usd` / (`director_past_total_revenue` + 1e6) to avoid division by zero. Fill `NaN`/`inf` with 0.
    *   **Create the Multi-class Target `success_category`**: Based on `box_office_revenue_usd` (after cleaning any `NaN`s or extremely low/high values):
        *   'Flop': If `box_office_revenue_usd` < $50,000,000
        *   'Moderate': If $50,000,000 <= `box_office_revenue_usd` < $200,000,000
        *   'Hit': If `box_office_revenue_usd` >= $200,000,000
        (Adjust these thresholds based on the synthetic data distribution to ensure a reasonable class balance).
    *   Define features `X` (numerical: `budget_usd`, `director_past_avg_rating`, `director_past_total_revenue`, `actor_past_avg_rating`, `actor_past_total_revenue`, `total_trailer_views_at_cutoff`, `total_social_mentions_at_cutoff`, `days_from_cutoff_to_release`, `budget_per_director_revenue`; categorical: `genre`) and target `y` (`success_category`).
    *   Split into training and testing sets (e.g., 70/30 split) using `sklearn.model_selection.train_test_split` (set `random_state=42`, `stratify` on `y` for class balance).

4.  **Data Visualization (Matplotlib/Seaborn)**: Create two separate plots to visually inspect relationships with `success_category`:
    *   A box plot (or violin plot) showing the distribution of `budget_usd` for each `success_category`. Consider a log scale for `budget_usd` if distribution is highly skewed.
    *   A stacked bar chart showing the proportion of `success_category` (across 'Flop', 'Moderate', 'Hit') for different `genre` values.
    *   Ensure appropriate labels and titles for both plots.

5.  **ML Pipeline & Evaluation (Multi-class Classification)**: 
    *   Create an `sklearn.pipeline.Pipeline` with a `sklearn.compose.ColumnTransformer` for preprocessing:
        *   For numerical features: Apply `sklearn.preprocessing.SimpleImputer(strategy='mean')` followed by `sklearn.preprocessing.StandardScaler`.
        *   For categorical features: Apply `sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')`.
    *   The final estimator in the pipeline should be `sklearn.ensemble.HistGradientBoostingClassifier` (set `random_state=42`).
    *   Train the pipeline on `X_train`, `y_train`. Predict the `success_category` on the test set (`X_test`).
    *   Calculate and print a `sklearn.metrics.classification_report` for the test set predictions.

## Focus
Multi-class Classification, Relational Feature Engineering, Time-aware Data Slicing (pre-release prediction), SQL Analytics

## Dataset
Synthetic data for movies, personnel (director/actors), and pre-release buzz.

## Hint
When generating `box_office_revenue_usd`, ensure a distribution that allows for distinct 'Flop', 'Moderate', and 'Hit' categories. Pay close attention to the SQL query to correctly aggregate pre-release buzz *up to the global cutoff date* and join personnel data, handling potential `NULL`s for movies with missing buzz or less known personnel. Remember to filter movies such that their `release_date` is *after* your `GLOBAL_PREDICTION_CUTOFF_DATE` to simulate predicting future movie success.
