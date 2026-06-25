# AI Daily Lab — 2026-06-25

## Task
Develop a machine learning pipeline to predict if a user will **rate a movie highly** (binary classification: rating_score >= 4) for a specific user-movie interaction, based on user preferences, movie attributes, and historical rating patterns up to a specific cutoff date.

## Focus
Predictive modeling for user-item interactions, involving time-series feature engineering with SQL, and binary classification.

## Dataset
Simulated movie ratings, user profiles, and movie attributes.

## Hint
When generating synthetic data for `ratings_df`, ensure `rating_datetime` is after both `users_df.signup_date` and `movies_df.release_date`. For SQL time-windowed aggregates, compute user-specific and movie-specific historical averages and counts *before* a `GLOBAL_PREDICTION_CUTOFF_DATE`, then join these aggregates to the future ratings that occurred *after* the cutoff. Use `julianday()` for date arithmetic in SQLite. For the target, defining 'high rating' as >=4 and handling class imbalance with `class_weight='balanced'` in your model will be key.
