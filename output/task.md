# AI Daily Lab — 2026-06-19

## Task
Develop a machine learning pipeline to predict the **daily booking demand category** ('Low', 'Medium', 'High') for a restaurant for the next 7 days, based on its static attributes, recent booking history, and overall market demand up to a specific cutoff date.

## Focus
Pandas/Numpy for data generation and advanced feature engineering, SQLite for time-windowed SQL aggregations, Matplotlib/Seaborn for data visualization, and Scikit-learn for ML pipeline development and multi-class classification evaluation.

## Dataset
Synthetic data involving `restaurants_df` (static attributes like cuisine, location, capacity, rating) and `daily_bookings_df` (daily total guests booked for each restaurant over time).

## Hint
Pay close attention to defining `GLOBAL_PREDICTION_CUTOFF_DATE` and ensuring that all features are derived from data strictly *before or on* this cutoff, while the target is derived from data *after* it. For SQL aggregations, use `julianday()` for robust date arithmetic and `CASE` statements within `SUM`/`AVG`/`COUNT` for conditional aggregation (e.g., for weekend averages or capacity thresholds). Ensure proper `LEFT JOIN`s to retain all restaurants, even those with no recent activity.
