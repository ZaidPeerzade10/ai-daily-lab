# AI Daily Lab — 2026-06-16

## Task
Develop a machine learning pipeline to predict the **traffic congestion level** ('Low', 'Medium', 'High') for a road segment at a specific timestamp, based on static road attributes, real-time sensor readings, and recent historical traffic patterns up to a specific cutoff date.

## Focus
Time-series feature engineering (aggregations within historical windows), multi-class classification, and data leakage prevention in a real-world context (traffic prediction).

## Dataset
Simulated data including `road_segments_df` (segment ID, length, lanes, speed limit, type), `traffic_sensors_df` (sensor ID, segment ID), and `traffic_readings_df` (sensor ID, timestamp, vehicle count, observed speed, actual congestion index).

## Hint
Pay close attention to time-based filtering in SQL for feature engineering. Ensure historical aggregates are calculated *only* from data preceding the prediction point, and the target is from the future. Use `julianday()` for date comparisons and `CASE` statements or `COALESCE` for handling missing aggregates in SQL. For the target, percentile-based binning is key to creating a balanced multi-class distribution.
