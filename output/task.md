# AI Daily Lab — 2026-06-01

## Task
Develop a machine learning pipeline to predict if a piece of industrial equipment will require maintenance within the next 7 days, based on its static attributes, recent sensor readings, and historical maintenance log up to a specific cutoff date.

## Focus
Time-series feature engineering, binary classification, class imbalance handling.

## Dataset
Simulated industrial equipment data including static attributes, sensor readings (temperature, vibration, pressure), and maintenance logs (type, cost, date).

## Hint
Pay close attention to time window definitions in SQL for aggregating sensor data (e.g., last 24 hours) and historical maintenance (e.g., last 90 days), all relative to the `GLOBAL_PREDICTION_CUTOFF_DATE`. When creating the target, iterate through each `equipment_id` and check for any maintenance event in the 7-day future window. Remember to use `class_weight='balanced'` in your classifier due to potential maintenance event imbalance.
