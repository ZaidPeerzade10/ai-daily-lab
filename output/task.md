# AI Daily Lab — 2026-05-01

## Task
Develop a machine learning pipeline to predict the **resolution time category** ('Fast', 'Medium', 'Slow') for new customer support tickets, based on ticket attributes, user profile, and hypothetical agent specialization.

## Focus
Multi-class classification on operational efficiency data, combining SQL for initial feature extraction and Pandas for target creation.

## Dataset
Synthetic data for customer support tickets, user profiles, and support agent characteristics.

## Hint
When creating `resolution_time_category`, use percentiles (e.g., 33rd and 66th) of the calculated `resolution_duration_hours` to define 'Fast', 'Medium', and 'Slow' tiers. Remember to handle potential `NaN` or `inf` values during calculations, especially for ratios or time differences where a divisor might be zero or near zero. For SQL, calculating `julianday(resolution_timestamp) - julianday(submission_timestamp)` will give days, then convert to hours. Stratified splitting is crucial for multi-class targets.
