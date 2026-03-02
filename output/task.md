# AI Daily Lab — 2026-03-02

## Task
Develop a machine learning pipeline to predict customer churn based on their recent activity and static profile information.

## Focus
Predicting customer churn (binary classification) using aggregated historical usage patterns and user demographics, featuring advanced SQL analytics for time-windowed feature engineering.

## Dataset
Simulate a subscription service with customer profiles and usage events.

## Hint
When generating `usage_events_df` for churned users, ensure their activity stops before a simulated `churn_date`. For SQL aggregation, carefully define `feature_cutoff_date` and `global_analysis_date`. In Pandas, when creating the `is_churned` target, define a `churn_observation_period` (e.g., 90 days after `feature_cutoff_date`) and mark users as churned if their `churn_date` (simulated in step 1) falls within this observation period. For non-churned users, ensure their simulated `churn_date` is either non-existent or far in the future.
