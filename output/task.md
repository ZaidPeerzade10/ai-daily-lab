# AI Daily Lab — 2026-03-23

## Task
Develop a machine learning pipeline to predict customer churn based on their initial 30 days of subscription usage patterns.

## Focus
Customer Churn Prediction, Time-Series Feature Engineering (early window), Binary Classification.

## Dataset
Synthetic data for customer subscriptions and daily usage logs.

## Hint
When simulating `churn_df`, ensure `churn_date` is `NULL` for non-churned customers, and a date after `signup_date` for churned ones. For SQL, use `LEFT JOIN` to ensure all customers are included even if they have no usage in the first 30 days. For pandas, carefully handle division by zero when calculating rates and frequencies.
