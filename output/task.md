# AI Daily Lab — 2026-03-20

## Task
Develop a machine learning pipeline to predict whether a customer will click on a specific promotional email, based on their profile, campaign details, and past email interaction history.

## Focus
Email Click-Through Rate (CTR) Prediction, Sequential Feature Engineering, Binary Classification

## Dataset
Simulated customer profiles, email campaign details, and historical email interaction logs.

## Hint
{'step2_sql_features': 'For SQL feature engineering, ensure you join `email_events` with `customers` and `campaigns`. When calculating sequential features (e.g., `customer_prior_total_emails_sent`, `customer_prior_emails_clicked`), use window functions (`SUM() OVER (...)`, `LAG() OVER (...)`) partitioned by `customer_id` and ordered by `campaigns.send_date`. This ensures you only consider events chronologically *before* the current email send for feature calculation. For `days_since_last_customer_email_send`, if a user has no prior sends, calculate days between `signup_date` and current `send_date`.', 'step3_pandas_target': "For the binary target `is_clicked`, it's already present in the `email_events_df` (and thus in your SQL output). Focus on handling `NaN`s correctly for engineered features, especially prior rates (fill with 0.0) and 'days since' features (fill with `days_since_signup_at_send` for a customer's first email).", 'step5_ml_pipeline': 'Remember to include `campaign_type`, `segment`, `loyalty_status` as categorical features for `OneHotEncoder`. The numerical features should include all calculated rates, counts, and date differences.'}
