# AI Daily Lab — 2026-03-29

## Task
Develop a machine learning pipeline to predict the click-through likelihood of an ad impression based on user profile, ad attributes, and the user's prior ad interaction history.

## Focus
Ad Click-Through Rate (CTR) Prediction, Sequential User Behavior, SQL Window Functions, Binary Classification.

## Dataset
Synthetic data mimicking user profiles, ad details, and ad impression logs with click outcomes.

## Hint
For SQL feature engineering, leverage `LAG()` and `SUM(CASE WHEN ... THEN 1 ELSE 0 END) OVER (PARTITION BY user_id ORDER BY impression_timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)` to compute sequential features relative to each impression. For date differences, use `julianday()`. Ensure appropriate handling of `NULL`s for initial user events (e.g., first impression, first click).
