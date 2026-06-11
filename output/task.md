# AI Daily Lab — 2026-06-11

## Task
Develop a machine learning pipeline to predict if a credit card transaction will be **fraudulent** (binary classification), based on transaction details, customer profile, merchant information, and historical transaction patterns up to a specific cutoff date.

## Focus
Binary classification, time-series feature engineering, handling highly imbalanced data, SQL analytics, data visualization.

## Dataset
Synthetic credit card transactions, customer profiles, and merchant details.

## Hint
Pay close attention to generating a realistic, highly imbalanced target variable for 'is_fraud' (e.g., 1-3% fraud). For SQL, ensure historical aggregates are calculated *up to and including* a `GLOBAL_PREDICTION_CUTOFF_DATE`, and the target transactions are those occurring *after* this cutoff. Use `class_weight='balanced'` in your ML model and `roc_auc_score` for evaluation due to the imbalance.
