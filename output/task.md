# AI Daily Lab — 2026-04-08

## Task
Develop a machine learning pipeline to predict the category of a user's *first purchase* occurring after their initial 90 days, based on their profile and early transaction behavior.

## Focus
Multi-class classification for future purchase category prediction, leveraging early user transaction patterns via SQL and Pandas feature engineering.

## Dataset
Synthetic user, product, and transaction data.

## Hint
1. For synthetic data, ensure a good mix of transaction dates for users, allowing for both 'initial 90-day' and 'post-90-day' transactions. 
2. In SQL, use `DATE(u.signup_date, '+90 days')` for the cutoff and conditional `SUM(CASE WHEN p.category = '...' THEN 1 ELSE 0 END)` for category-specific counts.
3. For the target in Pandas, identify each user's *first transaction* with `transaction_date` > `signup_date + 90 days`. Users without such a transaction should be dropped from the final modeling dataset.
4. Remember to `stratify` your train-test split for multi-class targets to maintain class balance.
