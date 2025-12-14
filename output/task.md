# AI Daily Lab — 2025-12-14

## Task
1. Generate a pandas DataFrame with synthetic transaction data, including `customer_id` (5-10 unique), `transaction_date` (spanning 6-12 months of daily data), `amount` (random float), and `product_category` (3-5 unique strings).
2. For each transaction, calculate two new features using *window functions* (Pandas `groupby` + `rolling` or `expanding`):
    *   `customer_30d_avg_spend`: The average `amount` for that specific `customer_id` over the *past 30 days*, inclusive of the current transaction date.
    *   `customer_cumulative_transactions`: The running total count of transactions for that specific `customer_id`, ordered by `transaction_date`.
3. Aggregate the data to find:
    *   The total `amount` spent by each `customer_id` for each `month`.
    *   The `product_category` with the highest total `amount` spent across *all* customers for each `month`.
4. Display the head of the DataFrame with the new features and print the aggregated monthly customer spending and the top product categories per month.

## Focus
pandas / numpy

## Dataset
Synthetic customer transaction data (customer_id, transaction_date, amount, product_category)

## Hint
For rolling features, first ensure your DataFrame is sorted by `customer_id` and `transaction_date`. Use `groupby('customer_id')['amount'].transform(lambda x: x.rolling(window='30D', on=df.loc[x.index, 'transaction_date']).mean())` for the rolling average. For cumulative count, `groupby('customer_id').cumcount() + 1` is effective. For monthly aggregations, extract the month using `dt.to_period('M')` or `dt.month` from the `transaction_date` column.
