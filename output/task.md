# AI Daily Lab — 2025-12-07

## Task
1. Generate a pandas DataFrame with a `timestamp` column (daily data for 2-3 years) and a `value` column (e.g., synthetic sales data with some trend and seasonality, using `np.sin` or similar).
2. From the `timestamp` column, create new features: `month` (numerical month), `day_of_week` (name of the day, e.g., 'Monday'), and `is_weekend` (boolean).
3. Calculate the average `value` aggregated by `month` and by `day_of_week`.
4. Create two visualizations using `seaborn` and `matplotlib.pyplot`:
    *   A line plot showing the average `value` trend across months.
    *   A bar plot showing the average `value` for each `day_of_week`.
5. Display the head of the DataFrame with the new features and print the aggregated dataframes for both monthly and daily trends.

## Focus
pandas / numpy, feature engineering, data visualization

## Dataset
Synthetic time-series data generated with pandas and numpy.

## Hint
Use `pd.date_range` to create the timestamp index. Leverage pandas `dt` accessor (e.g., `.dt.month`, `.dt.day_name()`, `.dt.weekday`) for feature extraction. Grouping and aggregation (`.groupby().mean()`) will be key for trends. Ensure `day_of_week` is ordered correctly in plots.
