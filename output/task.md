# AI Daily Lab — 2026-01-05

## Task
1. **Generate Synthetic Time Series Data**: Create a pandas DataFrame `df` with 500-700 daily entries. It should have a `date` column (daily dates starting from '2022-01-01') and a `value` column. Populate `value` with a synthetic time series exhibiting a linear trend, strong weekly seasonality (e.g., higher values on weekends), and some random noise.
2. **Feature Engineering - Advanced Time-Based Features**: Create the following new features in the DataFrame:
    *   `day_of_week`: Extract the day of the week (0-6 or names) from the `date` column.
    *   `is_weekend`: A binary flag (1 if weekend, 0 otherwise).
    *   `exponential_moving_average_7d`: Calculate a 7-day Exponential Moving Average (EMA) of the `value` column using `df['value'].ewm(span=7, adjust=False).mean()`.
3. **Handle Missing Values and Prepare for Modeling**: Drop any rows that contain `NaN` values (due to EMA calculation). Define the target variable `y` as the `value` column, and features `X` as `day_of_week`, `is_weekend`, and `exponential_moving_average_7d`. Convert `day_of_week` into one-hot encoded features using `pd.get_dummies()`.
4. **Visualize Engineered Features**: Create a plot (e.g., using `seaborn.lineplot` or `seaborn.boxplot`) to visualize the relationship between `day_of_week` (or `is_weekend`) and the original `value` to confirm the seasonality. Also, plot the original `value` and the `exponential_moving_average_7d` on the same time-series plot to show the smoothing effect.
5. **Build and Evaluate a Regression Model**: Split the dataset into training and testing sets (e.g., 80/20 split). Train a `LinearRegression` model using the engineered features. Evaluate its performance on the test set using `sklearn.metrics.mean_absolute_error` and `sklearn.metrics.r2_score`. Report both metrics.

## Focus
Time Series Feature Engineering, Pandas Manipulation, Regression Modeling

## Dataset
Synthetic Pandas DataFrame with time series data

## Hint
For weekly seasonality, you can add a sine/cosine wave component based on the day of the week or directly influence values based on `datetime.weekday()`. `pd.get_dummies()` is key for handling `day_of_week` before modeling. Remember to ensure that any NaNs from the EMA are handled before splitting.
