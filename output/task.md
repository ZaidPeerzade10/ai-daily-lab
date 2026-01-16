# AI Daily Lab — 2026-01-16

## Task
1. **Generate Synthetic Data**: Create a pandas DataFrame `X_df` with 1000 samples for a binary classification problem using `sklearn.datasets.make_classification` (e.g., 5 informative features, `random_state=42`). Let `y` be the target variable. Add two new columns to `X_df`:
    *   `product_code`: A categorical string feature with variations like 'TYPE-A_v1', 'type-B-v2', 'TYPE-A v3', 'TYPE_C_v1', 'Type-A'. Ensure a mix of 3-4 distinct 'types' (A, B, C, D) with messy suffixes/formats and inconsistent casing.
    *   `description`: A free-text string column. Populate it with short descriptions that sometimes include keywords like 'urgent', 'fragile', 'standard' (e.g., 'Product with urgent delivery', 'Standard item', 'Fragile glass product').
2. **Feature Engineering - String and Text Processing**: Create new features in `X_df` (or a copy) based on the new string columns:
    *   `product_type_clean`: Extract the clean 'TYPE-X' part from `product_code` (e.g., 'TYPE-A', 'TYPE-B', 'TYPE-C') and convert it to consistent uppercase. (Hint: Use regex to extract 'TYPE-letter').
    *   `is_urgent`: A binary (0 or 1) feature, 1 if 'urgent' is found in `description` (case-insensitive), 0 otherwise.
3. **ML Pipeline with ColumnTransformer**: Split the processed `X_df` (including your new engineered features) and `y` into training and testing sets (e.g., 70/30 split) using `train_test_split`.
    Create an `sklearn.pipeline.Pipeline` that uses `sklearn.compose.ColumnTransformer` for preprocessing:
    *   For the original numerical features (from `make_classification`): Apply `StandardScaler`.
    *   For `product_type_clean`: Apply `OneHotEncoder(handle_unknown='ignore')`.
    *   For `is_urgent`: Pass through (no transformation).
    *   Follow the `ColumnTransformer` with `sklearn.linear_model.LogisticRegression` (set `random_state=42`, `solver='liblinear'`).
4. **Evaluate Performance**: Evaluate the complete pipeline's performance using 5-fold cross-validation (`sklearn.model_selection.cross_val_score`) on the *training data*. Use `scoring='roc_auc'` as the metric. Report the mean and standard deviation of the ROC AUC scores.

## Focus
Feature Engineering (String/Text Manipulation), ML Pipelines with ColumnTransformer, Model Evaluation

## Dataset
Synthetic binary classification data augmented with custom messy string features.

## Hint
Use pandas `.str` methods (e.g., `.str.upper()`, `.str.contains()`, `.str.extract()` with regex) for feature engineering. Remember to correctly define the columns for each transformer within `ColumnTransformer` using named tuples (e.g., `('numeric_scaler', StandardScaler(), num_cols)`). For `is_urgent`, you can specify `('passthrough_urgent', 'passthrough', ['is_urgent'])` in the `ColumnTransformer` or list it with the numerical features if you want to scale it.
