# AI Daily Lab — 2025-12-01

## Task
1. Generate a synthetic classification dataset using `sklearn.datasets.make_classification` with at least 1000 samples, 5 numerical features, and 2 categorical features (one with 3 unique values, one with 5 unique values). Introduce missing values (e.g., `np.nan`) into two of the numerical features.
2. Create an `sklearn.compose.ColumnTransformer` to preprocess the data:
    *   For numerical features: Impute missing values with the mean, then apply `StandardScaler`.
    *   For categorical features: Apply `OneHotEncoder`.
3. Construct an `sklearn.pipeline.Pipeline` that first applies this `ColumnTransformer` and then trains a `RandomForestClassifier`.
4. Evaluate the complete pipeline's performance using 5-fold cross-validation (`sklearn.model_selection.cross_val_score`) and report the mean accuracy and its standard deviation.

## Focus
ML pipelines

## Dataset
Synthetic (generated using `make_classification`)

## Hint
Key `sklearn` components: `make_classification`, `ColumnTransformer`, `SimpleImputer`, `StandardScaler`, `OneHotEncoder`, `Pipeline`, `RandomForestClassifier`, `cross_val_score`.
