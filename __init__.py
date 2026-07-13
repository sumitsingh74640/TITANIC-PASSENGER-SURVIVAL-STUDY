# Methodology

## Preprocessing
- Numeric features are standardized (`StandardScaler`).
- Categorical features are one-hot encoded, unknown categories at inference time are ignored.

## Evaluation protocol
- Stratified train/test split (80/20) on `Survived`.
- 5-fold cross-validation on the training set for a robust accuracy estimate.
- Report test-set accuracy, ROC AUC, and confusion matrix.

## Model selection
| Model              | Rationale                              |
| ------------------ | -------------------------------------- |
| LogisticRegression | Interpretable linear baseline          |
| RandomForest       | Non-linear, low bias, low variance     |
| GradientBoosting   | Strong baseline for tabular data       |
