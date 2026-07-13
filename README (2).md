# Architecture

The pipeline is composed of six modules under `src/titanic_survival/`. Each has a single responsibility and can be unit-tested in isolation.

```
loader → features → eda → model → reporter
                ↑                        │
                └──── pipeline.run() ────┘
```

## Data flow
1. **loader.load_titanic()** reads `data/titanic.csv` and validates the required columns exist.
2. **features.engineer_features()** applies:
   - Stratified median imputation for `Age` (by Pclass × Sex).
   - Median imputation for `Fare` (by Pclass).
   - Mode imputation for `Embarked`.
   - Derived: `FamilySize`, `IsAlone`, `AgeBand`, `FareBand`.
3. **eda.run_all()** writes 5 PNGs to `reports/figures/`.
4. **model.train_and_evaluate()** builds a `ColumnTransformer + Classifier` pipeline for each model, runs 5-fold CV, and evaluates on a stratified 20% holdout.
5. **reporter.write_reports()** renders the Jinja2 HTML template and dumps a JSON summary.

## Design rationale
- **`ColumnTransformer`** cleanly separates numeric scaling from one-hot encoding.
- **Stratified imputation** reduces class-conditional bias that a simple global median introduces.
- **Three heterogeneous models** — linear, bagging, boosting — give a fair sense of dataset difficulty.
