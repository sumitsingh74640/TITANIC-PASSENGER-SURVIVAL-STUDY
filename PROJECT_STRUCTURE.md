# Project Structure

```
titanic-survival/
├── data/                    # Raw dataset
│   └── titanic.csv
├── src/titanic_survival/    # Python package (single responsibility per module)
│   ├── __init__.py
│   ├── config.py            # Paths + model configuration
│   ├── loader.py            # CSV ingestion + schema validation
│   ├── features.py          # Imputation, banding, family features
│   ├── eda.py               # Matplotlib/Seaborn figures
│   ├── model.py             # sklearn pipelines, train/eval
│   ├── reporter.py          # Jinja2 HTML + JSON report
│   └── pipeline.py          # Orchestration
├── notebooks/
│   └── 01_survival_study.ipynb
├── tests/                   # pytest suite (8 tests)
├── docs/                    # Extended documentation
├── reports/                 # Generated artifacts
├── main.py                  # Entrypoint
├── requirements.txt
├── pyproject.toml
├── LICENSE
├── README.md
└── .github/workflows/ci.yml
```

## Module Responsibilities

| Module      | Responsibility                                             |
| ----------- | ---------------------------------------------------------- |
| config      | Central paths, model hyperparameters                       |
| loader      | Read CSV, validate schema                                  |
| features    | Impute missing values, derive engineered columns           |
| eda         | Produce PNG figures for the report                         |
| model       | Preprocess, train, cross-validate, evaluate 3 classifiers  |
| reporter    | Render HTML report + JSON summary                          |
| pipeline    | Wire modules together into a single `run()` entrypoint     |
