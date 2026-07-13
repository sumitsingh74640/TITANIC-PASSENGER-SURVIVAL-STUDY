# Titanic Passenger Survival Study

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-8%20passing-brightgreen.svg)](tests/)
[![Status](https://img.shields.io/badge/status-production--ready-success.svg)](#)

> **A modular, production-grade machine-learning study of the 1912 Titanic disaster — from raw manifest to reproducible survival-prediction report.**

---

**Author:** Sumit Kumar  
**Intern ID:** CITS3958  
**Internship:** CODTECH Summer Internship

---

## Overview

This project investigates *which passengers survived the sinking of the RMS Titanic — and why*. It goes well beyond a notebook of scatter plots: it ships an end-to-end, reproducible pipeline that ingests the manifest, engineers survival-signal features, trains three competing classifiers, and produces an audit-ready HTML/JSON report.

## Features

- **Modular Python package** (`titanic_survival/`) — single responsibility per module.
- **Robust feature engineering** — stratified median imputation, family-structure features, discretized bands.
- **Three-model comparison** — Logistic Regression, Random Forest, Gradient Boosting with 5-fold CV.
- **Automated reporting** — self-contained dark-themed HTML report plus machine-readable JSON summary.
- **Full test suite** — 8 pytest tests covering ingestion, feature integrity, and modeling.
- **CI-ready** — GitHub Actions workflow for lint + tests on every push.

## Tech Stack

| Layer          | Technology                                    |
| -------------- | --------------------------------------------- |
| Language       | Python 3.10+                                  |
| Data           | Pandas, NumPy                                 |
| Modeling       | scikit-learn (LogReg, RF, GBM)                |
| Visualization  | Matplotlib, Seaborn                           |
| Reporting      | Jinja2 (HTML), JSON                           |
| Testing        | pytest                                        |
| CI             | GitHub Actions                                |

## Architecture

```
        ┌──────────┐   ┌────────────┐   ┌───────┐   ┌────────┐   ┌──────────┐
CSV ──▶ │  loader  │──▶│  features  │──▶│  eda  │──▶│ model  │──▶│ reporter │──▶ HTML + JSON
        └──────────┘   └────────────┘   └───────┘   └────────┘   └──────────┘
                              orchestrated by  pipeline.py
```

## Folder Structure

```
titanic-survival/
├── data/                 # raw dataset (titanic.csv)
├── src/titanic_survival/ # core Python package
│   ├── config.py         # paths + model config
│   ├── loader.py         # ingestion + schema check
│   ├── features.py       # imputation & feature engineering
│   ├── eda.py            # figure generation
│   ├── model.py          # sklearn pipelines + evaluation
│   ├── reporter.py       # HTML/JSON report writer
│   └── pipeline.py       # orchestrator
├── notebooks/            # exploratory Jupyter notebook
├── tests/                # pytest suite (8 tests)
├── reports/              # generated report.html + figures/
├── docs/                 # architecture, methodology, findings
├── main.py               # CLI entry point
├── requirements.txt
├── pyproject.toml
└── .github/workflows/ci.yml
```

## Installation

```bash
git clone <repo-url>
cd titanic-survival
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# Run full pipeline (EDA + models + report)
PYTHONPATH=src python main.py

# Open the generated report
open reports/report.html
```

## Sample Output

```
LogisticRegression   acc=0.821  auc=0.856  cv=0.796 ± 0.026
RandomForest         acc=0.793  auc=0.850  cv=0.818 ± 0.028
GradientBoosting     acc=0.810  auc=0.820  cv=0.819 ± 0.041
```

**Key findings:**
- Female passengers survived at **74.2%** vs male at **18.9%**.
- 1st-class survival (**63%**) was more than double 3rd-class (**24%**).
- Children under 12 had a **~58% survival rate** — evidence of the "women and children first" protocol.

## Screenshots

Report figures are saved to `reports/figures/` after running the pipeline.

## Testing

```bash
PYTHONPATH=src pytest tests/ -v
```

All 8 tests pass — covering loader, feature engineering, and model training.

## Future Improvements

- Hyperparameter search via `GridSearchCV`.
- SHAP-based feature attribution in the HTML report.
- Model persistence via `joblib` + FastAPI serving endpoint.
- Streamlit dashboard for interactive what-if analysis.

## License

MIT — see [LICENSE](LICENSE).

## Author

**Sumit Kumar** — B.Tech Student  
**Intern ID:** CITS3958

## Acknowledgements

- **CODTECH** — Summer Internship program.
- Dataset: Encyclopedia Titanica / Kaggle Titanic competition (canonical passenger manifest).
- scikit-learn, pandas, seaborn open-source communities.

## References

1. Kaggle — *Titanic: Machine Learning from Disaster*.
2. Encyclopedia Titanica — passenger records.
3. Pedregosa et al. — *Scikit-learn: Machine Learning in Python*, JMLR 12 (2011).
