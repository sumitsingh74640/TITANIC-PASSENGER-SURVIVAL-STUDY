# Troubleshooting

**`ModuleNotFoundError: titanic_survival`**  
Run with `PYTHONPATH=src python main.py` or `pip install -e .`.

**`FileNotFoundError: data/titanic.csv`**  
The pipeline expects the dataset at `data/titanic.csv`. It ships with the repository.

**Matplotlib backend errors on headless servers**  
The code uses the `Agg` backend by default — no display server needed.

**Tests hang or fail on Windows**  
Install the exact versions in `requirements.txt`. Use `pytest -x` to stop on first failure.
