"""Survival classification models with a scikit-learn pipeline."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import MODEL_CFG

@dataclass
class ModelResult:
    name: str
    accuracy: float
    roc_auc: float
    cv_mean: float
    cv_std: float
    report: str
    confusion: list[list[int]]

def _preprocessor() -> ColumnTransformer:
    num = list(MODEL_CFG.numeric_features)
    cat = list(MODEL_CFG.categorical_features)
    return ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ])

def _candidates() -> dict[str, Any]:
    rs = MODEL_CFG.random_state
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=rs),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=6, random_state=rs, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(random_state=rs),
    }

def train_and_evaluate(df: pd.DataFrame) -> tuple[list[ModelResult], Pipeline]:
    """Train candidate models, return per-model results and the best pipeline."""
    features = list(MODEL_CFG.numeric_features) + list(MODEL_CFG.categorical_features)
    X = df[features]
    y = df["Survived"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=MODEL_CFG.test_size,
        stratify=y, random_state=MODEL_CFG.random_state,
    )

    results: list[ModelResult] = []
    best: tuple[float, Pipeline, str] | None = None

    for name, clf in _candidates().items():
        pipe = Pipeline([("pre", _preprocessor()), ("clf", clf)])
        cv = cross_val_score(pipe, X_train, y_train,
                             cv=MODEL_CFG.cv_folds, scoring="accuracy", n_jobs=-1)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        results.append(ModelResult(
            name=name,
            accuracy=float(acc),
            roc_auc=float(auc),
            cv_mean=float(cv.mean()),
            cv_std=float(cv.std()),
            report=classification_report(y_test, y_pred, digits=3),
            confusion=confusion_matrix(y_test, y_pred).tolist(),
        ))
        if best is None or acc > best[0]:
            best = (acc, pipe, name)

    assert best is not None
    return results, best[1]
