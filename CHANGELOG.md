"""Feature engineering for the Titanic survival study."""
from __future__ import annotations
import numpy as np
import pandas as pd

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a new DataFrame with derived, model-ready features."""
    out = df.copy()

    # Impute Age with median stratified by Pclass and Sex to reduce bias.
    out["Age"] = out.groupby(["Pclass", "Sex"])["Age"].transform(
        lambda s: s.fillna(s.median())
    )
    out["Age"] = out["Age"].fillna(out["Age"].median())

    # Fare: fill with median by class.
    out["Fare"] = out.groupby("Pclass")["Fare"].transform(
        lambda s: s.fillna(s.median())
    )
    out["Fare"] = out["Fare"].fillna(out["Fare"].median())

    # Embarked: mode.
    if out["Embarked"].isna().any():
        out["Embarked"] = out["Embarked"].fillna(out["Embarked"].mode().iloc[0])

    # Family features.
    out["FamilySize"] = out["SibSp"] + out["Parch"] + 1
    out["IsAlone"] = (out["FamilySize"] == 1).astype(int)

    # Discretized bands (useful for tree-free / linear models & readable EDA).
    out["AgeBand"] = pd.cut(
        out["Age"], bins=[0, 12, 18, 35, 60, 120],
        labels=["Child", "Teen", "YoungAdult", "Adult", "Senior"],
    ).astype(str)
    out["FareBand"] = pd.qcut(
        out["Fare"], q=4, labels=["Low", "Mid", "High", "VeryHigh"], duplicates="drop",
    ).astype(str)

    return out
