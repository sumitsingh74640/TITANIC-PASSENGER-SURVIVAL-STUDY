"""Central configuration for paths and modeling parameters."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

@dataclass(frozen=True)
class Paths:
    root: Path = ROOT
    data: Path = ROOT / "data"
    reports: Path = ROOT / "reports"
    figures: Path = ROOT / "reports" / "figures"
    raw_csv: Path = ROOT / "data" / "titanic.csv"

@dataclass(frozen=True)
class ModelConfig:
    random_state: int = 42
    test_size: float = 0.2
    cv_folds: int = 5
    numeric_features: tuple = ("Age", "Fare", "SibSp", "Parch", "FamilySize")
    categorical_features: tuple = ("Sex", "Pclass", "Embarked", "AgeBand", "FareBand")

PATHS = Paths()
MODEL_CFG = ModelConfig()
