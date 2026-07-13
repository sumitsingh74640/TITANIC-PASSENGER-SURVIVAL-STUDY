"""Exploratory analyses producing figures and summary tables."""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from .config import PATHS

sns.set_theme(style="whitegrid", context="talk")

def _save(fig, name: str) -> Path:
    PATHS.figures.mkdir(parents=True, exist_ok=True)
    out = PATHS.figures / name
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out

def survival_overview(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["Survived"].value_counts().sort_index()
    ax.bar(["Perished", "Survived"], counts.values, color=["#c0392b", "#27ae60"])
    ax.set_title("Overall Survival Distribution")
    ax.set_ylabel("Passengers")
    return _save(fig, "01_survival_overview.png")

def survival_by_sex(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=df, x="Sex", y="Survived", ax=ax, errorbar=None,
                palette=["#3498db", "#e67e22"])
    ax.set_title("Survival Rate by Sex")
    ax.set_ylabel("Survival Rate")
    return _save(fig, "02_survival_by_sex.png")

def survival_by_class(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=df, x="Pclass", y="Survived", ax=ax, errorbar=None,
                palette="viridis")
    ax.set_title("Survival Rate by Passenger Class")
    ax.set_ylabel("Survival Rate")
    return _save(fig, "03_survival_by_class.png")

def age_distribution(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(data=df, x="Age", hue="Survived", bins=30, kde=True,
                 palette=["#c0392b", "#27ae60"], ax=ax)
    ax.set_title("Age Distribution by Survival")
    return _save(fig, "04_age_distribution.png")

def correlation_heatmap(df: pd.DataFrame) -> Path:
    num = df.select_dtypes(include="number")
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(num.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Numeric Feature Correlation")
    return _save(fig, "05_correlation.png")

def run_all(df: pd.DataFrame) -> list[Path]:
    return [
        survival_overview(df),
        survival_by_sex(df),
        survival_by_class(df),
        age_distribution(df),
        correlation_heatmap(df),
    ]
