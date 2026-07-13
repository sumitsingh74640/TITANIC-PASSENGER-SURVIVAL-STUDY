{
 "cells": [
  {"cell_type": "markdown", "metadata": {}, "source": ["# Titanic Passenger Survival Study\n", "\n", "**Author:** Sumit Kumar  |  **Intern ID:** CITS3958\n", "\n", "Walkthrough of the modular pipeline: load → engineer → EDA → model → report."]},
  {"cell_type": "code", "execution_count": null, "metadata": {}, "outputs": [], "source": ["import sys; sys.path.insert(0, '../src')\n", "from titanic_survival.loader import load_titanic\n", "df = load_titanic()\n", "df.head()"]},
  {"cell_type": "code", "execution_count": null, "metadata": {}, "outputs": [], "source": ["from titanic_survival.features import engineer_features\n", "df_eng = engineer_features(df)\n", "df_eng[['Age','Fare','FamilySize','IsAlone','AgeBand','FareBand']].head()"]},
  {"cell_type": "code", "execution_count": null, "metadata": {}, "outputs": [], "source": ["from titanic_survival.model import train_and_evaluate\n", "results, best = train_and_evaluate(df_eng)\n", "for r in results:\n", "    print(f'{r.name:22s} acc={r.accuracy:.3f} auc={r.roc_auc:.3f}')"]}
 ],
 "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.11"}},
 "nbformat": 4, "nbformat_minor": 5
}
