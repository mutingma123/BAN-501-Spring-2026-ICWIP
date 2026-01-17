# BAN-501 Spring 2026 - In-Class Work in Progress

Course materials for BAN-501, featuring interactive notebooks built with [marimo](https://marimo.io/).

## Contents

| File | Topic |
|------|-------|
| `1-linear-regression.py` | Linear regression from scratch using hill-climbing optimization, with statsmodels verification |
| `2-regularized-regression.py` | Ridge and Lasso regression with cross-validation for hyperparameter tuning |
| `3-logistic-regression.py` | Logistic regression for binary classification with odds ratios and model evaluation |
| `4-decision-trees.py` | Decision trees and random forests with feature importance analysis |
| `5-xgboost-optuna.py` | XGBoost with Optuna hyperparameter optimization and model comparison |

## Getting Started

### Prerequisites

Install [pixi](https://pixi.sh/), a fast package manager for reproducible environments:

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

### Setup

Clone the repository and install dependencies:

```bash
git clone <repo-url>
cd BAN-501-Spring-2026-ICWIP
pixi install
```

This creates an isolated environment with all required packages.

### Running Notebooks

Run a notebook as a Python script:

```bash
pixi run python 1-linear-regression.py
```

Or launch the marimo editor for interactive use:

```bash
pixi run marimo edit 1-linear-regression.py
```

## Dependencies

Managed via `pixi.toml`:

- **Python 3.14**
- **Data**: polars, pyarrow
- **Modeling**: statsmodels, scikit-learn, xgboost, optuna
- **Visualization**: matplotlib, seaborn
- **Notebooks**: marimo, ipywidgets
