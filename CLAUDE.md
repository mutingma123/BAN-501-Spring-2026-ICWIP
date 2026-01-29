# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Course materials for BAN-501 featuring interactive notebooks built with [marimo](https://marimo.io/). The notebooks cover machine learning topics in sequence:

1. Linear regression (hill-climbing optimization, statsmodels verification)
2. Regularized regression (Ridge, Lasso, Elastic Net)
3. Logistic regression (binary classification, metrics, ROC/AUC)
4. Decision trees (random forests, feature importance)
5. XGBoost with Optuna hyperparameter optimization

## Environment Setup

This project uses [pixi](https://pixi.sh/) for environment management with Python 3.14.

```bash
pixi install
pixi run python <script.py>
```

## Running Notebooks

Marimo notebooks are Python files with `@app.cell` decorators.

**Non-interactive execution** (for scripts/CI):
```bash
MPLBACKEND=Agg pixi run python 1-linear-regression.py
```

**Interactive editing**:
```bash
pixi run marimo edit 1-linear-regression.py
```

## Marimo Notebook Conventions

Marimo builds a reactive dependency graph from variable names. Follow these patterns:

**Underscore prefix for temporary variables:**
- Loop variables: `for _i, _item in enumerate(...)`
- Intermediate values: `_temp = load(); result = process(_temp)`
- Figure/axes: `_fig, _ax = plt.subplots()`
- File handles: `with open(path) as _f:`

**Import pattern:**
```python
# First cell - define imports once
@app.cell
def _():
    import polars as pl
    import matplotlib.pyplot as plt
    return pl, plt

# Other cells receive via function signature
@app.cell
def _(pl, plt):
    _df = pl.read_csv("data.csv")
    ...
```

**Avoid:**
- Re-importing modules in multiple cells
- Double-assigning the same variable name (use method chaining or `_` intermediates)
- Code outside `@app.cell` decorators

## Code Style

- Data manipulation: polars (not pandas)
- Visualization: matplotlib, seaborn
- Use keyword arguments for function calls with multiple parameters
- Set random seeds explicitly for reproducibility

## Data

```
data/
├── regression/train.parquet          # Ames Housing dataset
├── classification/playground-series-s5e8/train.parquet  # Bank marketing dataset
└── MNIST/                            # MNIST features and targets
```
