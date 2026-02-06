# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Course materials for BAN-501 featuring interactive notebooks built with [marimo](https://marimo.io/). The notebooks cover machine learning topics in sequence:

1. `1-linear-regression.py` — Hill-climbing optimization on synthetic data, statsmodels verification
2. `2-lasso-ridge-regression.py` — Ridge, Lasso, Elastic Net on Ames Housing (regression)
3. `3-logistic-regression.py` — Binary classification with statsmodels on bank marketing data
4. `4-decision-trees.py` — Decision trees with GridSearchCV, logistic regression comparison, permutation importance
5. `decision-trees.py` — Variant focusing on tree visualization (export_text, plot_tree)

Concept slides live in `concept-slides/beamer/` as LaTeX Beamer source, with compiled PDFs in `concept-slides/`.

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

## Notebook Architecture

Each notebook follows a consistent pattern:

1. **Single import cell** — All imports in one `@app.cell def _():` block, returning every name used by other cells. Includes `sns.set_style("whitegrid")`.
2. **Markdown cells** — `@app.cell(hide_code=True)` with `mo.md(r"""...""")` for section explanations with LaTeX math.
3. **Code cells** — Receive dependencies via function signature (e.g., `def _(pl, plt, X_train):`). Return any variables needed downstream.
4. **Empty trailing cell** — Each notebook ends with an empty `@app.cell` before `app.run()`.

Notebooks 2-4 share a common pipeline pattern for classification on the bank marketing dataset: load parquet → select features → train/test split with stratification → one-hot encode categoricals → fit model → evaluate with AUC/confusion matrix.

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

- Data manipulation: polars (not pandas), except when passing to statsmodels (which requires pandas)
- Visualization: matplotlib + seaborn with `sns.set_style("whitegrid")`, `edgecolor='k'` on barplots, 4:3 aspect ratio
- Use keyword arguments for function calls with multiple parameters
- Set random seeds explicitly for reproducibility (`random_state=42`, `seed=42`)

## Data

```
data/
├── regression/train.parquet          # Ames Housing dataset
├── classification/playground-series-s5e8/train.parquet  # Bank marketing dataset
└── MNIST/                            # MNIST features and targets (parquet)
```

## Concept Slides

LaTeX Beamer presentations in `concept-slides/beamer/`. Compile with:
```bash
cd concept-slides/beamer && pdflatex decision-tree-to-random-forest.tex
```
Generated auxiliary files (`.aux`, `.log`, `.out`, etc.) are gitignored.
