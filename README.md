# BAN-501 Spring 2026 - In-Class Work in Progress

Course materials for BAN-501, featuring interactive notebooks built with [marimo](https://marimo.io/).

## Contents

| File | Topic |
|------|-------|
| `1-linear-regression.py` | Linear regression from scratch using hill-climbing optimization, with statsmodels verification |
| `2-lasso-ridge-regression.py` | Ridge, Lasso, and Elastic Net regression with OLS baseline and RMSE evaluation |
| `3-logistic-regression.py` | Logistic regression for binary classification with odds ratios and model evaluation |
| `4-tree-based-models.py` | Decision trees and random forests with GridSearchCV and Optuna tuning, logistic regression comparison, and permutation importance |
| `5-xgboost.py` | XGBoost classification with Optuna hyperparameter tuning, compared against Random Forest |
| `6-concurrency.py` | Python concurrency concepts (threads vs processes), using helpers from `_concurrency_helpers.py` |
| `7-dimensionality-reduction-demo.py` | PCA and PaCMAP demo on MNIST, Random Forest on reduced features |
| `8-dimensionality-reduction.py` | Full dimensionality reduction pipeline on bank marketing data with PCA, PaCMAP, and classification |
| `9-clustering.py` | K-Means and HDBSCAN on synthetic data (blobs, circles), density-based vs centroid-based comparison |
| `10-pytorch-introduction.py` | Feedforward neural networks on MNIST with PyTorch |
| `11-pytorch-cnn.py` | CNNs vs feedforward networks on MNIST |
| `12-transfer-learning.py` | Fine-tuning pre-trained ResNet50 on selfie/non-selfie classification, with model persistence |
| `13-tfidf-vs-sentence-transformers.py` | TF-IDF and sentence-transformer embeddings on Amazon reviews, with PaCMAP visualization |
| `14-transformer-classification.py` | Fine-tuning DistilBERT for 5-class star-rating classification on Amazon reviews |

Concept slides live in `concept-slides/beamer/` as LaTeX Beamer source, with compiled PDFs in `concept-slides/`.

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

This creates an isolated environment with all required packages. The project defines two environments: `default` (CPU-only PyTorch) and `gpu` (CUDA 13.0 PyTorch, required for notebooks 10-14 to use GPU acceleration).

### Running Notebooks

Run a notebook as a Python script:

```bash
MPLBACKEND=Agg pixi run python 1-linear-regression.py
```

For notebooks that benefit from GPU acceleration (10-14), use the `gpu` environment:

```bash
MPLBACKEND=Agg pixi run -e gpu python 11-pytorch-cnn.py
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
- **Deep learning**: pytorch (CPU or GPU), torchvision, transformers, accelerate, sentence-transformers
- **Dimensionality reduction**: pacmap
- **Visualization**: matplotlib, seaborn
- **Notebooks**: marimo, ipywidgets, ipython, tqdm
