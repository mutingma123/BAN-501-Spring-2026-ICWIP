import marimo

__generated_with = "0.20.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    import seaborn as sns
    import torch
    import torch.nn as nn
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        ConfusionMatrixDisplay,
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    )
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader, TensorDataset

    sns.set_style("whitegrid")
    return mo, np, pl, plt, train_test_split


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Introduction to PyTorch: Feedforward Neural Networks on MNIST

    This notebook introduces **PyTorch** by building a feedforward neural network for
    handwritten digit classification on the MNIST dataset. We cover:

    1. **PyTorch fundamentals** — tensors, autograd, `nn.Module`
    2. **Data preparation** — scaling, tensors, DataLoaders
    3. **Model definition** — a two-hidden-layer feedforward network
    4. **Training loop** — forward pass, loss, backpropagation, optimizer step
    5. **Evaluation** — comparison against a Random Forest baseline
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What is PyTorch?

    PyTorch is an open-source deep learning framework built around four core ideas:

    - **Tensors** — multi-dimensional arrays (like NumPy arrays) that can run on GPUs
    - **Autograd** — automatic differentiation that tracks operations on tensors and computes
      gradients for backpropagation
    - **`nn.Module`** — a base class for defining neural network layers and architectures
    - **Optimizers** — algorithms (e.g., Adam, SGD) that update model parameters using the
      computed gradients

    The typical PyTorch workflow is:

    1. Define a model as a subclass of `nn.Module`
    2. Pass input data through the model (forward pass)
    3. Compute a loss function
    4. Call `loss.backward()` to compute gradients (backward pass)
    5. Call `optimizer.step()` to update weights
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## MNIST Dataset

    MNIST contains 70,000 grayscale images of handwritten digits (0–9). Each image is
    28x28 pixels, flattened into a vector of **784 features**. This is the same dataset
    used in the dimensionality reduction notebook, but here we use the full dataset with
    a stratified 80/10/10 train/validation/test split.
    """)
    return


@app.cell
def _(np, pl, train_test_split):
    _features = pl.read_parquet("data/MNIST/mnist_features.parquet")
    _targets = pl.read_parquet("data/MNIST/mnist_target.parquet")

    _X = _features.to_numpy().astype(np.float64) / 255.0
    _y = _targets.to_numpy().ravel().astype(np.int64)

    X_train, _X_temp, y_train, _y_temp = train_test_split(
        _X,
        _y,
        test_size=0.2,
        stratify=_y,
        random_state=42,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        _X_temp,
        _y_temp,
        test_size=0.5,
        stratify=_y_temp,
        random_state=42,
    )
    return X_test, X_train, X_val, y_test, y_train, y_val


@app.cell
def _(X_train, np, plt, y_train):
    _fig, _axes = plt.subplots(
        nrows=2,
        ncols=5,
        figsize=(10, 4.5),
    )

    _rng = np.random.default_rng(seed=42)
    for _ax in _axes.flat:
        _idx = _rng.integers(
            low=0,
            high=len(X_train),
        )
        _ax.imshow(
            X_train[_idx].reshape(28, 28),
            cmap="gray",
        )
        _ax.set_title(f"Label: {y_train[_idx]}")
        _ax.axis("off")

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Feature Scaling

    Neural networks are sensitive to the scale of input features. Pixel values originally
    range from 0 to 255, so we divide by 255 to rescale them to [0, 1]. This is simpler
    than standardization and well-suited for image data where the value bounds are known.
    The normalization is applied during data loading above.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## PyTorch Data Preparation

    PyTorch models operate on **tensors**, not NumPy arrays. We need to:

    1. Convert NumPy arrays to `torch.Tensor` objects
    2. Wrap them in a `TensorDataset` (pairs features with labels)
    3. Create `DataLoader` objects that yield **mini-batches** during training

    Mini-batch training processes a small subset of the data (here, 64 samples) at each
    step rather than the entire dataset. This provides a good balance between the noisy
    gradients of single-sample updates and the computational cost of full-batch updates.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
