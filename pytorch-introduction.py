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
    return (
        ConfusionMatrixDisplay,
        DataLoader,
        RandomForestClassifier,
        TensorDataset,
        accuracy_score,
        f1_score,
        mo,
        nn,
        np,
        pl,
        plt,
        precision_score,
        recall_score,
        torch,
        train_test_split,
    )


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
def _(
    DataLoader,
    TensorDataset,
    X_test,
    X_train,
    X_val,
    torch,
    y_test,
    y_train,
    y_val,
):
    _X_train_tensor = torch.tensor(
        X_train,
        dtype=torch.float32,
    )
    _y_train_tensor = torch.tensor(
        y_train,
        dtype=torch.long,
    )
    _X_val_tensor = torch.tensor(
        X_val,
        dtype=torch.float32,
    )
    _y_val_tensor = torch.tensor(
        y_val,
        dtype=torch.long,
    )
    X_test_tensor = torch.tensor(
        X_test,
        dtype=torch.float32,
    )
    y_test_tensor = torch.tensor(
        y_test,
        dtype=torch.long,
    )

    _train_dataset = TensorDataset(_X_train_tensor, _y_train_tensor)
    _val_dataset = TensorDataset(_X_val_tensor, _y_val_tensor)
    _test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(
        dataset=_train_dataset,
        batch_size=64,
        shuffle=True,
        generator=torch.Generator().manual_seed(42),
    )
    val_loader = DataLoader(
        dataset=_val_dataset,
        batch_size=64,
        shuffle=False,
    )
    return X_test_tensor, train_loader, val_loader


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Network Architecture

    We define a feedforward neural network with two hidden layers:

    $$
    \mathbf{x} \in \mathbb{R}^{784}
    \xrightarrow{\text{Linear}} \mathbb{R}^{128}
    \xrightarrow{\text{ReLU}}
    \xrightarrow{\text{Linear}} \mathbb{R}^{64}
    \xrightarrow{\text{ReLU}}
    \xrightarrow{\text{Linear}} \mathbb{R}^{10}
    $$

    - **Input layer**: 784 features (one per pixel)
    - **Hidden layer 1**: 128 neurons with ReLU activation
    - **Hidden layer 2**: 64 neurons with ReLU activation
    - **Output layer**: 10 neurons (one per digit class)

    The ReLU (Rectified Linear Unit) activation function $f(x) = \max(0, x)$ introduces
    non-linearity, allowing the network to learn complex decision boundaries. The output
    layer produces raw scores (logits) for each class — the loss function handles
    converting these to probabilities.
    """)
    return


@app.cell
def _(nn, torch):
    class MNISTClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(784, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 10),
            )

        def forward(self, x):
            return self.network(x)

    torch.manual_seed(42)
    model = MNISTClassifier()
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Training the Neural Network

    The training loop repeats these steps for each mini-batch across multiple epochs:

    1. **Forward pass** — compute predictions: $\hat{\mathbf{y}} = f(\mathbf{X}; \theta)$
    2. **Compute loss** — cross-entropy loss measures prediction error
    3. **Backward pass** — compute gradients: $\nabla_\theta \mathcal{L}$
    4. **Update weights** — the optimizer adjusts parameters: $\theta \leftarrow \theta - \alpha \nabla_\theta \mathcal{L}$

    | Hyperparameter | Value | Rationale |
    |---------------|-------|-----------|
    | Learning rate | 0.001 | Adam default; stable convergence |
    | Batch size | 64 | Balance between gradient noise and speed |
    | Epochs | 20 | Sufficient for convergence on MNIST |
    | Optimizer | Adam | Adaptive learning rates per parameter |
    | Loss function | CrossEntropyLoss | Standard for multiclass classification |
    """)
    return


@app.cell
def _(X_test_tensor, model, nn, torch, train_loader, val_loader):
    _criterion = nn.CrossEntropyLoss()
    _optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=0.001,
    )

    NUM_EPOCHS = 20

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for _epoch in range(NUM_EPOCHS):
        model.train()
        _running_loss = 0.0
        _correct = 0
        _total = 0

        for _X_batch, _y_batch in train_loader:
            _optimizer.zero_grad()
            _outputs = model(_X_batch)
            _loss = _criterion(_outputs, _y_batch)
            _loss.backward()
            _optimizer.step()

            _running_loss += _loss.item() * _X_batch.size(0)
            _predicted = _outputs.argmax(dim=1)
            _correct += (_predicted == _y_batch).sum().item()
            _total += _y_batch.size(0)

        _train_loss = _running_loss / _total
        _train_acc = _correct / _total
        train_losses.append(_train_loss)
        train_accuracies.append(_train_acc)

        model.eval()
        _val_loss = 0.0
        _val_correct = 0
        _val_total = 0

        with torch.no_grad():
            for _X_batch, _y_batch in val_loader:
                _outputs = model(_X_batch)
                _loss = _criterion(_outputs, _y_batch)
                _val_loss += _loss.item() * _X_batch.size(0)
                _predicted = _outputs.argmax(dim=1)
                _val_correct += (_predicted == _y_batch).sum().item()
                _val_total += _y_batch.size(0)

        val_losses.append(_val_loss / _val_total)
        val_accuracies.append(_val_correct / _val_total)

        print(
            f"Epoch {_epoch + 1:2d}/{NUM_EPOCHS} — "
            f"Train Loss: {_train_loss:.4f}, Train Acc: {_train_acc:.4f}, "
            f"Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_accuracies[-1]:.4f}"
        )

    model.eval()
    with torch.no_grad():
        _logits = model(X_test_tensor)
        nn_predictions = _logits.argmax(dim=1).numpy()
    return (
        nn_predictions,
        train_accuracies,
        train_losses,
        val_accuracies,
        val_losses,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Training Diagnostics

    The plots below show how loss and accuracy evolve over training epochs:

    - **Loss curves**: Training loss should steadily decrease. If validation loss starts
      increasing while training loss continues to drop, the model is overfitting.
    - **Accuracy curves**: Both should rise and plateau. A large gap between train and
      validation accuracy also indicates overfitting.
    """)
    return


@app.cell
def _(plt, train_accuracies, train_losses, val_accuracies, val_losses):
    _fig, (_ax1, _ax2) = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(10, 4.5),
    )

    _epochs = range(1, len(train_losses) + 1)

    _ax1.plot(
        _epochs,
        train_losses,
        label="Train",
        marker="o",
        markersize=4,
        color="blue",
    )
    _ax1.plot(
        _epochs,
        val_losses,
        label="Validation",
        marker="s",
        markersize=4,
        color="orange",
    )
    _ax1.set_xlabel("Epoch")
    _ax1.set_ylabel("Loss")
    _ax1.set_title("Loss over Epochs")
    _ax1.legend()

    _ax2.plot(
        _epochs,
        train_accuracies,
        label="Train",
        marker="o",
        markersize=4,
        color="blue",
    )
    _ax2.plot(
        _epochs,
        val_accuracies,
        label="Validation",
        marker="s",
        markersize=4,
        color="orange",
    )
    _ax2.set_xlabel("Epoch")
    _ax2.set_ylabel("Accuracy")
    _ax2.set_title("Accuracy over Epochs")
    _ax2.legend()

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Random Forest Baseline

    To put the neural network's performance in context, we train a Random Forest classifier
    on the same normalized features. This connects back to the tree-based models covered in
    earlier notebooks and provides a non-neural baseline for comparison.
    """)
    return


@app.cell
def _(RandomForestClassifier, X_test, X_train, y_train):
    _rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    _rf_model.fit(X_train, y_train)

    rf_predictions = _rf_model.predict(X_test)
    return (rf_predictions,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Results
    """)
    return


@app.cell
def _(
    accuracy_score,
    f1_score,
    nn_predictions,
    pl,
    precision_score,
    recall_score,
    rf_predictions,
    y_test,
):
    _models = [
        ("Neural Network", nn_predictions),
        ("Random Forest", rf_predictions),
    ]

    _names = []
    _accuracies = []
    _precisions = []
    _recalls = []
    _f1s = []

    for _name, _pred in _models:
        _names.append(_name)
        _accuracies.append(accuracy_score(y_test, _pred))
        _precisions.append(
            precision_score(y_test, _pred, average="weighted"),
        )
        _recalls.append(
            recall_score(y_test, _pred, average="weighted"),
        )
        _f1s.append(
            f1_score(y_test, _pred, average="weighted"),
        )

    _summary_df = pl.DataFrame({
        "Model": _names,
        "Accuracy": _accuracies,
        "Precision": _precisions,
        "Recall": _recalls,
        "F1": _f1s,
    })

    _summary_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Confusion Matrices
    """)
    return


@app.cell
def _(ConfusionMatrixDisplay, nn_predictions, plt, rf_predictions, y_test):
    _models = [
        ("Neural Network", nn_predictions),
        ("Random Forest", rf_predictions),
    ]

    _fig, _axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(10, 4.5),
    )

    for _i, (_name, _pred) in enumerate(_models):
        ConfusionMatrixDisplay.from_predictions(
            y_true=y_test,
            y_pred=_pred,
            ax=_axes[_i],
            cmap="Blues",
        )
        _axes[_i].set_title(_name)

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Takeaways

    | | Neural Network | Random Forest |
    |---|---|---|
    | **Approach** | Learns feature representations via backpropagation | Splits on raw features directly |
    | **Hyperparameters** | Architecture, learning rate, epochs, batch size | Number of trees, max depth, split criteria |
    | **Training** | Iterative gradient descent over mini-batches | Parallel tree construction (embarrassingly parallel) |
    | **Scalability** | Scales to large datasets; benefits from GPUs | Memory-intensive with many trees and features |
    | **Interpretability** | Black box; requires techniques like saliency maps | Feature importance available directly |

    **Where to go from here:**

    - **Convolutional Neural Networks (CNNs)** — exploit spatial structure in images rather
      than treating pixels as independent features
    - **Regularization** — dropout layers and weight decay to reduce overfitting
    - **GPU acceleration** — move tensors to GPU with `.to("cuda")` for faster training
    - **Learning rate scheduling** — adjust the learning rate during training for better
      convergence
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
