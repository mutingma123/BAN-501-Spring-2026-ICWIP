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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bfloat16 = (
        torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    )
    print(f"Using device: {device}")
    if use_bfloat16:
        print("bfloat16 supported — enabling mixed-precision training")
    return (
        ConfusionMatrixDisplay,
        DataLoader,
        TensorDataset,
        accuracy_score,
        device,
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
        use_bfloat16,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Convolutional Neural Networks vs Feedforward Networks on MNIST

    This notebook compares two neural network architectures for handwritten digit
    classification on the MNIST dataset:

    1. **Feedforward Neural Network** — the fully connected architecture from the
       previous notebook, which treats each pixel as an independent feature
    2. **Convolutional Neural Network (CNN)** — an architecture that exploits the
       spatial structure of images through local receptive fields and weight sharing
    3. **Side-by-side comparison** — training dynamics, classification metrics, and
       confusion matrices for both models under identical training conditions
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## MNIST Dataset

    MNIST contains 70,000 grayscale images of handwritten digits (0-9). Each image is
    28x28 pixels, stored as a flattened vector of **784 features** in the parquet files.
    The feedforward network consumes these flat vectors directly, while the CNN reshapes
    them back into 2D images internally.

    We use a stratified 80/10/10 train/validation/test split.
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
    range from 0 to 255, so we divide by 255 to rescale them to [0, 1]. This normalization
    is applied during data loading above and is used by both models.
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

    Both models share the same DataLoaders. The data is stored as flat 784-element
    vectors; the CNN reshapes these to 28x28 images inside its `forward()` method.
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
    ## Feedforward Network Architecture

    The feedforward network has two hidden layers with ReLU activations:

    $$
    \mathbf{x} \in \mathbb{R}^{784}
    \xrightarrow{\text{Linear}} \mathbb{R}^{128}
    \xrightarrow{\text{ReLU}}
    \xrightarrow{\text{Linear}} \mathbb{R}^{64}
    \xrightarrow{\text{ReLU}}
    \xrightarrow{\text{Linear}} \mathbb{R}^{10}
    $$

    This architecture treats each of the 784 pixels as an independent input feature.
    There is no notion of spatial proximity: pixel (0, 0) and pixel (27, 27) are
    equally "far apart" from the network's perspective. The model has approximately
    109,000 trainable parameters.
    """)
    return


@app.cell
def _(device, nn, torch):
    class FeedforwardClassifier(nn.Module):
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
    ffn_model = FeedforwardClassifier().to(device)
    return (ffn_model,)


@app.cell
def _(device, nn, torch, use_bfloat16):
    def train_model(model, train_loader, val_loader, X_test_tensor, num_epochs=20, lr=0.001):
        _criterion = nn.CrossEntropyLoss()
        _optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=lr,
        )
        _autocast = torch.amp.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_bfloat16,
        )

        _train_losses = []
        _val_losses = []
        _train_accuracies = []
        _val_accuracies = []

        for _epoch in range(num_epochs):
            model.train()
            _running_loss = 0.0
            _correct = 0
            _total = 0

            for _X_batch, _y_batch in train_loader:
                _X_batch = _X_batch.to(device)
                _y_batch = _y_batch.to(device)
                _optimizer.zero_grad()
                with _autocast:
                    _outputs = model(_X_batch)
                    _loss = _criterion(_outputs, _y_batch)
                _loss.backward()
                _optimizer.step()

                _running_loss += _loss.item() * _X_batch.size(0)
                _predicted = _outputs.argmax(dim=1)
                _correct += (_predicted == _y_batch).sum().item()
                _total += _y_batch.size(0)

            _epoch_train_loss = _running_loss / _total
            _epoch_train_acc = _correct / _total
            _train_losses.append(_epoch_train_loss)
            _train_accuracies.append(_epoch_train_acc)

            model.eval()
            _val_loss = 0.0
            _val_correct = 0
            _val_total = 0

            with torch.no_grad(), _autocast:
                for _X_batch, _y_batch in val_loader:
                    _X_batch = _X_batch.to(device)
                    _y_batch = _y_batch.to(device)
                    _outputs = model(_X_batch)
                    _loss = _criterion(_outputs, _y_batch)
                    _val_loss += _loss.item() * _X_batch.size(0)
                    _predicted = _outputs.argmax(dim=1)
                    _val_correct += (_predicted == _y_batch).sum().item()
                    _val_total += _y_batch.size(0)

            _val_losses.append(_val_loss / _val_total)
            _val_accuracies.append(_val_correct / _val_total)

            print(
                f"Epoch {_epoch + 1:2d}/{num_epochs} — "
                f"Train Loss: {_epoch_train_loss:.4f}, Train Acc: {_epoch_train_acc:.4f}, "
                f"Val Loss: {_val_losses[-1]:.4f}, Val Acc: {_val_accuracies[-1]:.4f}"
            )

        model.eval()
        with torch.no_grad(), _autocast:
            _logits = model(X_test_tensor.to(device))
            _predictions = _logits.argmax(dim=1).cpu().numpy()

        return {
            "train_losses": _train_losses,
            "val_losses": _val_losses,
            "train_accuracies": _train_accuracies,
            "val_accuracies": _val_accuracies,
            "predictions": _predictions,
        }

    return (train_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Training the Feedforward Network

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
def _(X_test_tensor, ffn_model, train_loader, train_model, val_loader):
    ffn_results = train_model(
        model=ffn_model,
        train_loader=train_loader,
        val_loader=val_loader,
        X_test_tensor=X_test_tensor,
    )
    return (ffn_results,)


@app.cell
def _(ffn_results, plt):
    _fig, (_ax1, _ax2) = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(10, 4.5),
    )

    _epochs = range(1, len(ffn_results["train_losses"]) + 1)

    _ax1.plot(
        _epochs,
        ffn_results["train_losses"],
        label="Train",
        marker="o",
        markersize=4,
        color="blue",
    )
    _ax1.plot(
        _epochs,
        ffn_results["val_losses"],
        label="Validation",
        marker="s",
        markersize=4,
        color="orange",
    )
    _ax1.set_xlabel("Epoch")
    _ax1.set_ylabel("Loss")
    _ax1.set_title("FFN: Loss over Epochs")
    _ax1.legend()

    _ax2.plot(
        _epochs,
        ffn_results["train_accuracies"],
        label="Train",
        marker="o",
        markersize=4,
        color="blue",
    )
    _ax2.plot(
        _epochs,
        ffn_results["val_accuracies"],
        label="Validation",
        marker="s",
        markersize=4,
        color="orange",
    )
    _ax2.set_xlabel("Epoch")
    _ax2.set_ylabel("Accuracy")
    _ax2.set_title("FFN: Accuracy over Epochs")
    _ax2.legend()

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Why Convolutions?

    The feedforward network ignores the spatial layout of pixels entirely. A pixel in the
    top-left corner is no more "related" to its neighbors than to a pixel in the
    bottom-right corner. For image data, this discards useful structure.

    Convolutional neural networks address this by introducing three key ideas:

    - **Local receptive fields** — each neuron connects to a small patch of the input
      (e.g., 3x3 pixels) rather than the entire image, so it learns local patterns
      like edges and corners
    - **Weight sharing** — the same filter (kernel) slides across every position in the
      image, so a pattern learned in one location is detected everywhere
    - **Pooling** — max pooling reduces spatial dimensions by summarizing small
      regions, providing some translation invariance and reducing computation
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## CNN Architecture

    The CNN processes the input as a 2D image (1x28x28) through two convolutional
    blocks, then flattens the result and passes it through fully connected layers:

    | Layer | Output Shape | Parameters |
    |---|---|---|
    | Input | 1 x 28 x 28 | — |
    | Conv2d(1, 32, 3) + ReLU | 32 x 26 x 26 | 320 |
    | MaxPool2d(2) | 32 x 13 x 13 | 0 |
    | Conv2d(32, 64, 3) + ReLU | 64 x 11 x 11 | 18,496 |
    | MaxPool2d(2) | 64 x 5 x 5 | 0 |
    | Flatten | 1,600 | 0 |
    | Linear(1600, 128) + ReLU | 128 | 204,928 |
    | Linear(128, 10) | 10 | 1,290 |
    | **Total** | | **225,034** |

    The convolutional layers themselves use relatively few parameters (about 19,000)
    because each filter is small (3x3) and shared across all spatial positions. Most
    of the parameters are in the first fully connected layer, which bridges the
    transition from spatial feature maps to class predictions.

    **Reshaping the input.** Our data is stored as flat vectors of 784 values, but
    `Conv2d` expects a 4D tensor with shape `(batch_size, channels, height, width)`.
    The `forward()` method calls `x.view(-1, 1, 28, 28)` to perform this reshape.
    Each of the four dimensions has a specific meaning: `-1` tells PyTorch to infer
    the batch size automatically from however many samples are in the current
    mini-batch, `1` is the number of color channels (grayscale), and `28, 28` are
    the image height and width. This operation does not copy or modify the underlying
    data; it reinterprets the same 784 values as a 28x28 grid. After the
    convolutional layers produce feature maps of shape `(batch_size, 64, 5, 5)`, a
    second reshape (`x.view(x.size(0), -1)`) flattens them back to
    `(batch_size, 1600)` for the fully connected layers.
    """)
    return


@app.cell
def _(device, nn, torch):
    class CNNClassifier(nn.Module):
        ...

    torch.manual_seed(42)
    cnn_model = CNNClassifier().to(device)
    return (cnn_model,)


@app.cell
def _(X_test_tensor, cnn_model, train_loader, train_model, val_loader):
    cnn_results = train_model(
        model=cnn_model,
        train_loader=train_loader,
        val_loader=val_loader,
        X_test_tensor=X_test_tensor,
    )
    return (cnn_results,)


@app.cell
def _(cnn_results, plt):
    _fig, (_ax1, _ax2) = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(10, 4.5),
    )

    _epochs = range(1, len(cnn_results["train_losses"]) + 1)

    _ax1.plot(
        _epochs,
        cnn_results["train_losses"],
        label="Train",
        marker="o",
        markersize=4,
        color="blue",
    )
    _ax1.plot(
        _epochs,
        cnn_results["val_losses"],
        label="Validation",
        marker="s",
        markersize=4,
        color="orange",
    )
    _ax1.set_xlabel("Epoch")
    _ax1.set_ylabel("Loss")
    _ax1.set_title("CNN: Loss over Epochs")
    _ax1.legend()

    _ax2.plot(
        _epochs,
        cnn_results["train_accuracies"],
        label="Train",
        marker="o",
        markersize=4,
        color="blue",
    )
    _ax2.plot(
        _epochs,
        cnn_results["val_accuracies"],
        label="Validation",
        marker="s",
        markersize=4,
        color="orange",
    )
    _ax2.set_xlabel("Epoch")
    _ax2.set_ylabel("Accuracy")
    _ax2.set_title("CNN: Accuracy over Epochs")
    _ax2.legend()

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model Comparison

    Both models were trained with identical hyperparameters (Adam optimizer, learning
    rate 0.001, batch size 64, 20 epochs) on the same data splits. The plots and
    metrics below isolate the effect of the architectural difference.
    """)
    return


@app.cell
def _(cnn_results, ffn_results, plt):
    _fig, (_ax1, _ax2) = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(10, 4.5),
    )

    _epochs = range(1, len(ffn_results["train_losses"]) + 1)

    _ax1.plot(
        _epochs,
        ffn_results["train_losses"],
        label="FFN Train",
        color="blue",
        linestyle="-",
        marker="o",
        markersize=3,
    )
    _ax1.plot(
        _epochs,
        ffn_results["val_losses"],
        label="FFN Val",
        color="blue",
        linestyle="--",
        marker="s",
        markersize=3,
    )
    _ax1.plot(
        _epochs,
        cnn_results["train_losses"],
        label="CNN Train",
        color="red",
        linestyle="-",
        marker="o",
        markersize=3,
    )
    _ax1.plot(
        _epochs,
        cnn_results["val_losses"],
        label="CNN Val",
        color="red",
        linestyle="--",
        marker="s",
        markersize=3,
    )
    _ax1.set_xlabel("Epoch")
    _ax1.set_ylabel("Loss")
    _ax1.set_title("Loss Comparison")
    _ax1.legend()

    _ax2.plot(
        _epochs,
        ffn_results["train_accuracies"],
        label="FFN Train",
        color="blue",
        linestyle="-",
        marker="o",
        markersize=3,
    )
    _ax2.plot(
        _epochs,
        ffn_results["val_accuracies"],
        label="FFN Val",
        color="blue",
        linestyle="--",
        marker="s",
        markersize=3,
    )
    _ax2.plot(
        _epochs,
        cnn_results["train_accuracies"],
        label="CNN Train",
        color="red",
        linestyle="-",
        marker="o",
        markersize=3,
    )
    _ax2.plot(
        _epochs,
        cnn_results["val_accuracies"],
        label="CNN Val",
        color="red",
        linestyle="--",
        marker="s",
        markersize=3,
    )
    _ax2.set_xlabel("Epoch")
    _ax2.set_ylabel("Accuracy")
    _ax2.set_title("Accuracy Comparison")
    _ax2.legend()

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Results
    """)
    return


@app.cell
def _(
    accuracy_score,
    cnn_results,
    f1_score,
    ffn_results,
    pl,
    precision_score,
    recall_score,
    y_test,
):
    _models = [
        ("Feedforward NN", ffn_results["predictions"]),
        ("CNN", cnn_results["predictions"]),
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
def _(ConfusionMatrixDisplay, cnn_results, ffn_results, plt, y_test):
    _models = [
        ("Feedforward NN", ffn_results["predictions"]),
        ("CNN", cnn_results["predictions"]),
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

    | | Feedforward NN | CNN |
    |---|---|---|
    | **Input representation** | Flat vector (784) | 2D image (1x28x28) |
    | **Spatial awareness** | None; pixel position ignored | Local patterns via 3x3 kernels |
    | **Weight sharing** | None; every connection is unique | Conv filters shared across all positions |
    | **Parameters** | ~109K | ~225K |

    The CNN outperforms the feedforward network despite both using the same optimizer,
    learning rate, and number of epochs. The convolutional layers learn local features
    (edges, corners, strokes) once and detect them everywhere in the image, while the
    pooling layers provide translation invariance. The feedforward network, by contrast,
    must independently learn what each pixel means at each position.

    **Where to go from here:**

    - **Regularization** — dropout layers and batch normalization to reduce overfitting
    - **Deeper architectures** — adding more convolutional blocks or using residual
      connections (ResNets)
    - **Data augmentation** — random rotations, shifts, and scaling to increase the
      effective training set size
    - **GPU acceleration** — this notebook automatically uses CUDA when available;
      run with the `gpu` environment in pixi for GPU-accelerated training
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
