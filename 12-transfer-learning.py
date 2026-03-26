import marimo

__generated_with = "0.20.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    import seaborn as sns
    import torch
    import torch.nn as nn
    from PIL import Image
    from sklearn.metrics import (
        ConfusionMatrixDisplay,
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    )
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader, Subset
    from torchvision import models, transforms
    from torchvision.datasets import ImageFolder

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
        ImageFolder,
        Path,
        Subset,
        accuracy_score,
        device,
        f1_score,
        mo,
        models,
        nn,
        np,
        pl,
        plt,
        precision_score,
        recall_score,
        torch,
        train_test_split,
        transforms,
        use_bfloat16,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Transfer Learning with ResNet50: Selfie vs Non-Selfie Classification

    This notebook demonstrates **transfer learning**, a technique that adapts a model
    pre-trained on a large dataset to a new, smaller task. We cover three topics:

    1. **What transfer learning is** and why it works
    2. **Fine-tuning a pre-trained ResNet50** by freezing the convolutional backbone and
       training only a new classification head on the selfie dataset
    3. **Model persistence** — saving a trained model to disk and loading it back for
       inference in a new session
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What is Transfer Learning?

    Training a deep convolutional network from scratch requires large amounts of labeled
    data and significant compute time. Transfer learning sidesteps both problems by
    reusing a model that was already trained on a large, general-purpose dataset.

    ResNet50, for example, was trained on ImageNet (1.4 million images across 1,000
    classes). During that training, its convolutional layers learned to detect
    general visual features: edges and textures in the early layers, object parts and
    spatial patterns in the deeper layers. These features transfer well to new image
    classification tasks, even when the target classes are different from the original
    1,000 ImageNet categories.

    The transfer learning workflow has two steps:

    1. **Freeze the backbone.** Lock the pre-trained convolutional layers so their
       weights are not modified during training. Formally, given pre-trained parameters
       $\theta_{\text{backbone}}$, we set $\nabla_{\theta_{\text{backbone}}} \mathcal{L} = 0$.
    2. **Train a new head.** Replace the original 1,000-class output layer with a new
       layer matching the target task (here, 2 classes) and optimize only its parameters
       $\theta_{\text{head}}$.

    Because the backbone already encodes useful visual representations, the new head
    typically converges quickly, even with a small dataset and without a GPU.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The Selfie Dataset

    The dataset contains 7,862 JPEG images split evenly between two classes:

    | Class | Count |
    |---|---|
    | Selfie | 3,931 |
    | NonSelfie | 3,931 |

    The images vary in resolution and aspect ratio. We resize them to 224x224 pixels to
    match ResNet50's expected input size. We use a stratified **80/10/10** split for
    training, validation, and test sets, maintaining the class balance in each split.
    """)
    return


@app.cell
def _(ImageFolder, np, train_test_split, transforms):
    imagenet_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    full_dataset = ImageFolder(
        root="data/selfie_data",
        transform=imagenet_transform,
    )

    class_names = full_dataset.classes
    _targets = np.array(full_dataset.targets)
    _indices = np.arange(len(full_dataset))

    train_indices, _temp_indices, _y_train, _y_temp = train_test_split(
        _indices,
        _targets,
        test_size=0.2,
        stratify=_targets,
        random_state=42,
    )
    val_indices, test_indices, _y_val, _y_test = train_test_split(
        _temp_indices,
        _y_temp,
        test_size=0.5,
        stratify=_y_temp,
        random_state=42,
    )

    print(f"Classes: {dict(zip(class_names, range(len(class_names))))}")
    print(f"Training:   {len(train_indices):,} images")
    print(f"Validation: {len(val_indices):,} images")
    print(f"Test:       {len(test_indices):,} images")
    return class_names, full_dataset, test_indices, train_indices, val_indices


@app.cell
def _(class_names, full_dataset, np, plt, train_indices):
    _rng = np.random.default_rng(seed=42)
    _sample_indices = _rng.choice(
        train_indices,
        size=10,
        replace=False,
    )

    _mean = np.array([0.485, 0.456, 0.406])
    _std = np.array([0.229, 0.224, 0.225])

    _fig, _axes = plt.subplots(
        nrows=2,
        ncols=5,
        figsize=(10, 4.5),
    )

    for _ax, _idx in zip(_axes.flat, _sample_indices):
        _img, _label = full_dataset[_idx]
        _img_np = _img.numpy().transpose(1, 2, 0)
        _img_np = np.clip(_img_np * _std + _mean, 0, 1)
        _ax.imshow(_img_np)
        _ax.set_title(class_names[_label])
        _ax.axis("off")

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## DataLoaders

    We wrap each split in a `torch.utils.data.Subset` (which indexes into the full
    `ImageFolder` dataset) and then in a `DataLoader` for batched iteration. The batch
    size is 32, which keeps memory usage reasonable for 224x224 RGB images. Only the
    training loader shuffles its data; the validation and test loaders iterate in a
    fixed order for reproducibility.
    """)
    return


@app.cell
def _(
    DataLoader,
    Subset,
    full_dataset,
    test_indices,
    torch,
    train_indices,
    val_indices,
):
    _train_subset = Subset(full_dataset, train_indices)
    _val_subset = Subset(full_dataset, val_indices)
    _test_subset = Subset(full_dataset, test_indices)

    train_loader = DataLoader(
        dataset=_train_subset,
        batch_size=32,
        shuffle=True,
        generator=torch.Generator().manual_seed(42),
    )
    val_loader = DataLoader(
        dataset=_val_subset,
        batch_size=32,
        shuffle=False,
    )
    test_loader = DataLoader(
        dataset=_test_subset,
        batch_size=32,
        shuffle=False,
    )
    return test_loader, train_loader, val_loader


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## ResNet50 Architecture

    ResNet50 is a 50-layer deep residual network that uses **skip connections** to allow
    gradients to flow directly through the network, solving the vanishing gradient
    problem that makes very deep networks difficult to train. It was pre-trained on
    ImageNet with a 1,000-class output layer.

    For our binary classification task, we:

    1. Load the pre-trained weights (`IMAGENET1K_V2`, the best available checkpoint)
    2. Freeze all backbone parameters by setting `requires_grad = False`
    3. Replace the final fully connected layer (`Linear(2048, 1000)`) with a new
       `Linear(2048, 2)` layer for our two classes

    | Component | Parameters | Trainable |
    |---|---|---|
    | ResNet50 backbone | ~23.5M | No (frozen) |
    | New classifier head | 4,098 | Yes |
    | **Total** | **~23.5M** | **4,098** |

    Only 0.02% of the model's parameters are updated during training. Before training,
    the new head has random weights, so we evaluate the model in this untrained state
    first to establish a baseline.
    """)
    return


@app.cell
def _(device, models, nn, torch):
    torch.manual_seed(42)

    resnet_model = models.resnet50(weights="IMAGENET1K_V2")

    for _param in resnet_model.parameters():
        _param.requires_grad = False

    resnet_model.fc = nn.Linear(
        in_features=resnet_model.fc.in_features,
        out_features=2,
    )

    resnet_model = resnet_model.to(device)

    _total_params = sum(_p.numel() for _p in resnet_model.parameters())
    _trainable_params = sum(
        _p.numel() for _p in resnet_model.parameters() if _p.requires_grad
    )
    print(f"Total parameters:     {_total_params:,}")
    print(f"Trainable parameters: {_trainable_params:,}")
    return (resnet_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Baseline: Untrained Head

    Before training, the new classification head has random weights. On a balanced
    binary dataset, we expect accuracy near 50%, no better than flipping a coin.
    Evaluating the model in this untrained state establishes a baseline that makes the
    effect of fine-tuning concrete.
    """)
    return


@app.cell
def _(device, nn, np, torch, use_bfloat16):
    def evaluate_model(model, data_loader):
        _autocast = torch.amp.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_bfloat16,
        )

        model.eval()
        _all_predictions = []
        _all_labels = []

        with torch.no_grad(), _autocast:
            for _X_batch, _y_batch in data_loader:
                _X_batch = _X_batch.to(device)
                _outputs = model(_X_batch)
                _predicted = _outputs.argmax(dim=1)
                _all_predictions.append(_predicted.cpu().numpy())
                _all_labels.append(_y_batch.numpy())

        return {
            "predictions": np.concatenate(_all_predictions),
            "labels": np.concatenate(_all_labels),
        }

    def train_model(model, train_loader, val_loader, test_loader, num_epochs=10, lr=0.001):
        _criterion = nn.CrossEntropyLoss()
        _optimizer = torch.optim.Adam(
            params=filter(lambda _p: _p.requires_grad, model.parameters()),
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
                f"Val Loss: {_val_losses[-1]:.4f}, Val Acc: {_val_accuracies[-1]:.4f}",
                flush=True,
            )

        _test_results = evaluate_model(
            model=model,
            data_loader=test_loader,
        )

        return {
            "train_losses": _train_losses,
            "val_losses": _val_losses,
            "train_accuracies": _train_accuracies,
            "val_accuracies": _val_accuracies,
            "predictions": _test_results["predictions"],
            "labels": _test_results["labels"],
        }

    return evaluate_model, train_model


@app.cell
def _(evaluate_model, np, resnet_model, test_loader):
    baseline_results = evaluate_model(
        model=resnet_model,
        data_loader=test_loader,
    )

    _accuracy = np.mean(baseline_results["predictions"] == baseline_results["labels"])
    print(f"Baseline accuracy (random head, no training): {_accuracy:.4f}")
    return (baseline_results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Training

    The baseline confirms that a random head is no better than guessing. We now train
    the head so it learns to map the backbone's feature representations to our two
    classes. Since the backbone is frozen, gradients only flow through the 4,098
    parameters in the new classification head, making each training step fast even on
    CPU.

    | Hyperparameter | Value | Rationale |
    |---|---|---|
    | Learning rate | 0.001 | Adam default; sufficient for a small linear head |
    | Batch size | 32 | Fits 224x224 RGB images in CPU memory |
    | Epochs | 10 | A linear head on frozen features converges quickly |
    | Optimizer | Adam | Adaptive learning rates per parameter |
    | Loss function | CrossEntropyLoss | Standard for classification |
    """)
    return


@app.cell
def _(resnet_model, test_loader, train_loader, train_model, val_loader):
    finetuned_results = train_model(
        model=resnet_model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )
    return (finetuned_results,)


@app.cell
def _(finetuned_results, plt):
    _fig, (_ax1, _ax2) = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(10, 4.5),
    )

    _epochs = range(1, len(finetuned_results["train_losses"]) + 1)

    _ax1.plot(
        _epochs,
        finetuned_results["train_losses"],
        label="Train",
        marker="o",
        markersize=4,
        color="blue",
    )
    _ax1.plot(
        _epochs,
        finetuned_results["val_losses"],
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
        finetuned_results["train_accuracies"],
        label="Train",
        marker="o",
        markersize=4,
        color="blue",
    )
    _ax2.plot(
        _epochs,
        finetuned_results["val_accuracies"],
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
    ## Test Set Evaluation

    We now compare the baseline (random head) against the fine-tuned model on the
    held-out test set. These images were never seen during training or used for any
    decisions, so the metrics give an unbiased estimate of generalization performance.
    """)
    return


@app.cell
def _(
    accuracy_score,
    baseline_results,
    class_names,
    f1_score,
    finetuned_results,
    pl,
    precision_score,
    recall_score,
):
    _models = [
        ("ResNet50 (random head)", baseline_results),
        ("ResNet50 (fine-tuned)", finetuned_results),
    ]

    _names = []
    _accuracies = []
    _precisions = []
    _recalls = []
    _f1s = []

    for _name, _result in _models:
        _names.append(_name)
        _accuracies.append(accuracy_score(_result["labels"], _result["predictions"]))
        _precisions.append(
            precision_score(
                _result["labels"],
                _result["predictions"],
                average="binary",
                pos_label=class_names.index("Selfie"),
            ),
        )
        _recalls.append(
            recall_score(
                _result["labels"],
                _result["predictions"],
                average="binary",
                pos_label=class_names.index("Selfie"),
            ),
        )
        _f1s.append(
            f1_score(
                _result["labels"],
                _result["predictions"],
                average="binary",
                pos_label=class_names.index("Selfie"),
            ),
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


@app.cell
def _(
    ConfusionMatrixDisplay,
    baseline_results,
    class_names,
    finetuned_results,
    plt,
):
    _models = [
        ("Random Head (no training)", baseline_results),
        ("Fine-tuned Head", finetuned_results),
    ]

    _fig, _axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(10, 4.5),
    )

    for _i, (_name, _result) in enumerate(_models):
        ConfusionMatrixDisplay.from_predictions(
            y_true=_result["labels"],
            y_pred=_result["predictions"],
            display_labels=class_names,
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
    ## Model Persistence

    Training takes time, so we want to save the trained model to disk and reload it
    later for inference without retraining. PyTorch offers two approaches:

    | Approach | Saves | Portable | File size |
    |---|---|---|---|
    | `torch.save(model.state_dict(), path)` | Parameter values only | Yes (requires architecture code) | Smaller |
    | `torch.save(model, path)` | Entire model object | Fragile (tied to exact class definitions) | Larger |

    The recommended approach is saving the **state dict** (a dictionary mapping layer
    names to their tensor values). This is more portable and produces smaller files.
    The tradeoff is that you need to reconstruct the model architecture before loading
    the weights back in, but that is straightforward since we are modifying a standard
    `torchvision` model.
    """)
    return


@app.cell
def _(Path, resnet_model, torch):
    model_save_path = Path("data/selfie_data/resnet50_selfie.pth")

    torch.save(
        obj=resnet_model.state_dict(),
        f=model_save_path,
    )

    _size_mb = model_save_path.stat().st_size / (1024 * 1024)
    print(f"Model saved to: {model_save_path}")
    print(f"File size: {_size_mb:.1f} MB")
    return (model_save_path,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Loading a Saved Model

    To use the saved model in a new script or session, we:

    1. Create a fresh ResNet50 (no pre-trained weights needed this time)
    2. Replace the final layer with `Linear(2048, 2)`, matching the architecture we
       trained
    3. Load the saved state dict with `torch.load` and `model.load_state_dict`
    4. Set the model to evaluation mode

    This simulates the scenario where training and inference happen in separate scripts
    or at different times.
    """)
    return


@app.cell
def _(device, model_save_path, models, nn, torch):
    loaded_model = models.resnet50(weights=None)

    loaded_model.fc = nn.Linear(
        in_features=loaded_model.fc.in_features,
        out_features=2,
    )

    loaded_model.load_state_dict(
        torch.load(
            f=model_save_path,
            map_location=device,
            weights_only=True,
        )
    )

    loaded_model = loaded_model.to(device)
    loaded_model.eval()

    print("Model loaded successfully from disk")
    return (loaded_model,)


@app.cell
def _(
    class_names,
    device,
    full_dataset,
    loaded_model,
    np,
    plt,
    test_indices,
    torch,
    use_bfloat16,
):
    _rng = np.random.default_rng(seed=42)
    _sample_indices = _rng.choice(
        test_indices,
        size=8,
        replace=False,
    )

    _mean = np.array([0.485, 0.456, 0.406])
    _std = np.array([0.229, 0.224, 0.225])

    _autocast = torch.amp.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=use_bfloat16,
    )

    _fig, _axes = plt.subplots(
        nrows=2,
        ncols=4,
        figsize=(12, 6),
    )

    for _ax, _idx in zip(_axes.flat, _sample_indices):
        _img, _true_label = full_dataset[_idx]
        _input = _img.unsqueeze(0).to(device)

        with torch.no_grad(), _autocast:
            _logits = loaded_model(_input)
            _probs = torch.softmax(_logits, dim=1)
            _pred_label = _logits.argmax(dim=1).item()
            _confidence = _probs[0, _pred_label].item()

        _img_np = _img.numpy().transpose(1, 2, 0)
        _img_np = np.clip(_img_np * _std + _mean, 0, 1)
        _ax.imshow(_img_np)

        _true_name = class_names[_true_label]
        _pred_name = class_names[_pred_label]
        _color = "green" if _pred_label == _true_label else "red"
        _ax.set_title(
            f"True: {_true_name}\nPred: {_pred_name} ({_confidence:.1%})",
            color=_color,
            fontsize=9,
        )
        _ax.axis("off")

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## False Positives and False Negatives

    The confusion matrix tells us *how many* images were misclassified, but inspecting
    the images themselves can reveal *why*. With Selfie as the positive class:

    - **False positive** — a NonSelfie image the model predicted as Selfie
    - **False negative** — a Selfie image the model predicted as NonSelfie

    Looking at these errors often surfaces patterns: perhaps the model struggles with
    certain camera angles, lighting conditions, or image compositions that blur the line
    between the two classes.
    """)
    return


@app.cell
def _(class_names, finetuned_results, full_dataset, np, plt, test_indices):
    _preds = finetuned_results["predictions"]
    _labels = finetuned_results["labels"]

    _selfie_idx = class_names.index("Selfie")
    _nonselfie_idx = class_names.index("NonSelfie")

    _fp_mask = (_preds == _selfie_idx) & (_labels == _nonselfie_idx)
    _fn_mask = (_preds == _nonselfie_idx) & (_labels == _selfie_idx)
    _fp_positions = np.where(_fp_mask)[0]
    _fn_positions = np.where(_fn_mask)[0]

    print(f"False positives: {len(_fp_positions)}")
    print(f"False negatives: {len(_fn_positions)}")

    _mean = np.array([0.485, 0.456, 0.406])
    _std = np.array([0.229, 0.224, 0.225])

    _rng = np.random.default_rng(seed=42)
    _n_fp = min(4, len(_fp_positions))
    _n_fn = min(4, len(_fn_positions))
    _fp_sample = _rng.choice(_fp_positions, size=_n_fp, replace=False)
    _fn_sample = _rng.choice(_fn_positions, size=_n_fn, replace=False)

    _fig, _axes = plt.subplots(
        nrows=2,
        ncols=4,
        figsize=(12, 6),
    )

    for _i, _ax in enumerate(_axes[0]):
        if _i < _n_fp:
            _img, _ = full_dataset[test_indices[_fp_sample[_i]]]
            _img_np = _img.numpy().transpose(1, 2, 0)
            _img_np = np.clip(_img_np * _std + _mean, 0, 1)
            _ax.imshow(_img_np)
            _ax.set_title(
                f"True: NonSelfie\nPred: Selfie",
                color="red",
                fontsize=9,
            )
        _ax.axis("off")

    for _i, _ax in enumerate(_axes[1]):
        if _i < _n_fn:
            _img, _ = full_dataset[test_indices[_fn_sample[_i]]]
            _img_np = _img.numpy().transpose(1, 2, 0)
            _img_np = np.clip(_img_np * _std + _mean, 0, 1)
            _ax.imshow(_img_np)
            _ax.set_title(
                f"True: Selfie\nPred: NonSelfie",
                color="red",
                fontsize=9,
            )
        _ax.axis("off")

    _axes[0, 0].set_ylabel("False Positives", fontsize=11, fontweight="bold")
    _axes[1, 0].set_ylabel("False Negatives", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Takeaways

    | | Random Head (no training) | Fine-tuned Head | From Scratch (notebook 11 CNN) |
    |---|---|---|---|
    | **Pre-trained weights** | Yes (ImageNet) | Yes (ImageNet) | No |
    | **Head trained** | No | Yes | Yes (all layers) |
    | **Test accuracy** | ~50% (random chance) | High | Moderate |
    | **Training time** | None | Seconds per epoch | Minutes per epoch |

    The random-head baseline demonstrates that the pre-trained backbone alone does not
    solve the new task. The backbone extracts general visual features, but the
    classification head must be trained to map those features to the correct class
    labels. Fine-tuning just 4,098 parameters (0.02% of the model) bridges this gap.

    **Where to go from here:**

    - **Gradual unfreezing** — after training the head, unfreeze the last few backbone
      layers and continue training with a smaller learning rate to fine-tune the
      high-level features
    - **Data augmentation** — random horizontal flips, color jitter, and random rotation
      during training to improve generalization
    - **Learning rate scheduling** — reduce the learning rate as training progresses
      (e.g., with `StepLR` or `CosineAnnealingLR`)
    - **Other backbones** — ResNet18 for faster inference, EfficientNet for a better
      accuracy-efficiency tradeoff
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
