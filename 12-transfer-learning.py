import marimo

__generated_with = "0.20.2"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Transfer Learning: Fine-Tuning a Pretrained CNN

    Training a deep convolutional network from scratch requires large datasets and substantial compute time. Transfer learning sidesteps both constraints by reusing a model that has already learned general visual features from a large corpus. In this notebook we take a ResNet50 backbone pretrained on ImageNet (1.2 million images, 1,000 classes) and attach a small classification head for a binary task: distinguishing selfies from non-selfies.

    The workflow has three phases:

    1. **Pretrained backbone + custom head** -- freeze the convolutional layers and replace the final classifier
    2. **Fine-tuning** -- train only the new head while the frozen backbone extracts features
    3. **Model persistence and inference** -- save the trained weights, reload them, and classify new images
    """)
    return


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
    from PIL import Image
    from torchvision.datasets import ImageFolder

    # seaborn whitegrid gives clean axis lines for all plots
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
        Image,
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
        sns,
        torch,
        train_test_split,
        transforms,
        use_bfloat16,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Why Transfer Learning?

    A CNN trained on ImageNet learns a hierarchy of visual features. Early layers detect edges and textures, middle layers combine those into shapes, and deep layers recognize object parts. These features transfer well to new image tasks because the low-level and mid-level patterns are shared across domains.

    Rather than learning all of these features from our relatively small selfie dataset (~7,800 images), we keep the pretrained convolutional layers fixed and only train a lightweight classification head on top. This strategy offers two practical benefits. First, training is fast because gradient computation skips the frozen backbone entirely. Second, the model is less prone to overfitting because the vast majority of its parameters are not updated on our small dataset.

    Formally, if the pretrained backbone is a function $g_\phi: \mathbb{R}^{3 \times 224 \times 224} \to \mathbb{R}^{d}$ with frozen parameters $\phi$, we learn only a linear classifier $h_\theta: \mathbb{R}^{d} \to \mathbb{R}^{K}$ by minimizing the cross-entropy loss over the training set:

    $$
    \mathcal{L}(\theta) = -\frac{1}{N}\sum_{i=1}^{N} \log \frac{\exp(h_\theta(g_\phi(\mathbf{x}_i))_{y_i})}{\sum_{k=1}^{K}\exp(h_\theta(g_\phi(\mathbf{x}_i))_k)}
    $$

    where $K=2$ (selfie vs. non-selfie) and only $\theta$ receives gradient updates.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data Loading and ImageNet Preprocessing

    The pretrained backbone expects inputs that match the preprocessing it saw during pretraining. For ResNet50 trained on ImageNet, this means resizing images to 232 pixels on the shorter side, center-cropping to $224 \times 224$, converting to a tensor, and normalizing each color channel with the ImageNet training set statistics:

    $$
    \hat{x}_c = \frac{x_c - \mu_c}{\sigma_c}, \quad \mu = [0.485,\; 0.456,\; 0.406], \quad \sigma = [0.229,\; 0.224,\; 0.225]
    $$

    Skipping this normalization or using different statistics would produce activations the backbone was never calibrated for, degrading the quality of the extracted features.

    The `ImageFolder` class reads images from a directory structure where each subfolder name becomes a class label. We then split the indices into stratified 80/10/10 partitions to preserve the class balance across training, validation, and test sets.
    """)
    return


@app.cell
def _(ImageFolder, np, train_test_split, transforms):
    # ImageNet channel means and stds -- must match the preprocessing
    # the backbone saw during pre-training
    imagenet_transform = transforms.Compose([
        transforms.Resize(232),
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

    # Stratified 80/10/10 split preserves class balance in each partition
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
    return class_names, full_dataset, imagenet_transform, test_indices, train_indices, val_indices


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Sample Training Images

    The images below are sampled from the training partition and un-normalized for display. Because the ImageNet transform normalizes each channel independently, we reverse that operation ($x_c = \hat{x}_c \cdot \sigma_c + \mu_c$) before rendering. The original pixel values are clipped to $[0, 1]$ to avoid display artifacts.
    """)
    return


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
        # Reverse the ImageNet normalization to recover viewable pixel values
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

    PyTorch `DataLoader` objects batch, shuffle, and iterate over the dataset during training. The training loader shuffles images each epoch so the model does not memorize the order of examples. Validation and test loaders iterate in a fixed order for reproducible evaluation. A batch size of 32 is a common starting point for image classification: small enough to fit in memory alongside the ResNet50 backbone, large enough to provide stable gradient estimates.
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

    # Only the training loader shuffles; val and test iterate in fixed order for reproducibility
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
    ## Model Architecture: Frozen Backbone + Trainable Head

    ResNet50 is a 50-layer deep residual network with roughly 23.5 million parameters. Loading the `IMAGENET1K_V2` weights gives us a backbone that already produces rich feature representations for natural images.

    We freeze every parameter in the backbone by setting `requires_grad = False`, which tells PyTorch to skip these parameters during backpropagation. The original classification head is a linear layer that maps the 2,048-dimensional feature vector to 1,000 ImageNet classes. We replace it with a new `nn.Linear(2048, 2)` layer whose weights are randomly initialized. Only this layer, with $2{,}048 \times 2 + 2 = 4{,}098$ trainable parameters, will be updated during fine-tuning.

    This setup is sometimes called "linear probing" because we are fitting a linear classifier on top of frozen features. It is the simplest form of transfer learning and a natural first step before deciding whether to unfreeze deeper layers.
    """)
    return


@app.cell
def _(device, models, nn, torch):
    torch.manual_seed(42)

    resnet_model = models.resnet50(weights="IMAGENET1K_V2")

    # Freeze all 23.5M backbone parameters so gradients do not update them
    for _param in resnet_model.parameters():
        _param.requires_grad = False

    # Replace the 1000-class ImageNet head with a 2-class head (4,098 trainable params)
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
    ## Training and Evaluation Functions

    The training loop follows the same structure as the previous notebooks: forward pass, loss computation, backward pass, optimizer step. Two details differ from training a model from scratch.

    First, the optimizer receives only parameters where `requires_grad` is `True`. Passing the full parameter list would work but waste memory on optimizer state for frozen weights. Filtering up front keeps memory usage proportional to the trainable parameter count.

    Second, mixed-precision training with `torch.amp.autocast` is used when the GPU supports `bfloat16`. The forward pass runs in lower precision for speed, while gradient accumulation remains in `float32` for numerical stability. On CPU this is a no-op.
    """)
    return


@app.cell
def _(device, nn, np, torch, use_bfloat16):
    # Shared evaluation and training functions used by both baseline and fine-tuned runs
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
        # Only pass parameters where requires_grad=True -- the frozen backbone params are excluded
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Baseline: Random Head (No Training)

    Before fine-tuning, the classification head contains random weights. The frozen backbone still extracts meaningful features, but the head maps them to class predictions arbitrarily. On a balanced binary task we expect roughly 50% accuracy, no better than a coin flip. This baseline establishes the starting point so we can measure how much the head learns during fine-tuning.
    """)
    return


@app.cell
def _(evaluate_model, np, resnet_model, test_loader):
    # Baseline: the randomly initialized head maps backbone features arbitrarily, expect ~50% accuracy
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
    ## Fine-Tuning the Classification Head

    We now train only the 4,098 parameters in the classification head. Because the backbone is frozen, each training epoch involves far less computation than training the full network. The optimizer (Adam, learning rate 0.001) updates the head weights while the backbone's 23.5 million parameters remain fixed throughout.
    """)
    return


@app.cell
def _(resnet_model, test_loader, train_loader, train_model, val_loader):
    # Fine-tune only the fc head; backbone stays frozen so each epoch is fast even on CPU
    finetuned_results = train_model(
        model=resnet_model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_epochs=5,
    )
    return (finetuned_results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Training Diagnostics

    The plots below show loss and accuracy for both training and validation sets across epochs. Because only a small head is being trained on top of strong frozen features, convergence is typically rapid. Watch for validation loss diverging from training loss, which would indicate overfitting of the head.
    """)
    return


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
    ## Results

    The table below compares the random-head baseline against the fine-tuned model using accuracy, precision, recall, and F1 score on the held-out test set. Precision and recall are computed with the "Selfie" class as the positive label.
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Confusion Matrices

    The confusion matrices below provide a more detailed breakdown than scalar metrics. Each cell shows the count of predictions for a given (true class, predicted class) pair. Off-diagonal entries are misclassifications. Comparing the random-head matrix to the fine-tuned matrix illustrates how training the head concentrates predictions along the diagonal.
    """)
    return


@app.cell
def _(sns):
    sns.set_style('white')
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
    ## Model Persistence: Saving and Loading

    After training, we need to persist the model so it can be reloaded for inference without retraining. PyTorch offers two approaches: saving the entire model object with `torch.save(model)`, or saving only the `state_dict` (an `OrderedDict` mapping layer names to their weight tensors). The `state_dict` approach is preferred because it produces smaller files, avoids pickling Python class definitions, and remains portable across code changes.

    The save-and-load workflow has three steps:

    1. **Save** the `state_dict` to a `.pth` file with `torch.save(model.state_dict(), path)`
    2. **Rebuild** the architecture in code (same layers, same dimensions)
    3. **Load** the saved weights into the fresh architecture with `model.load_state_dict(...)`

    The architecture must match exactly at load time. If the saved model had a `Linear(2048, 2)` head but the loading code creates `Linear(2048, 10)`, PyTorch will raise a shape mismatch error.
    """)
    return


@app.cell
def _(Path, resnet_model, torch):
    # --- Step 1: Save the fine-tuned model to disk ---
    #
    # model.state_dict() returns an OrderedDict mapping each layer name to its
    # weight/bias tensor. This captures everything the model learned during training.
    # It does NOT save the Python class or architecture code, just the numbers.
    #
    # Why state_dict over torch.save(model)?
    #   - Smaller file (no pickled class definitions)
    #   - Portable across code changes (you control the architecture at load time)
    #   - Recommended by PyTorch documentation
    Path('models').mkdir(exist_ok=True)
    model_save_path = Path("models/resnet50_selfie.pth")

    torch.save(
        obj=resnet_model.state_dict(),
        f=model_save_path,
    )

    _size_mb = model_save_path.stat().st_size / (1024 * 1024)
    print(f"Model saved to: {model_save_path}")
    print(f"File size: {_size_mb:.1f} MB")

    # You can inspect the keys to see what was saved
    _state = torch.load(model_save_path, weights_only=True)
    print(f"\nSaved {len(_state)} parameter tensors. First 5 keys:")
    for _i, _key in enumerate(list(_state.keys())[:5]):
        print(f"  {_key}: shape {list(_state[_key].shape)}")
    print(f"  ...")
    print(f"\nLast 2 keys (the fine-tuned classification head):")
    for _key in list(_state.keys())[-2:]:
        print(f"  {_key}: shape {list(_state[_key].shape)}")
    return (model_save_path,)


@app.cell
def _(device, model_save_path, models, nn, torch):
    # --- Step 2: Load the fine-tuned model from disk ---
    #
    # Since we saved only the state_dict (weight tensors), we must rebuild the
    # architecture first, then pour the saved weights back in. Three substeps:

    # 2a. Create a blank ResNet50 skeleton (weights=None means random initialization)
    loaded_model = models.resnet50(weights=None)

    # 2b. Replace the head with the same 2-class layer we used during training.
    #     The architecture must match exactly or load_state_dict will raise an error.
    loaded_model.fc = nn.Linear(
        in_features=loaded_model.fc.in_features,
        out_features=2,
    )

    # 2c. Load the saved weights into the matching architecture.
    #     map_location ensures the tensors land on the correct device (CPU or GPU).
    #     weights_only=True is a safety flag that prevents arbitrary code execution
    #     from untrusted .pth files.
    loaded_model.load_state_dict(
        torch.load(
            f=model_save_path,
            map_location=device,
            weights_only=True,
        )
    )

    loaded_model = loaded_model.to(device)

    # Set to evaluation mode: disables dropout and uses running stats for BatchNorm.
    # Forgetting this is a common bug that silently degrades inference accuracy.
    loaded_model.eval()

    print(f"Model loaded from: {model_save_path}")
    print(f"Device: {device}")
    print(f"Mode: eval (ready for inference)")
    return (loaded_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Inference with the Loaded Model

    With the model loaded and set to evaluation mode (`model.eval()`), we can classify new images. Two patterns are shown below: single-image inference and batch inference.

    For a single image, the key step is adding a batch dimension with `unsqueeze(0)`. PyTorch models always expect a batch dimension as the first axis, so a tensor of shape $(3, 224, 224)$ must become $(1, 3, 224, 224)$ before the forward pass. The model outputs raw logits, which are converted to class probabilities with the softmax function:

    $$
    P(y = k \mid \mathbf{x}) = \frac{\exp(z_k)}{\sum_{j=1}^{K} \exp(z_j)}
    $$

    For batch inference, the `DataLoader` already provides tensors with the batch dimension, so no reshaping is needed. Processing a full batch in one forward pass is more efficient than looping over individual images because it takes advantage of parallelism in matrix operations on both CPUs and GPUs.
    """)
    return


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
    # --- Step 3a: Inference on a single image ---
    #
    # This is the minimal workflow for classifying one image with a loaded model:
    #   1. Get the preprocessed image tensor from the dataset
    #   2. Add a batch dimension (models expect batches, not single images)
    #   3. Forward pass to get raw logits
    #   4. Apply softmax to convert logits to probabilities
    #   5. Read off the predicted class

    # Pick one test image
    _idx = test_indices[0]
    _img_tensor, _true_label = full_dataset[_idx]

    print(f"Image tensor shape: {list(_img_tensor.shape)}")
    print(f"  (channels, height, width) -- already preprocessed by ImageNet transform\n")

    # unsqueeze(0) adds the batch dimension: (3, 224, 224) -> (1, 3, 224, 224)
    _input_batch = _img_tensor.unsqueeze(0).to(device)
    print(f"Input batch shape:  {list(_input_batch.shape)}")
    print(f"  (batch_size, channels, height, width)\n")

    _autocast = torch.amp.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=use_bfloat16,
    )

    # Forward pass through the loaded model
    with torch.no_grad(), _autocast:
        _logits = loaded_model(_input_batch)
        _probs = torch.softmax(_logits, dim=1)

    print(f"Raw logits:    {_logits.cpu().float().numpy().round(3)}")
    print(f"Probabilities: {_probs.cpu().float().numpy().round(3)}")
    print(f"  Index 0 = {class_names[0]}, Index 1 = {class_names[1]}\n")

    _pred_label = _logits.argmax(dim=1).item()
    _confidence = _probs[0, _pred_label].item()

    print(f"Predicted class: {class_names[_pred_label]} ({_confidence:.1%} confidence)")
    print(f"True class:      {class_names[_true_label]}")

    # Show the image
    _mean = np.array([0.485, 0.456, 0.406])
    _std = np.array([0.229, 0.224, 0.225])
    _img_np = _img_tensor.numpy().transpose(1, 2, 0)
    _img_np = np.clip(_img_np * _std + _mean, 0, 1)

    _fig, _ax = plt.subplots(figsize=(3, 3))
    _ax.imshow(_img_np)
    _color = "green" if _pred_label == _true_label else "red"
    _ax.set_title(
        f"True: {class_names[_true_label]}\nPred: {class_names[_pred_label]} ({_confidence:.1%})",
        color=_color,
        fontsize=10,
    )
    _ax.axis("off")
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(
    class_names,
    device,
    loaded_model,
    np,
    plt,
    test_loader,
    torch,
    use_bfloat16,
):
    # --- Step 3b: Inference on a batch of images ---
    #
    # In practice you often classify many images at once. The DataLoader already
    # groups images into batches, so we grab the first batch and run it through
    # the model in a single forward pass.

    _autocast = torch.amp.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=use_bfloat16,
    )

    # Grab the first batch from the test loader
    _images, _labels = next(iter(test_loader))
    print(f"Batch shape: {list(_images.shape)}")
    print(f"  {_images.shape[0]} images, each {_images.shape[1]}x{_images.shape[2]}x{_images.shape[3]}\n")

    _images_device = _images.to(device)

    # One forward pass classifies the entire batch at once
    with torch.no_grad(), _autocast:
        _logits = loaded_model(_images_device)
        _probs = torch.softmax(_logits, dim=1)
        _preds = _logits.argmax(dim=1).cpu().numpy()
        _confs = _probs.max(dim=1).values.cpu().float().numpy()

    _labels_np = _labels.numpy()
    _correct = (_preds == _labels_np).sum()
    print(f"Batch accuracy: {_correct}/{len(_labels_np)} ({_correct / len(_labels_np):.1%})\n")

    # Visualize the first 8 images from this batch
    _mean = np.array([0.485, 0.456, 0.406])
    _std = np.array([0.229, 0.224, 0.225])

    _fig, _axes = plt.subplots(
        nrows=2,
        ncols=4,
        figsize=(12, 6),
    )

    for _i, _ax in enumerate(_axes.flat):
        _img_np = _images[_i].numpy().transpose(1, 2, 0)
        _img_np = np.clip(_img_np * _std + _mean, 0, 1)
        _ax.imshow(_img_np)

        _true_name = class_names[_labels_np[_i]]
        _pred_name = class_names[_preds[_i]]
        _color = "green" if _preds[_i] == _labels_np[_i] else "red"
        _ax.set_title(
            f"True: {_true_name}\nPred: {_pred_name} ({_confs[_i]:.1%})",
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
    ## Classifying a New Unlabeled Image

    The inference examples above pulled images from the `ImageFolder` dataset, where preprocessing was already applied and a ground truth label was available. In a real deployment scenario, you receive a raw image file with no label and must handle the full pipeline yourself.

    Three steps replace what `ImageFolder` did automatically:

    1. **Open the file** with `PIL.Image.open()` to get a PIL image object
    2. **Convert to RGB** with `.convert("RGB")` -- some images may be grayscale or RGBA, but the model expects three channels
    3. **Apply the same preprocessing transform** (`imagenet_transform`) that was used during training -- resize, center-crop, convert to tensor, and normalize with the ImageNet channel statistics

    After preprocessing, the workflow is identical to the single-image inference above: add a batch dimension, run the forward pass, and read off the predicted class. The example below uses an existing file from the dataset for convenience, but in practice this would be any new image.
    """)
    return


@app.cell
def _(Image, Path, class_names, device, imagenet_transform, loaded_model, np, plt, torch, use_bfloat16):
    # --- Classify a new image from a file path ---
    #
    # This is the real-world inference workflow: you have a file on disk,
    # no label, and need a prediction.

    # Step 1: Open the raw image file and convert to RGB
    _image_path = Path("data/selfie_data/Selfie/Selfie42906.jpg")
    _pil_image = Image.open(_image_path).convert("RGB")

    print(f"Image path:  {_image_path}")
    print(f"Image size:  {_pil_image.size} (width, height)")
    print(f"Image mode:  {_pil_image.mode}\n")

    # Step 2: Apply the same ImageNet preprocessing used during training
    _img_tensor = imagenet_transform(_pil_image)
    print(f"Tensor shape after transform: {list(_img_tensor.shape)}")
    print(f"  (channels, height, width)\n")

    # Step 3: Add batch dimension and run inference
    _input_batch = _img_tensor.unsqueeze(0).to(device)

    _autocast = torch.amp.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=use_bfloat16,
    )

    with torch.no_grad(), _autocast:
        _logits = loaded_model(_input_batch)
        _probs = torch.softmax(_logits, dim=1)

    _pred_label = _logits.argmax(dim=1).item()
    _confidence = _probs[0, _pred_label].item()

    print(f"Predicted class: {class_names[_pred_label]} ({_confidence:.1%} confidence)")
    print(f"Probabilities:   {dict(zip(class_names, _probs.cpu().float().numpy().round(3).flat))}")

    # Display the original image alongside the preprocessed version
    _mean = np.array([0.485, 0.456, 0.406])
    _std = np.array([0.229, 0.224, 0.225])
    _img_np = _img_tensor.numpy().transpose(1, 2, 0)
    _img_np = np.clip(_img_np * _std + _mean, 0, 1)

    _fig, (_ax1, _ax2) = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(6, 3),
    )

    _ax1.imshow(_pil_image)
    _ax1.set_title("Original image", fontsize=10)
    _ax1.axis("off")

    _ax2.imshow(_img_np)
    _ax2.set_title(
        f"Pred: {class_names[_pred_label]} ({_confidence:.1%})",
        fontsize=10,
    )
    _ax2.axis("off")

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
