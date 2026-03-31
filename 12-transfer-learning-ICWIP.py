import marimo

__generated_with = "0.20.2"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pacmap
    import polars as pl
    import seaborn as sns
    import torch
    import torch.nn as nn
    from sklearn.cluster import KMeans
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
        ImageFolder,
        KMeans,
        Path,
        Subset,
        accuracy_score,
        device,
        f1_score,
        models,
        nn,
        np,
        pacmap,
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


@app.cell
def _(ImageFolder, np, train_test_split, transforms):
    # ImageNet channel means and stds -- must match the preprocessing
    # the backbone saw during pre-training
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
        # Reverse the ImageNet normalization to recover viewable pixel values
        _img_np = _img.numpy().transpose(1, 2, 0)
        _img_np = np.clip(_img_np * _std + _mean, 0, 1)
        _ax.imshow(_img_np)
        _ax.set_title(class_names[_label])
        _ax.axis("off")

    plt.tight_layout()
    plt.show()
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


@app.cell
def _(resnet_model, test_loader, train_loader, train_model, val_loader):
    # Fine-tune only the fc head; backbone stays frozen so each epoch is fast even on CPU
    finetuned_results = train_model(
        model=resnet_model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_epochs=4,
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
    model_save_path = Path("data/selfie_data/resnet50_selfie.pth")

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

    print(f"Raw logits:    {_logits.cpu().numpy().round(3)}")
    print(f"Probabilities: {_probs.cpu().numpy().round(3)}")
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
        _confs = _probs.max(dim=1).values.cpu().numpy()

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


@app.cell
def _(device, nn, np, resnet_model, test_loader, torch, use_bfloat16):
    # Embeddings are the 2048-dim vectors from the penultimate layer --
    # the backbone's learned representation of each image
    _autocast = torch.amp.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=use_bfloat16,
    )

    # nn.Identity() passes the backbone output through unchanged,
    # bypassing the classification head
    _original_fc = resnet_model.fc
    resnet_model.fc = nn.Identity()

    resnet_model.eval()
    _all_embeddings = []
    _all_labels = []

    with torch.no_grad(), _autocast:
        for _X_batch, _y_batch in test_loader:
            _X_batch = _X_batch.to(device)
            _emb = resnet_model(_X_batch)
            _all_embeddings.append(_emb.cpu().numpy())
            _all_labels.append(_y_batch.numpy())

    # Restore the trained fc head so the model is usable for classification again
    resnet_model.fc = _original_fc

    embeddings = np.concatenate(_all_embeddings)
    test_labels = np.concatenate(_all_labels)

    print(f"Embedding shape: {embeddings.shape}")
    print(f"Each image is now represented as a {embeddings.shape[1]}-dimensional vector")
    return embeddings, test_labels


@app.cell
def _(class_names, embeddings, np, pacmap, plt, sns, test_labels):
    # Project 2048-dim embeddings to 2D for visualization --
    # shows how the fine-tuned backbone separates the two classes
    _pacmap_model = pacmap.PaCMAP(n_components=2)
    embeddings_2d = _pacmap_model.fit_transform(embeddings)

    _label_names = np.array([class_names[_i] for _i in test_labels])

    _fig, _ax = plt.subplots(figsize=(8, 5))

    sns.scatterplot(
        x=embeddings_2d[:, 0],
        y=embeddings_2d[:, 1],
        hue=_label_names,
        palette={"NonSelfie": "steelblue", "Selfie": "coral"},
        edgecolor="k",
        alpha=0.75,
        ax=_ax,
    )
    _ax.set_xlabel("PaCMAP 1")
    _ax.set_ylabel("PaCMAP 2")
    _ax.set_title("PaCMAP Projection of ResNet50 Embeddings (Test Set)")
    _ax.legend(
        title="Class",
        bbox_to_anchor=(1.01, 1.01),
        loc="upper left",
    )

    # Well-separated clusters here confirm the backbone learned
    # discriminative visual features
    plt.tight_layout()
    plt.show()
    return (embeddings_2d,)


@app.cell
def _(
    KMeans,
    accuracy_score,
    class_names,
    embeddings,
    embeddings_2d,
    finetuned_results,
    np,
    plt,
    sns,
    test_labels,
):
    # K-Means on the 2048-dim embeddings -- clustering in the original
    # feature space, not the 2D projection
    _kmeans = KMeans(
        n_clusters=2,
        random_state=42,
        n_init=10,
    )
    _cluster_labels = _kmeans.fit_predict(embeddings)

    # Cluster IDs are arbitrary, so we check both possible label alignments
    _accuracy_direct = accuracy_score(test_labels, _cluster_labels)
    _accuracy_flipped = accuracy_score(test_labels, 1 - _cluster_labels)
    _cluster_accuracy = max(_accuracy_direct, _accuracy_flipped)

    # Compare: unsupervised clustering accuracy vs supervised fine-tuned accuracy
    _supervised_accuracy = accuracy_score(
        finetuned_results["labels"],
        finetuned_results["predictions"],
    )
    print(f"K-Means clustering accuracy (unsupervised): {_cluster_accuracy:.4f}")
    print(f"Fine-tuned classifier accuracy (supervised): {_supervised_accuracy:.4f}")

    # Use the better alignment for visualization
    _aligned_labels = _cluster_labels if _accuracy_direct >= _accuracy_flipped else 1 - _cluster_labels
    _label_names = np.array([class_names[_i] for _i in test_labels])
    _cluster_names = np.array([class_names[_i] for _i in _aligned_labels])

    _fig, (_ax1, _ax2) = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(14, 5),
    )

    sns.scatterplot(
        x=embeddings_2d[:, 0],
        y=embeddings_2d[:, 1],
        hue=_label_names,
        palette={"NonSelfie": "steelblue", "Selfie": "coral"},
        edgecolor="k",
        alpha=0.75,
        ax=_ax1,
    )
    _ax1.set_xlabel("PaCMAP 1")
    _ax1.set_ylabel("PaCMAP 2")
    _ax1.set_title("True Labels")
    _ax1.legend(
        title="Class",
        bbox_to_anchor=(1.01, 1.01),
        loc="upper left",
    )

    sns.scatterplot(
        x=embeddings_2d[:, 0],
        y=embeddings_2d[:, 1],
        hue=_cluster_names,
        palette={"NonSelfie": "steelblue", "Selfie": "coral"},
        edgecolor="k",
        alpha=0.75,
        ax=_ax2,
    )
    _ax2.set_xlabel("PaCMAP 1")
    _ax2.set_ylabel("PaCMAP 2")
    _ax2.set_title("K-Means Cluster Assignments")
    _ax2.legend(
        title="Cluster",
        bbox_to_anchor=(1.01, 1.01),
        loc="upper left",
    )

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
