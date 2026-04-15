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
    from sklearn.metrics import (
        ConfusionMatrixDisplay,
        accuracy_score,
        f1_score,
    )
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader, Dataset
    from tqdm import tqdm
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    sns.set_style("whitegrid")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bfloat16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    print(f"Using device: {device}")
    if use_bfloat16:
        print("bfloat16 supported — enabling mixed-precision training")
    return (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        ConfusionMatrixDisplay,
        DataLoader,
        Dataset,
        Path,
        accuracy_score,
        device,
        f1_score,
        mo,
        nn,
        np,
        pl,
        plt,
        sns,
        torch,
        tqdm,
        train_test_split,
        use_bfloat16,
    )


@app.cell
def _():
    # Configuration — adjust these for faster demos or full training
    SAMPLE_SIZE = None  # Set to None to use all 10,000 samples
    MAX_LENGTH = 128  # Max tokens per review
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    EPOCHS_HEAD_ONLY = 3
    EPOCHS_FULL_FINETUNE = 3
    MODEL_NAME = "distilbert-base-uncased"
    return (
        BATCH_SIZE,
        EPOCHS_FULL_FINETUNE,
        EPOCHS_HEAD_ONLY,
        LEARNING_RATE,
        MAX_LENGTH,
        MODEL_NAME,
        SAMPLE_SIZE,
    )


@app.cell
def _(SAMPLE_SIZE, np, pl):
    # Load Amazon reviews and convert ratings to integer labels (0-4)
    _df = pl.read_parquet("data/amazon_reviews/amazon_reviews-10000.parquet")
    _df = _df.with_columns(
        (pl.col("rating").cast(pl.Int32) - 1).alias("label")
    )

    if SAMPLE_SIZE is not None:
        _rng = np.random.default_rng(seed=42)
        _indices = _rng.choice(
            len(_df),
            size=min(SAMPLE_SIZE, len(_df)),
            replace=False,
        )
        _df = _df[_indices.tolist()]

    texts = _df["text"].to_list()
    labels = _df["label"].to_numpy()

    print(f"Dataset size: {len(texts):,} reviews")
    print(f"\nClass distribution:")
    for _rating in range(5):
        _count = (labels == _rating).sum()
        print(f"  {_rating + 1}-star: {_count:,} ({_count / len(labels) * 100:.1f}%)")
    return labels, texts


@app.cell
def _(labels, np, texts, train_test_split):
    # Stratified 80/10/10 split
    _indices = np.arange(len(texts))

    train_indices, _temp_indices, _y_train, _y_temp = train_test_split(
        _indices,
        labels,
        test_size=0.2,
        stratify=labels,
        random_state=42,
    )
    val_indices, test_indices, _y_val, _y_test = train_test_split(
        _temp_indices,
        _y_temp,
        test_size=0.5,
        stratify=_y_temp,
        random_state=42,
    )

    print(f"Training:   {len(train_indices):,} samples")
    print(f"Validation: {len(val_indices):,} samples")
    print(f"Test:       {len(test_indices):,} samples")
    return test_indices, train_indices, val_indices


@app.cell
def _(AutoTokenizer, MODEL_NAME, texts):
    # Load tokenizer and show example
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    _sample_text = texts[0][:200]
    _tokens = tokenizer(
        _sample_text,
        truncation=True,
        padding=False,
        return_tensors=None,
    )

    print(f"Sample text: {_sample_text}")
    print(f"\nTokenized IDs: {_tokens['input_ids'][:20]}...")
    print(f"Decoded tokens: {tokenizer.convert_ids_to_tokens(_tokens['input_ids'][:20])}")
    return (tokenizer,)


@app.cell
def _(Dataset, labels, texts, tokenizer, torch):
    # Custom Dataset that tokenizes text and returns tensors
    class ReviewDataset(Dataset):
        def __init__(self, indices, texts, labels, tokenizer, max_length):
            self.indices = indices
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            _text_idx = self.indices[idx]
            _text = self.texts[_text_idx]
            _label = self.labels[_text_idx]

            _encoding = self.tokenizer(
                _text,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )

            return {
                "input_ids": _encoding["input_ids"].squeeze(0),
                "attention_mask": _encoding["attention_mask"].squeeze(0),
                "labels": torch.tensor(_label, dtype=torch.long),
            }

    return (ReviewDataset,)


@app.cell
def _(
    BATCH_SIZE,
    DataLoader,
    MAX_LENGTH,
    ReviewDataset,
    labels,
    test_indices,
    texts,
    tokenizer,
    torch,
    train_indices,
    val_indices,
):
    # Create DataLoaders
    train_dataset = ReviewDataset(
        indices=train_indices,
        texts=texts,
        labels=labels,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
    )
    val_dataset = ReviewDataset(
        indices=val_indices,
        texts=texts,
        labels=labels,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
    )
    test_dataset = ReviewDataset(
        indices=test_indices,
        texts=texts,
        labels=labels,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(42),
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
    print(f"Test batches:  {len(test_loader)}")
    return (
        test_dataset,
        test_loader,
        train_dataset,
        train_loader,
        val_dataset,
        val_loader,
    )


@app.cell
def _(AutoModelForSequenceClassification, MODEL_NAME, device, torch):
    # Load model with frozen base — only classification head is trainable
    torch.manual_seed(42)

    model_frozen = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=5,
    )

    # Freeze all base transformer layers
    for _param in model_frozen.distilbert.parameters():
        _param.requires_grad = False

    model_frozen = model_frozen.to(device)

    _total_params = sum(_p.numel() for _p in model_frozen.parameters())
    _trainable_params = sum(
        _p.numel() for _p in model_frozen.parameters() if _p.requires_grad
    )
    print(f"Total parameters:     {_total_params:,}")
    print(f"Trainable parameters: {_trainable_params:,}")
    print(f"Frozen parameters:    {_total_params - _trainable_params:,}")
    return (model_frozen,)


@app.cell
def _(device, np, torch, tqdm, use_bfloat16):
    # Training and evaluation functions
    def train_epoch(model, train_loader, optimizer, criterion):
        _autocast = torch.amp.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_bfloat16,
        )

        model.train()
        _total_loss = 0.0
        _correct = 0
        _total = 0

        for _batch in tqdm(train_loader, desc="Training", leave=False):
            _input_ids = _batch["input_ids"].to(device)
            _attention_mask = _batch["attention_mask"].to(device)
            _labels = _batch["labels"].to(device)

            optimizer.zero_grad()

            with _autocast:
                _outputs = model(
                    input_ids=_input_ids,
                    attention_mask=_attention_mask,
                    labels=_labels,
                )
                _loss = _outputs.loss

            _loss.backward()
            optimizer.step()

            _total_loss += _loss.item() * _input_ids.size(0)
            _preds = _outputs.logits.argmax(dim=1)
            _correct += (_preds == _labels).sum().item()
            _total += _labels.size(0)

        return _total_loss / _total, _correct / _total

    def evaluate(model, data_loader):
        _autocast = torch.amp.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_bfloat16,
        )

        model.eval()
        _all_preds = []
        _all_labels = []
        _total_loss = 0.0
        _total = 0

        _criterion = torch.nn.CrossEntropyLoss()

        with torch.no_grad(), _autocast:
            for _batch in data_loader:
                _input_ids = _batch["input_ids"].to(device)
                _attention_mask = _batch["attention_mask"].to(device)
                _labels = _batch["labels"].to(device)

                _outputs = model(
                    input_ids=_input_ids,
                    attention_mask=_attention_mask,
                )
                _loss = _criterion(_outputs.logits, _labels)
                _total_loss += _loss.item() * _input_ids.size(0)
                _total += _labels.size(0)

                _preds = _outputs.logits.argmax(dim=1)
                _all_preds.extend(_preds.cpu().numpy())
                _all_labels.extend(_labels.cpu().numpy())

        return {
            "loss": _total_loss / _total,
            "predictions": np.array(_all_preds),
            "labels": np.array(_all_labels),
        }

    return evaluate, train_epoch


@app.cell
def _(
    EPOCHS_HEAD_ONLY,
    LEARNING_RATE,
    evaluate,
    model_frozen,
    torch,
    train_epoch,
    train_loader,
    val_loader,
):
    # Phase 1: Train with frozen base (head only)
    print("Phase 1: Training classification head only (base frozen)")
    print("=" * 60)

    _optimizer = torch.optim.AdamW(
        params=filter(lambda _p: _p.requires_grad, model_frozen.parameters()),
        lr=LEARNING_RATE,
    )
    _criterion = torch.nn.CrossEntropyLoss()

    head_only_history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    for _epoch in range(EPOCHS_HEAD_ONLY):
        _train_loss, _train_acc = train_epoch(
            model=model_frozen,
            train_loader=train_loader,
            optimizer=_optimizer,
            criterion=_criterion,
        )

        _val_results = evaluate(
            model=model_frozen,
            data_loader=val_loader,
        )
        _val_acc = (_val_results["predictions"] == _val_results["labels"]).mean()

        head_only_history["train_loss"].append(_train_loss)
        head_only_history["train_acc"].append(_train_acc)
        head_only_history["val_loss"].append(_val_results["loss"])
        head_only_history["val_acc"].append(_val_acc)

        print(
            f"Epoch {_epoch + 1}/{EPOCHS_HEAD_ONLY} — "
            f"Train Loss: {_train_loss:.4f}, Train Acc: {_train_acc:.4f}, "
            f"Val Loss: {_val_results['loss']:.4f}, Val Acc: {_val_acc:.4f}"
        )

    print(f"\nHead-only training complete. Final val accuracy: {head_only_history['val_acc'][-1]:.4f}")
    return (head_only_history,)


@app.cell
def _(head_only_history, plt):
    # Plot Phase 1 training curves
    _fig, (_ax1, _ax2) = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(10, 4),
    )

    _epochs = range(1, len(head_only_history["train_loss"]) + 1)

    _ax1.plot(_epochs, head_only_history["train_loss"], "o-", label="Train")
    _ax1.plot(_epochs, head_only_history["val_loss"], "o-", label="Val")
    _ax1.set_xlabel("Epoch")
    _ax1.set_ylabel("Loss")
    _ax1.set_title("Phase 1: Head Only — Loss")
    _ax1.legend()

    _ax2.plot(_epochs, head_only_history["train_acc"], "o-", label="Train")
    _ax2.plot(_epochs, head_only_history["val_acc"], "o-", label="Val")
    _ax2.set_xlabel("Epoch")
    _ax2.set_ylabel("Accuracy")
    _ax2.set_title("Phase 1: Head Only — Accuracy")
    _ax2.legend()

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(AutoModelForSequenceClassification, MODEL_NAME, device, torch):
    # Load fresh model for full fine-tuning — all parameters trainable
    torch.manual_seed(42)

    model_full = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=5,
    )
    model_full = model_full.to(device)

    _total_params = sum(_p.numel() for _p in model_full.parameters())
    _trainable_params = sum(
        _p.numel() for _p in model_full.parameters() if _p.requires_grad
    )
    print(f"Total parameters:     {_total_params:,}")
    print(f"Trainable parameters: {_trainable_params:,}")
    return (model_full,)


@app.cell
def _(
    EPOCHS_FULL_FINETUNE,
    LEARNING_RATE,
    evaluate,
    model_full,
    torch,
    train_epoch,
    train_loader,
    val_loader,
):
    # Phase 2: Full fine-tuning
    print("Phase 2: Full fine-tuning (all layers trainable)")
    print("=" * 60)

    _optimizer = torch.optim.AdamW(
        params=model_full.parameters(),
        lr=LEARNING_RATE,
    )
    _criterion = torch.nn.CrossEntropyLoss()

    full_finetune_history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    for _epoch in range(EPOCHS_FULL_FINETUNE):
        _train_loss, _train_acc = train_epoch(
            model=model_full,
            train_loader=train_loader,
            optimizer=_optimizer,
            criterion=_criterion,
        )

        _val_results = evaluate(
            model=model_full,
            data_loader=val_loader,
        )
        _val_acc = (_val_results["predictions"] == _val_results["labels"]).mean()

        full_finetune_history["train_loss"].append(_train_loss)
        full_finetune_history["train_acc"].append(_train_acc)
        full_finetune_history["val_loss"].append(_val_results["loss"])
        full_finetune_history["val_acc"].append(_val_acc)

        print(
            f"Epoch {_epoch + 1}/{EPOCHS_FULL_FINETUNE} — "
            f"Train Loss: {_train_loss:.4f}, Train Acc: {_train_acc:.4f}, "
            f"Val Loss: {_val_results['loss']:.4f}, Val Acc: {_val_acc:.4f}"
        )

    print(f"\nFull fine-tuning complete. Final val accuracy: {full_finetune_history['val_acc'][-1]:.4f}")
    return (full_finetune_history,)


@app.cell
def _(full_finetune_history, plt):
    # Plot Phase 2 training curves
    _fig, (_ax1, _ax2) = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(10, 4),
    )

    _epochs = range(1, len(full_finetune_history["train_loss"]) + 1)

    _ax1.plot(_epochs, full_finetune_history["train_loss"], "o-", label="Train")
    _ax1.plot(_epochs, full_finetune_history["val_loss"], "o-", label="Val")
    _ax1.set_xlabel("Epoch")
    _ax1.set_ylabel("Loss")
    _ax1.set_title("Phase 2: Full Fine-tuning — Loss")
    _ax1.legend()

    _ax2.plot(_epochs, full_finetune_history["train_acc"], "o-", label="Train")
    _ax2.plot(_epochs, full_finetune_history["val_acc"], "o-", label="Val")
    _ax2.set_xlabel("Epoch")
    _ax2.set_ylabel("Accuracy")
    _ax2.set_title("Phase 2: Full Fine-tuning — Accuracy")
    _ax2.legend()

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(full_finetune_history, head_only_history, plt):
    # Compare the two approaches
    _fig, _ax = plt.subplots(figsize=(8, 5))

    _ax.bar(
        x=["Head Only", "Full Fine-tuning"],
        height=[
            head_only_history["val_acc"][-1],
            full_finetune_history["val_acc"][-1],
        ],
        edgecolor="k",
    )
    _ax.set_ylabel("Validation Accuracy")
    _ax.set_title("Comparison: Head-Only vs Full Fine-tuning")
    _ax.set_ylim(0, 1)

    for _i, _v in enumerate([head_only_history["val_acc"][-1], full_finetune_history["val_acc"][-1]]):
        _ax.text(_i, _v + 0.02, f"{_v:.3f}", ha="center", fontsize=12)

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(
    ConfusionMatrixDisplay,
    accuracy_score,
    evaluate,
    f1_score,
    model_full,
    plt,
    test_loader,
):
    # Final evaluation on test set using the fully fine-tuned model
    test_results = evaluate(
        model=model_full,
        data_loader=test_loader,
    )

    _test_acc = accuracy_score(test_results["labels"], test_results["predictions"])
    _test_f1 = f1_score(
        test_results["labels"],
        test_results["predictions"],
        average="macro",
    )

    print(f"Test Set Results (Full Fine-tuning)")
    print(f"  Accuracy: {_test_acc:.4f}")
    print(f"  Macro F1: {_test_f1:.4f}")

    _fig, _ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true=test_results["labels"],
        y_pred=test_results["predictions"],
        display_labels=["1-star", "2-star", "3-star", "4-star", "5-star"],
        ax=_ax,
        cmap="Blues",
    )
    _ax.set_title("Test Set Confusion Matrix")
    plt.tight_layout()
    plt.show()
    return (test_results,)


@app.cell
def _(np, plt, sns, test_indices, test_results, texts):
    # Error analysis: examine misclassified examples
    _preds = test_results["predictions"]
    _labels = test_results["labels"]
    _errors = _preds != _labels

    print(f"Total test samples: {len(_labels)}")
    print(f"Correct: {(~_errors).sum()} ({(~_errors).mean() * 100:.1f}%)")
    print(f"Errors:  {_errors.sum()} ({_errors.mean() * 100:.1f}%)")

    # Analyze error patterns: which classes get confused with which?
    _error_matrix = np.zeros((5, 5), dtype=int)
    for _true, _pred in zip(_labels[_errors], _preds[_errors]):
        _error_matrix[_true, _pred] += 1

    print("\nError breakdown by true label:")
    for _true_class in range(5):
        _class_mask = _labels == _true_class
        _class_errors = _errors[_class_mask].sum()
        _class_total = _class_mask.sum()
        if _class_total > 0:
            print(f"  {_true_class + 1}-star: {_class_errors}/{_class_total} errors ({_class_errors / _class_total * 100:.1f}%)")

    # Plot error heatmap (excluding diagonal which would be correct predictions)
    _fig, _ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        _error_matrix,
        annot=True,
        fmt="d",
        cmap="Reds",
        xticklabels=["1-star", "2-star", "3-star", "4-star", "5-star"],
        yticklabels=["1-star", "2-star", "3-star", "4-star", "5-star"],
        ax=_ax,
        linewidths=0.1,
        linecolor="k",
    )
    _ax.set_xlabel("Predicted")
    _ax.set_ylabel("True")
    _ax.set_title("Error Distribution (Misclassifications Only)")
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(np, test_indices, test_results, texts):
    # Show example misclassified reviews
    _preds = test_results["predictions"]
    _labels = test_results["labels"]
    _errors = np.where(_preds != _labels)[0]

    print("Sample Misclassified Reviews")
    print("=" * 70)

    # Show up to 2 examples for each type of significant error
    _shown_types = set()
    _examples_shown = 0

    for _idx in _errors:
        _true = _labels[_idx]
        _pred = _preds[_idx]
        _error_type = (_true, _pred)

        # Focus on large rating gaps (off by 2+ stars)
        if abs(_true - _pred) >= 2 and _error_type not in _shown_types and _examples_shown < 8:
            _text_idx = test_indices[_idx]
            _review = texts[_text_idx]

            print(f"\nTrue: {_true + 1}-star | Predicted: {_pred + 1}-star")
            print(f"Review: {_review[:300]}{'...' if len(_review) > 300 else ''}")
            print("-" * 70)

            _shown_types.add(_error_type)
            _examples_shown += 1

    if _examples_shown == 0:
        print("No large errors (off by 2+ stars) found in test set.")
    return


@app.cell
def _(MAX_LENGTH, device, model_full, tokenizer, torch):
    # Inference example: classify new text
    def classify_review(text):
        model_full.eval()

        _encoding = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        _input_ids = _encoding["input_ids"].to(device)
        _attention_mask = _encoding["attention_mask"].to(device)

        with torch.no_grad():
            _outputs = model_full(
                input_ids=_input_ids,
                attention_mask=_attention_mask,
            )

        _probs = torch.softmax(_outputs.logits, dim=1).squeeze()
        _pred = _probs.argmax().item()

        return {
            "predicted_rating": _pred + 1,
            "confidence": _probs[_pred].item(),
            "all_probabilities": {
                f"{i+1}-star": _probs[i].item()
                for i in range(5)
            },
        }

    # Test with sample reviews
    _samples = [
        "This product is amazing! Best purchase I've ever made.",
        "Terrible quality. Broke after one day. Do not buy.",
        "It's okay. Nothing special but gets the job done.",
    ]

    for _text in _samples:
        _result = classify_review(_text)
        print(f"Review: \"{_text}\"")
        print(f"  Predicted: {_result['predicted_rating']}-star ({_result['confidence']:.2%} confidence)")
        print()
    return (classify_review,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
