import marimo

__generated_with = "0.20.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import pathlib

    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    import seaborn as sns
    import torch
    from sklearn.metrics import (
        ConfusionMatrixDisplay,
        accuracy_score,
        f1_score,
    )
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader, Dataset
    from tqdm import tqdm
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

    sns.set_style("whitegrid")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bfloat16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    print(f"Using device: {device}")
    if use_bfloat16:
        print("bfloat16 supported — enabling mixed-precision training")
    return (
        AutoModel,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        ConfusionMatrixDisplay,
        DataLoader,
        Dataset,
        accuracy_score,
        device,
        f1_score,
        mo,
        np,
        pathlib,
        pl,
        plt,
        sns,
        torch,
        tqdm,
        train_test_split,
        use_bfloat16,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Fine-Tuning a Transformer for Text Classification

    This notebook fine-tunes DistilBERT to predict Amazon product review ratings on a
    five-point scale. It covers three main ideas: (i) how transformers produce
    context-sensitive token representations, contrasted with static word embeddings;
    (ii) how a pre-trained backbone can be adapted for classification by attaching a
    trainable head; and (iii) how a two-phase training strategy — training the head only
    first, then fully fine-tuning all layers — affects convergence and final accuracy.

    The dataset contains 10,000 Amazon product reviews, each labeled with a star rating
    from 1 to 5.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Configuration

    These constants control the trade-off between training cost and coverage. `SAMPLE_SIZE`
    limits the dataset to a manageable subset; set it to `None` to train on all 10,000
    reviews. `MAX_LENGTH` is the token budget per review: sequences are truncated to this
    length and shorter ones are padded. `BATCH_SIZE` sets how many reviews are processed
    per gradient update, and `LEARNING_RATE` controls the step size. The two epoch counts
    govern the two training phases described below.
    """)
    return


@app.cell
def _():
    # Configuration — adjust these for faster demos or full training
    SAMPLE_SIZE = None  # Set to None to use all 10,000 samples
    MAX_LENGTH = 128  # Max tokens per review
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5

    EPOCHS_HEAD_ONLY = 5
    EPOCHS_FULL_FINETUNE = 5

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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Loading the Data

    We read a parquet file containing 10,000 Amazon product reviews. Each review has a
    `rating` field from 1 to 5, which we shift down by one to produce integer class labels
    in $[0, 4]$. This lets PyTorch treat the task as 5-way classification with 0-indexed
    targets.

    If `SAMPLE_SIZE` is set, we draw a random subset with a fixed seed so results are
    reproducible across runs.
    """)
    return


@app.cell
def _(SAMPLE_SIZE, np, pathlib, pl):
    data_filepath = pathlib.Path("data/amazon_reviews/amazon_reviews-10000.parquet")

    # Load Amazon reviews and convert ratings to integer labels (0-4)
    _df = pl.read_parquet(data_filepath)
    _df = _df.with_columns(
        label = (pl.col("rating").cast(pl.Int32) - 1)
    )

    if SAMPLE_SIZE is not None:
        np.random.seed(42)
        _indices = np.random.choice(
            a=len(_df),
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Splitting the Data

    We use a stratified 80/10/10 train/validation/test split. Stratification ensures each
    split preserves the original class distribution, which matters here because star
    ratings are not uniformly distributed in the wild. The validation set guides
    hyperparameter choices during training; the test set is held out entirely until the
    final evaluation.
    """)
    return


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Tokenizer

    DistilBERT uses a WordPiece tokenizer that breaks each word into subword pieces.
    Common words map to a single token, while rare or compound words are split into
    recognizable fragments — for example, "tokenization" might become `["token",
    "##ization"]`, where `##` marks a continuation piece. This scheme keeps the
    vocabulary finite while still handling any word a reviewer might use.

    The cell below loads the tokenizer, encodes a sample review, and prints the raw token
    IDs alongside their decoded subword strings.
    """)
    return


@app.cell
def _(AutoTokenizer, MODEL_NAME, texts):
    # Load tokenizer and show example
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=MODEL_NAME,
    )

    _sample_text = texts[0][:200]
    _tokens = tokenizer(
        text=_sample_text,
        truncation=True,
        padding=False,
        return_tensors=None,
    )

    print(f"Sample text: {_sample_text}")
    print(f"\nTokenized IDs: {_tokens['input_ids'][:20]}...")
    print(f"Decoded tokens: {tokenizer.convert_ids_to_tokens(_tokens['input_ids'][:20])}")
    return (tokenizer,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Static Word Embeddings

    Before examining how transformers improve on earlier approaches, it helps to
    understand what they replace. At the bottom of DistilBERT sits a plain embedding
    layer: a lookup table with one fixed 768-dimensional vector per vocabulary token. No
    matter the context, every occurrence of a word gets the same vector. A word like
    "bank" has the same embedding whether it appears in "river bank" or "bank account".

    The cell below retrieves the static embedding for the word "it" directly from the
    weight matrix to illustrate the lookup table idea.
    """)
    return


@app.cell
def _(AutoModel, MODEL_NAME, device, tokenizer):
    # Demonstrate: checking vocabulary and retrieving static embeddings
    _word = "it"

    # Check if the word is in the tokenizer's vocabulary
    _in_vocab = _word in tokenizer.vocab
    print(f"Is '{_word}' in vocabulary? {_in_vocab}")

    # Get the token ID
    _token_id = tokenizer.convert_tokens_to_ids(_word)
    print(f"Token ID for '{_word}': {_token_id}")

    # Load the base DistilBERT model (without classification head) to access embeddings
    _base_model = AutoModel.from_pretrained(
        pretrained_model_name_or_path=MODEL_NAME,
    ).to(device)

    # Extract the static embedding from the embedding layer
    # Note: AutoModel returns DistilBertModel directly, so we access embeddings directly
    _embedding_layer = _base_model.embeddings.word_embeddings
    _static_embedding = _embedding_layer.weight[_token_id].detach().cpu()

    print(f"\nStatic embedding shape: {_static_embedding.shape}")
    print(f"First 10 values: {_static_embedding[:10].numpy()}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Contextual Embeddings

    The self-attention mechanism in each transformer layer lets every token gather
    information from every other token in the sequence. As a result, the hidden state for
    a token changes depending on the surrounding words — this is what makes transformer
    representations *contextual*.

    The classic example is the pronoun "it": in "the cat didn't cross the road because it
    was tired", "it" refers to the cat, while in "the cat didn't cross the road because it
    was wide", "it" refers to the road. If the representations are truly context-sensitive,
    the embedding of "it" in the first sentence should be closer to "cat" than to "road",
    and vice versa in the second. We verify this by comparing cosine similarities between
    the final-layer hidden states of the key tokens.
    """)
    return


@app.cell
def _(AutoModel, MODEL_NAME, device, plt, tokenizer, torch):
    # Demonstrate: contextual embeddings differ based on surrounding words
    # Classic disambiguation example: "it" refers to different things in each sentence

    _sentence1 = "The cat didn't cross the road because it was tired"
    _sentence2 = "The cat didn't cross the road because it was wide"

    print("Sentence 1:", _sentence1)
    print("  → 'it' refers to 'cat' (the cat was tired)")
    print("\nSentence 2:", _sentence2)
    print("  → 'it' refers to 'road' (the road was wide)")

    # Tokenize both sentences
    _tokens1 = tokenizer(
        text=_sentence1,
        return_tensors="pt",
        padding=True,
    )
    _tokens2 = tokenizer(
        text=_sentence2,
        return_tensors="pt",
        padding=True,
    )

    # Find the position of "it" in each sentence
    _token_list1 = tokenizer.convert_ids_to_tokens(_tokens1["input_ids"][0])
    _token_list2 = tokenizer.convert_ids_to_tokens(_tokens2["input_ids"][0])

    _it_pos1 = _token_list1.index("it")
    _it_pos2 = _token_list2.index("it")
    _cat_pos = _token_list1.index("cat")
    _road_pos = _token_list1.index("road")

    print(f"\nTokens: {_token_list1}")
    print(f"Position of 'it': {_it_pos1}, 'cat': {_cat_pos}, 'road': {_road_pos}")

    # Load base model and get contextual embeddings
    _base_model = AutoModel.from_pretrained(
        pretrained_model_name_or_path=MODEL_NAME,
    ).to(device)
    _base_model.eval()

    with torch.no_grad():
        _outputs1 = _base_model(
            input_ids=_tokens1["input_ids"].to(device),
            attention_mask=_tokens1["attention_mask"].to(device),
        )
        _outputs2 = _base_model(
            input_ids=_tokens2["input_ids"].to(device),
            attention_mask=_tokens2["attention_mask"].to(device),
        )

    # Extract final layer hidden states
    _hidden1 = _outputs1.last_hidden_state[0].cpu()
    _hidden2 = _outputs2.last_hidden_state[0].cpu()

    # Get contextual embeddings for key tokens
    _it_emb1 = _hidden1[_it_pos1]
    _it_emb2 = _hidden2[_it_pos2]
    _cat_emb = _hidden1[_cat_pos]
    _road_emb = _hidden1[_road_pos]

    # Compute cosine similarities
    def _cosine_sim(a, b):
        return torch.nn.functional.cosine_similarity(
            x1=a.unsqueeze(0),
            x2=b.unsqueeze(0),
        ).item()

    _sim_it1_cat = _cosine_sim(_it_emb1, _cat_emb)
    _sim_it1_road = _cosine_sim(_it_emb1, _road_emb)
    _sim_it2_cat = _cosine_sim(_it_emb2, _cat_emb)
    _sim_it2_road = _cosine_sim(_it_emb2, _road_emb)
    _sim_it1_it2 = _cosine_sim(_it_emb1, _it_emb2)

    print(f"\n--- Cosine Similarities ---")
    print(f"Sentence 1 'it' vs 'cat':  {_sim_it1_cat:.4f}")
    print(f"Sentence 1 'it' vs 'road': {_sim_it1_road:.4f}")
    print(f"Sentence 2 'it' vs 'cat':  {_sim_it2_cat:.4f}")
    print(f"Sentence 2 'it' vs 'road': {_sim_it2_road:.4f}")
    print(f"\nSentence 1 'it' vs Sentence 2 'it': {_sim_it1_it2:.4f}")

    # Visualize the similarities
    _fig, _ax = plt.subplots(figsize=(6, 4))

    _bar_labels = ["'it' (tired)\nvs 'cat'", "'it' (tired)\nvs 'road'",
                   "'it' (wide)\nvs 'cat'", "'it' (wide)\nvs 'road'"]
    _sims = [_sim_it1_cat, _sim_it1_road, _sim_it2_cat, _sim_it2_road]
    _colors = ["steelblue", "lightgray", "lightgray", "orange"]

    _bars = _ax.bar(
        x=_bar_labels,
        height=_sims,
        color=_colors,
        edgecolor="k",
    )
    _ax.set_ylabel("Cosine Similarity")
    _ax.set_title("Contextual Embeddings: Which Noun Does 'it' Refer To?")
    _ax.set_ylim(0, 1)

    for _bar, _sim in zip(_bars, _sims):
        _ax.text(
            x=_bar.get_x() + _bar.get_width() / 2,
            y=_bar.get_height() + 0.02,
            s=f"{_sim:.3f}",
            ha="center",
            fontsize=10,
        )

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Custom Dataset

    PyTorch's `DataLoader` expects a `Dataset` object that implements two methods:
    `__len__` (the total number of samples) and `__getitem__` (a single sample by integer
    index). `ReviewDataset` stores the raw text list and label array in memory and
    tokenizes each review on demand inside `__getitem__`. Tokenizing lazily rather than
    all up front avoids holding all padded tensors in memory simultaneously and keeps
    memory use proportional to batch size.
    """)
    return


@app.cell
def _(Dataset, torch):
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
                text=_text,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )

            return {
                "input_ids": _encoding["input_ids"].squeeze(0),
                "attention_mask": _encoding["attention_mask"].squeeze(0),
                "labels": torch.tensor(
                    data=_label,
                    dtype=torch.long,
                ),
            }

    return (ReviewDataset,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## DataLoaders

    A `DataLoader` wraps a `Dataset` and handles batching, shuffling, and parallel data
    loading. We shuffle the training set at each epoch using a fixed-seed `Generator` so
    the shuffle order is reproducible across runs. The validation and test loaders are
    left unshuffled because evaluation order does not affect metrics.
    """)
    return


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
    return test_loader, train_loader, val_loader


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Phase 1: Frozen Base Model

    We load DistilBERT with a randomly initialized 5-class classification head and then
    freeze all parameters in the transformer backbone. Only the head's weights will be
    updated during Phase 1. This two-step approach gives the head a chance to reach a
    reasonable state before we risk distorting the pretrained language representations in
    the backbone.

    The cell prints the split between trainable (head only, roughly 600K parameters) and
    frozen (backbone, roughly 66M parameters).
    """)
    return


@app.cell
def _(AutoModelForSequenceClassification, MODEL_NAME, device, torch):
    # Load model with frozen base — only classification head is trainable
    torch.manual_seed(42)

    model_frozen = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=MODEL_NAME,
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Training and Evaluation Functions

    `train_epoch` runs one full pass over the training loader. When `labels` are passed to
    the HuggingFace model, it computes cross-entropy loss internally and exposes it via
    `outputs.loss`, so no external criterion is needed. `evaluate` runs inference with
    gradients disabled and computes loss separately using an explicit `CrossEntropyLoss`
    criterion, because labels are not passed to the model during evaluation (we want raw
    logits for metric computation). Both functions enable bfloat16 mixed-precision via
    `torch.amp.autocast` when the hardware supports it.
    """)
    return


@app.cell
def _(device, np, torch, tqdm, use_bfloat16):
    # Training and evaluation functions
    def train_epoch(model, train_loader, optimizer):
        _autocast = torch.amp.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_bfloat16,
        )

        model.train()
        _total_loss = 0.0
        _correct = 0
        _total = 0

        for _batch in tqdm(
            iterable=train_loader,
            desc="Training",
            leave=False,
        ):
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
                _loss = _criterion(
                    input=_outputs.logits,
                    target=_labels,
                )
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Phase 1 Training

    With the base frozen, only the classification head updates. The model should converge
    quickly because it can use DistilBERT's pretrained [CLS] representation directly —
    but peak accuracy is bounded by how well that fixed representation captures the
    sentiment signal needed for star-rating prediction.
    """)
    return


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Phase 1 Training Curves

    The loss and accuracy curves below show how the classification head learns over the
    frozen-base epochs. A decreasing validation loss that tracks the training loss without
    diverging indicates the head is fitting without overfitting to the training set.
    """)
    return


@app.cell
def _(head_only_history, plt):
    # Plot Phase 1 training curves
    _fig, (_ax1, _ax2) = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(10, 4),
    )

    _epochs = range(1, len(head_only_history["train_loss"]) + 1)

    _ax1.plot(
        _epochs,
        head_only_history["train_loss"],
        "o-",
        label="Train",
        color='steelblue',
    )
    _ax1.plot(
        _epochs,
        head_only_history["val_loss"],
        "o-",
        label="Val",
        color='orange',
    )
    _ax1.set_xlabel("Epoch")
    _ax1.set_ylabel("Loss")
    _ax1.set_title("Phase 1: Head Only — Loss")
    _ax1.legend()

    _ax2.plot(
        _epochs,
        head_only_history["train_acc"],
        "o-",
        label="Train",
        color='blue',
    )
    _ax2.plot(
        _epochs,
        head_only_history["val_acc"],
        "o-",
        label="Val",
        color='orange',
    )
    _ax2.set_xlabel("Epoch")
    _ax2.set_ylabel("Accuracy")
    _ax2.set_title("Phase 1: Head Only — Accuracy")
    _ax2.legend()

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Phase 2: Full Fine-Tuning

    We load a fresh copy of DistilBERT with all parameters trainable. Starting from a
    clean initialization (rather than continuing from Phase 1) keeps the comparison fair
    and lets us attribute any accuracy difference to the training strategy rather than
    accumulated training steps. The same `LEARNING_RATE` applies; because the backbone is
    already pretrained for language understanding, this small learning rate is enough to
    shift representations toward the review-rating task without destroying the pretrained
    features.
    """)
    return


@app.cell
def _(AutoModelForSequenceClassification, MODEL_NAME, device, torch):
    # Load fresh model for full fine-tuning — all parameters trainable
    torch.manual_seed(42)

    model_full = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=MODEL_NAME,
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Phase 2 Training

    With all parameters free to update, the model can reshape both the token
    representations and the classifier simultaneously. Expect higher final accuracy than
    Phase 1, though training takes longer per epoch because every gradient flows back
    through the full transformer stack.
    """)
    return


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Phase 2 Training Curves

    Compare these curves to Phase 1. Full fine-tuning typically shows a slower start but
    a deeper eventual improvement because the entire network adapts to the task, not just
    the classification head.
    """)
    return


@app.cell
def _(full_finetune_history, plt):
    # Plot Phase 2 training curves
    _fig, (_ax1, _ax2) = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(10, 4),
    )

    _epochs = range(1, len(full_finetune_history["train_loss"]) + 1)

    _ax1.plot(
        _epochs,
        full_finetune_history["train_loss"],
        "o-",
        label="Train",
        color='steelblue',
    )
    _ax1.plot(
        _epochs,
        full_finetune_history["val_loss"],
        "o-",
        label="Val",
        color='orange',
    )
    _ax1.set_xlabel("Epoch")
    _ax1.set_ylabel("Loss")
    _ax1.set_title("Phase 2: Full Fine-tuning — Loss")
    _ax1.legend()

    _ax2.plot(
        _epochs,
        full_finetune_history["train_acc"],
        "o-",
        label="Train",
        color='blue',
    )
    _ax2.plot(
        _epochs,
        full_finetune_history["val_acc"],
        "o-",
        label="Val",
        color='orange',
    )
    _ax2.set_xlabel("Epoch")
    _ax2.set_ylabel("Accuracy")
    _ax2.set_title("Phase 2: Full Fine-tuning — Accuracy")
    _ax2.legend()

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Test Set Evaluation

    We run the fully fine-tuned model once on the held-out test set to get an unbiased
    estimate of generalization performance. Accuracy and macro F1 are reported; macro F1
    weights each class equally, which matters here because the class distribution is
    skewed toward higher star ratings. The confusion matrix shows which star ratings are
    most often confused with one another.
    """)
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

    _test_acc = accuracy_score(
        y_true=test_results["labels"],
        y_pred=test_results["predictions"],
    )
    _test_f1 = f1_score(
        y_true=test_results["labels"],
        y_pred=test_results["predictions"],
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Error Analysis

    The heatmap below shows only the misclassified examples — correct predictions along
    the diagonal are excluded. Adjacent star ratings tend to generate the most errors
    because the sentiment difference between, say, 3-star and 4-star is subtler than
    between 1-star and 5-star. A well-calibrated model should show errors concentrated
    near the off-diagonals, with very few extreme misclassifications such as predicting
    5-star for a 1-star review.
    """)
    return


@app.cell
def _(np, plt, sns, test_results):
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
        data=_error_matrix,
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
def _(MAX_LENGTH, device, model_full, tokenizer, torch):
    # Inference example: classify new text
    def classify_review(text):
        model_full.eval()

        _encoding = tokenizer(
            text=text,
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

        _probs = torch.softmax(
            input=_outputs.logits,
            dim=1,
        ).squeeze()
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
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
