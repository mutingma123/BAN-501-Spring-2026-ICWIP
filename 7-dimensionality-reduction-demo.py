import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    import matplotlib.pyplot as plt
    import numpy as np
    import optuna
    import pacmap
    import polars as pl
    import seaborn as sns
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        ConfusionMatrixDisplay,
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    )
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.preprocessing import StandardScaler

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sns.set_style("whitegrid")
    return (
        ConfusionMatrixDisplay,
        PCA,
        RandomForestClassifier,
        StandardScaler,
        accuracy_score,
        cross_val_score,
        f1_score,
        mo,
        np,
        optuna,
        pacmap,
        pl,
        plt,
        precision_score,
        recall_score,
        sns,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Dimensionality Reduction and Classification on MNIST

    This notebook explores two dimensionality reduction techniques on the MNIST handwritten digit
    dataset:

    - **PCA** (Principal Component Analysis) — a linear method
    - **PaCMAP** (Pairwise Controlled Manifold Approximation Projection) — a non-linear method

    After visualizing the 2D embeddings, we train random forest classifiers on three
    representations of the data — raw (784-dim), PCA (2-dim), and PaCMAP (2-dim) — to see how
    dimensionality reduction affects classification performance.
    """)
    return


@app.cell
def _(pl, train_test_split):
    feature_data = pl.read_parquet("data/MNIST/mnist_features.parquet")
    target_data = pl.read_parquet("data/MNIST/mnist_target.parquet")

    _X_train, _X_test, _y_train, _y_test = train_test_split(
        feature_data,
        target_data,
        test_size=5_000,
        random_state=42,
    )

    feature_data = _X_test.to_numpy()
    target_data = _y_test.to_numpy()
    return feature_data, target_data


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## MNIST Dataset

    Each sample is a 28x28 grayscale image of a handwritten digit (0–9), flattened into a vector
    of 784 features. Below is an example image from the dataset.
    """)
    return


@app.cell
def _(feature_data, plt):
    _idx = 300

    image_array = feature_data[_idx].reshape(28, 28)

    _fig, _ax = plt.subplots(1, 1, figsize=(2, 2))

    _ax.imshow(image_array)

    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## PCA (Principal Component Analysis)

    PCA is a **linear** dimensionality reduction method. It finds the directions (principal
    components) along which the data varies the most and projects the data onto those axes.

    With only 2 components, we capture the two directions of greatest variance — but for a
    784-dimensional dataset like MNIST, this discards a large amount of information. The 2D
    scatter plot shows some clustering by digit class, though with significant overlap.
    """)
    return


@app.cell
def _(PCA, StandardScaler, feature_data, plt, sns, target_data):
    scaler = StandardScaler()
    scaler.fit(feature_data)
    scaled_features = scaler.transform(feature_data)

    PCA_model = PCA(n_components=2)
    PCA_model.fit(scaled_features)
    PCA_feature_data = PCA_model.transform(scaled_features)

    _fig, _ax = plt.subplots(1, 1, figsize=(4, 4))

    sns.scatterplot(
        x=PCA_feature_data[:, 0],
        y=PCA_feature_data[:, 1],
        edgecolor="k",
        hue=target_data.flatten(),
        palette="tab10",
    )
    _ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.01),
    )
    plt.show()
    return (PCA_feature_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## PaCMAP (Pairwise Controlled Manifold Approximation Projection)

    PaCMAP is a **non-linear** dimensionality reduction method designed to preserve both local
    and global structure in the data. Unlike PCA, which is restricted to linear projections,
    PaCMAP can capture complex, non-linear relationships between data points.

    The 2D embedding typically shows much cleaner separation between digit classes compared to
    PCA, because PaCMAP optimizes for preserving neighborhood relationships.
    """)
    return


@app.cell
def _(feature_data, pacmap, plt, sns, target_data):
    pacmap_model = pacmap.PaCMAP(n_components=2)
    pacmap_feature_data = pacmap_model.fit_transform(feature_data)

    _fig, _ax = plt.subplots(1, 1, figsize=(4, 4))

    sns.scatterplot(
        x=pacmap_feature_data[:, 0],
        y=pacmap_feature_data[:, 1],
        edgecolor="k",
        hue=target_data.flatten(),
        palette="tab10",
    )
    _ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.01),
    )
    plt.show()
    return (pacmap_feature_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Classification Comparison

    Do the 2D embeddings retain enough information for a classifier to distinguish between digit
    classes? We fit an Optuna-tuned random forest on three representations:

    1. **Raw features** (784 dimensions) — the full pixel data
    2. **PCA embedding** (2 dimensions) — linear projection
    3. **PaCMAP embedding** (2 dimensions) — non-linear projection

    We use the same train/test split across all three to ensure a fair comparison.
    """)
    return


@app.cell
def _(
    PCA_feature_data,
    feature_data,
    np,
    pacmap_feature_data,
    target_data,
    train_test_split,
):
    _indices = np.arange(len(feature_data))
    train_idx, test_idx = train_test_split(
        _indices,
        test_size=0.3,
        random_state=42,
        stratify=target_data.ravel(),
    )

    X_train_raw = feature_data[train_idx]
    X_test_raw = feature_data[test_idx]
    X_train_pca = PCA_feature_data[train_idx]
    X_test_pca = PCA_feature_data[test_idx]
    X_train_pacmap = pacmap_feature_data[train_idx]
    X_test_pacmap = pacmap_feature_data[test_idx]
    y_train = target_data[train_idx]
    y_test = target_data[test_idx]
    return (
        X_test_pacmap,
        X_test_pca,
        X_test_raw,
        X_train_pacmap,
        X_train_pca,
        X_train_raw,
        y_test,
        y_train,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Random Forest on Raw Features (784-dim)
    """)
    return


@app.cell
def _(
    RandomForestClassifier,
    X_test_raw,
    X_train_raw,
    cross_val_score,
    np,
    optuna,
    y_train,
):
    def _rf_raw_objective(_trial):
        _params = {
            "n_estimators": _trial.suggest_int("n_estimators", 50, 300),
            "max_depth": _trial.suggest_int("max_depth", 5, 30),
            "min_samples_split": _trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": _trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": _trial.suggest_categorical(
                "max_features", ["sqrt", "log2"],
            ),
            "random_state": 42,
            "n_jobs": -1,
        }
        _clf = RandomForestClassifier(**_params)
        _scores = cross_val_score(
            estimator=_clf,
            X=X_train_raw,
            y=y_train.ravel(),
            cv=5,
            scoring="accuracy",
        )
        return np.mean(_scores)

    _rf_raw_study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    _rf_raw_study.optimize(
        _rf_raw_objective,
        n_trials=50,
        show_progress_bar=True,
    )

    print(f"Best parameters: {_rf_raw_study.best_params}")
    print(f"Best CV accuracy: {_rf_raw_study.best_value:.4f}")

    _rf_raw_model = RandomForestClassifier(
        **_rf_raw_study.best_params,
        random_state=42,
        n_jobs=-1,
    )
    _rf_raw_model.fit(X_train_raw, y_train.ravel())

    rf_raw_predictions = _rf_raw_model.predict(X_test_raw)
    return (rf_raw_predictions,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Random Forest on PCA Embedding (2-dim)
    """)
    return


@app.cell
def _(
    RandomForestClassifier,
    X_test_pca,
    X_train_pca,
    cross_val_score,
    np,
    optuna,
    y_train,
):
    def _rf_pca_objective(_trial):
        _params = {
            "n_estimators": _trial.suggest_int("n_estimators", 50, 300),
            "max_depth": _trial.suggest_int("max_depth", 5, 30),
            "min_samples_split": _trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": _trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": _trial.suggest_categorical(
                "max_features", ["sqrt", "log2"],
            ),
            "random_state": 42,
            "n_jobs": -1,
        }
        _clf = RandomForestClassifier(**_params)
        _scores = cross_val_score(
            estimator=_clf,
            X=X_train_pca,
            y=y_train.ravel(),
            cv=5,
            scoring="accuracy",
        )
        return np.mean(_scores)

    _rf_pca_study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    _rf_pca_study.optimize(
        _rf_pca_objective,
        n_trials=50,
        show_progress_bar=True,
    )

    print(f"Best parameters: {_rf_pca_study.best_params}")
    print(f"Best CV accuracy: {_rf_pca_study.best_value:.4f}")

    _rf_pca_model = RandomForestClassifier(
        **_rf_pca_study.best_params,
        random_state=42,
        n_jobs=-1,
    )
    _rf_pca_model.fit(X_train_pca, y_train.ravel())

    rf_pca_predictions = _rf_pca_model.predict(X_test_pca)
    return (rf_pca_predictions,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Random Forest on PaCMAP Embedding (2-dim)
    """)
    return


@app.cell
def _(
    RandomForestClassifier,
    X_test_pacmap,
    X_train_pacmap,
    cross_val_score,
    np,
    optuna,
    y_train,
):
    def _rf_pacmap_objective(_trial):
        _params = {
            "n_estimators": _trial.suggest_int("n_estimators", 50, 300),
            "max_depth": _trial.suggest_int("max_depth", 5, 30),
            "min_samples_split": _trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": _trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": _trial.suggest_categorical(
                "max_features", ["sqrt", "log2"],
            ),
            "random_state": 42,
            "n_jobs": -1,
        }
        _clf = RandomForestClassifier(**_params)
        _scores = cross_val_score(
            estimator=_clf,
            X=X_train_pacmap,
            y=y_train.ravel(),
            cv=5,
            scoring="accuracy",
        )
        return np.mean(_scores)

    _rf_pacmap_study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    _rf_pacmap_study.optimize(
        _rf_pacmap_objective,
        n_trials=50,
        show_progress_bar=True,
    )

    print(f"Best parameters: {_rf_pacmap_study.best_params}")
    print(f"Best CV accuracy: {_rf_pacmap_study.best_value:.4f}")

    _rf_pacmap_model = RandomForestClassifier(
        **_rf_pacmap_study.best_params,
        random_state=42,
        n_jobs=-1,
    )
    _rf_pacmap_model.fit(X_train_pacmap, y_train.ravel())

    rf_pacmap_predictions = _rf_pacmap_model.predict(X_test_pacmap)
    return (rf_pacmap_predictions,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Results

    ### Metrics Summary

    We compare accuracy, precision, recall, and F1 score (all weighted for multiclass) across
    the three representations.
    """)
    return


@app.cell
def _(
    accuracy_score,
    f1_score,
    pl,
    precision_score,
    recall_score,
    rf_pacmap_predictions,
    rf_pca_predictions,
    rf_raw_predictions,
    y_test,
):
    _models = [
        ("Raw (784-dim)", rf_raw_predictions),
        ("PCA (2-dim)", rf_pca_predictions),
        ("PaCMAP (2-dim)", rf_pacmap_predictions),
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

    summary_df = pl.DataFrame({
        "Model": _names,
        "Accuracy": _accuracies,
        "Precision": _precisions,
        "Recall": _recalls,
        "F1": _f1s,
    })

    summary_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Confusion Matrices
    """)
    return


@app.cell
def _(
    ConfusionMatrixDisplay,
    plt,
    rf_pacmap_predictions,
    rf_pca_predictions,
    rf_raw_predictions,
    y_test,
):
    _models = [
        ("Raw (784-dim)", rf_raw_predictions),
        ("PCA (2-dim)", rf_pca_predictions),
        ("PaCMAP (2-dim)", rf_pacmap_predictions),
    ]

    _fig, _axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(12, 4.5),
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


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
