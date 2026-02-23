import marimo

__generated_with = "0.19.11"
app = marimo.App(width="full")


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
        roc_auc_score,
        roc_curve,
    )
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sns.set_style("whitegrid")
    return (
        ConfusionMatrixDisplay,
        OneHotEncoder,
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
        roc_auc_score,
        roc_curve,
        sns,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Dimensionality Reduction and Classification

    This notebook explores two dimensionality reduction techniques applied to the bank marketing
    dataset:

    - **PCA** (Principal Component Analysis) — a linear method that finds directions of maximum variance
    - **PaCMAP** (Pairwise Controlled Manifold Approximation Projection) — a non-linear method that preserves both local and global structure

    After visualizing the 2D embeddings, we train Optuna-tuned random forest classifiers on three
    representations — raw encoded features (10-dim), PCA (2-dim), and PaCMAP (2-dim) — to see how
    dimensionality reduction affects classification performance.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Dataset: Bank Marketing

    We use a bank marketing dataset where the goal is to predict whether a client will subscribe
    to a term deposit (`y = 1`) or not (`y = 0`).

    **Source**: This dataset is based on direct marketing campaigns of a Portuguese bank.

    **Class imbalance**: The target variable is imbalanced (~88% no, ~12% yes), which is common
    in real-world classification problems.
    """)
    return


@app.cell
def _(pl):
    raw_data = pl.read_parquet("data/classification/playground-series-s5e8/train.parquet")
    print(f"Dataset shape: {raw_data.shape[0]:,} rows x {raw_data.shape[1]} columns")
    raw_data.head()
    return (raw_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Feature Selection

    For this demonstration, we select a subset of 7 features:

    **Numeric features** (4):
    - `age`: Client's age
    - `balance`: Average yearly balance in euros
    - `duration`: Last contact duration in seconds (highly predictive but only known after call)
    - `campaign`: Number of contacts during this campaign

    **Categorical features** (3):
    - `marital`: Marital status (married, single, divorced)
    - `education`: Education level (primary, secondary, tertiary, unknown)
    - `housing`: Has housing loan? (yes, no)

    We sample 10,000 rows for faster demonstration.
    """)
    return


@app.cell
def _(raw_data, train_test_split):
    numeric_features = ["age", "balance", "duration", "campaign"]
    categorical_features = ["marital", "education", "housing"]
    target_column = "y"

    model_data = raw_data.select(
        numeric_features + categorical_features + [target_column]
    ).sample(
        n=10_000,
        seed=42,
        shuffle=True,
    )

    X = model_data.drop(target_column)
    y = model_data[target_column].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print(f"Training set: {len(X_train):,} samples")
    print(f"Test set: {len(X_test):,} samples")
    print(f"Class distribution in test set: {y_test.mean():.2%} positive")
    return (
        X_test,
        X_train,
        categorical_features,
        numeric_features,
        y_test,
        y_train,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## One-Hot Encoding

    Scikit-learn models require numeric input. We convert categorical features into binary columns
    using one-hot encoding. The encoder is fit only on training data to prevent data leakage.
    """)
    return


@app.cell
def _(
    OneHotEncoder,
    X_test,
    X_train,
    categorical_features,
    numeric_features,
    pl,
):
    encoder = OneHotEncoder(
        sparse_output=False,
        handle_unknown="ignore",
        drop="first",
    )

    encoder.fit(X_train.select(categorical_features))

    encoder.set_output(transform="polars")
    _X_train_cat = encoder.transform(X_train.select(categorical_features))
    _X_test_cat = encoder.transform(X_test.select(categorical_features))

    X_train_encoded = pl.concat([
        X_train.select(numeric_features),
        _X_train_cat,
    ], how="horizontal").to_numpy()

    X_test_encoded = pl.concat([
        X_test.select(numeric_features),
        _X_test_cat,
    ], how="horizontal").to_numpy()

    encoded_cat_names = list(encoder.get_feature_names_out(categorical_features))
    all_feature_names = numeric_features + encoded_cat_names

    print(f"Encoded feature count: {len(all_feature_names)}")
    print(f"Features: {all_feature_names}")
    return X_test_encoded, X_train_encoded, all_feature_names


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Feature Scaling

    PCA requires standardized features because it maximizes variance — features on larger scales
    would dominate the principal components. We apply `StandardScaler` (zero mean, unit variance)
    before PCA.

    PaCMAP handles scaling internally (via its `apply_pca` parameter), so it operates on the
    unscaled encoded features directly.
    """)
    return


@app.cell
def _(StandardScaler, X_test_encoded, X_train_encoded):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_encoded)
    X_test_scaled = scaler.transform(X_test_encoded)

    print(f"Scaled training set shape: {X_train_scaled.shape}")
    return X_train_scaled, X_test_scaled


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Variance Explained by Principal Components

    A scree plot shows how much variance each principal component captures. The cumulative line
    helps determine how many components are needed to retain a target percentage of total variance.
    """)
    return


@app.cell
def _(PCA, X_train_scaled, np, plt):
    _pca_full = PCA()
    _pca_full.fit(X_train_scaled)

    _var_ratios = _pca_full.explained_variance_ratio_
    _cumulative = np.cumsum(_var_ratios)
    _n_components = len(_var_ratios)

    _fig, _ax = plt.subplots(figsize=(8, 5))

    _ax.bar(
        range(1, _n_components + 1),
        _var_ratios,
        color="steelblue",
        edgecolor="k",
        label="Individual",
    )
    _ax.plot(
        range(1, _n_components + 1),
        _cumulative,
        "o-",
        color="coral",
        linewidth=2,
        label="Cumulative",
    )

    _ax.set_xlabel("Principal Component")
    _ax.set_ylabel("Explained Variance Ratio")
    _ax.set_title("Scree Plot")
    _ax.set_xticks(range(1, _n_components + 1))
    _ax.legend()

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## PCA: 2-Component Projection

    We project the 10-dimensional encoded feature space down to 2 dimensions using PCA. This
    allows us to visualize the data and use it as input for classification.
    """)
    return


@app.cell
def _(PCA, X_test_scaled, X_train_scaled):
    pca_2d_model = PCA(n_components=2)
    X_train_pca = pca_2d_model.fit_transform(X_train_scaled)
    X_test_pca = pca_2d_model.transform(X_test_scaled)

    print(f"Variance explained by 2 components: {pca_2d_model.explained_variance_ratio_.sum():.2%}")
    print(f"PC1: {pca_2d_model.explained_variance_ratio_[0]:.2%}")
    print(f"PC2: {pca_2d_model.explained_variance_ratio_[1]:.2%}")
    return X_train_pca, X_test_pca, pca_2d_model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## PCA Loadings

    The loadings matrix shows how each original feature contributes to each principal component.
    Large positive or negative values indicate features that strongly influence that component.
    The `RdBu_r` colormap highlights the direction: red for positive, blue for negative.
    """)
    return


@app.cell
def _(all_feature_names, pca_2d_model, plt, sns):
    _fig, _ax = plt.subplots(figsize=(8, 3))

    sns.heatmap(
        pca_2d_model.components_,
        xticklabels=all_feature_names,
        yticklabels=["PC1", "PC2"],
        cmap="RdBu_r",
        center=0,
        annot=True,
        fmt=".2f",
        linewidths=0.1,
        linecolor="k",
        ax=_ax,
    )

    _ax.set_title("PCA Loadings (2 Components)")

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## PCA: 2D Scatter Plot

    Training samples projected onto the first two principal components, colored by target class.
    """)
    return


@app.cell
def _(X_train_pca, plt, sns, y_train):
    _fig, _ax = plt.subplots(figsize=(8, 5))

    sns.scatterplot(
        x=X_train_pca[:, 0],
        y=X_train_pca[:, 1],
        hue=y_train,
        palette={0: "steelblue", 1: "coral"},
        alpha=0.5,
        s=10,
        ax=_ax,
    )

    _ax.set_xlabel("PC1")
    _ax.set_ylabel("PC2")
    _ax.set_title("PCA 2D Projection (Training Set)")
    _ax.legend(title="Target")

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## PaCMAP (Pairwise Controlled Manifold Approximation Projection)

    PaCMAP is a **non-linear** dimensionality reduction method that preserves both local and
    global data structure. Unlike PCA, it can capture complex, non-linear relationships.

    PaCMAP handles feature scaling internally (via its `apply_pca` parameter), so we pass
    the unscaled encoded features directly.
    """)
    return


@app.cell
def _(X_test_encoded, X_train_encoded, pacmap):
    pacmap_model = pacmap.PaCMAP(
        n_components=2,
        random_state=42,
        save_tree=True,
    )
    X_train_pacmap = pacmap_model.fit_transform(X_train_encoded)
    X_test_pacmap = pacmap_model.transform(X_test_encoded)

    print(f"PaCMAP training shape: {X_train_pacmap.shape}")
    print(f"PaCMAP test shape: {X_test_pacmap.shape}")
    return X_train_pacmap, X_test_pacmap


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## PaCMAP: 2D Scatter Plot

    Training samples embedded into 2 dimensions using PaCMAP, colored by target class.
    """)
    return


@app.cell
def _(X_train_pacmap, plt, sns, y_train):
    _fig, _ax = plt.subplots(figsize=(8, 5))

    sns.scatterplot(
        x=X_train_pacmap[:, 0],
        y=X_train_pacmap[:, 1],
        hue=y_train,
        palette={0: "steelblue", 1: "coral"},
        alpha=0.5,
        s=10,
        ax=_ax,
    )

    _ax.set_xlabel("PaCMAP 1")
    _ax.set_ylabel("PaCMAP 2")
    _ax.set_title("PaCMAP 2D Projection (Training Set)")
    _ax.legend(title="Target")

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Classification Comparison

    Do the 2D embeddings retain enough information for a classifier to distinguish between
    classes? We fit an Optuna-tuned random forest on three representations:

    1. **Raw encoded features** (10 dimensions) — the full one-hot encoded feature set
    2. **PCA embedding** (2 dimensions) — linear projection
    3. **PaCMAP embedding** (2 dimensions) — non-linear projection

    Each model is tuned with 50 Optuna trials using ROC AUC as the scoring metric, appropriate
    for the imbalanced class distribution.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Random Forest on Raw Encoded Features (10-dim)
    """)
    return


@app.cell
def _(
    RandomForestClassifier,
    X_test_encoded,
    X_train_encoded,
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
            "n_jobs": 1,
        }
        _clf = RandomForestClassifier(**_params)
        _scores = cross_val_score(
            estimator=_clf,
            X=X_train_encoded,
            y=y_train,
            cv=5,
            scoring="roc_auc",
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
    print(f"Best CV AUC: {_rf_raw_study.best_value:.4f}")

    _rf_raw_model = RandomForestClassifier(
        **_rf_raw_study.best_params,
        random_state=42,
        n_jobs=-1,
    )
    _rf_raw_model.fit(X_train_encoded, y_train)

    rf_raw_proba = _rf_raw_model.predict_proba(X_test_encoded)[:, 1]
    rf_raw_pred = _rf_raw_model.predict(X_test_encoded)
    return rf_raw_pred, rf_raw_proba


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
            "n_jobs": 1,
        }
        _clf = RandomForestClassifier(**_params)
        _scores = cross_val_score(
            estimator=_clf,
            X=X_train_pca,
            y=y_train,
            cv=5,
            scoring="roc_auc",
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
    print(f"Best CV AUC: {_rf_pca_study.best_value:.4f}")

    _rf_pca_model = RandomForestClassifier(
        **_rf_pca_study.best_params,
        random_state=42,
        n_jobs=-1,
    )
    _rf_pca_model.fit(X_train_pca, y_train)

    rf_pca_proba = _rf_pca_model.predict_proba(X_test_pca)[:, 1]
    rf_pca_pred = _rf_pca_model.predict(X_test_pca)
    return rf_pca_pred, rf_pca_proba


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
            "n_jobs": 1,
        }
        _clf = RandomForestClassifier(**_params)
        _scores = cross_val_score(
            estimator=_clf,
            X=X_train_pacmap,
            y=y_train,
            cv=5,
            scoring="roc_auc",
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
    print(f"Best CV AUC: {_rf_pacmap_study.best_value:.4f}")

    _rf_pacmap_model = RandomForestClassifier(
        **_rf_pacmap_study.best_params,
        random_state=42,
        n_jobs=-1,
    )
    _rf_pacmap_model.fit(X_train_pacmap, y_train)

    rf_pacmap_proba = _rf_pacmap_model.predict_proba(X_test_pacmap)[:, 1]
    rf_pacmap_pred = _rf_pacmap_model.predict(X_test_pacmap)
    return rf_pacmap_pred, rf_pacmap_proba


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Results

    ### Metrics Summary

    We compare accuracy, precision, recall, F1 score, and ROC AUC across the three
    representations. ROC AUC is computed from predicted probabilities and is the primary
    metric given the class imbalance.
    """)
    return


@app.cell
def _(
    accuracy_score,
    f1_score,
    pl,
    precision_score,
    recall_score,
    rf_pacmap_pred,
    rf_pacmap_proba,
    rf_pca_pred,
    rf_pca_proba,
    rf_raw_pred,
    rf_raw_proba,
    roc_auc_score,
    y_test,
):
    _models = [
        ("Raw (10-dim)", rf_raw_pred, rf_raw_proba),
        ("PCA (2-dim)", rf_pca_pred, rf_pca_proba),
        ("PaCMAP (2-dim)", rf_pacmap_pred, rf_pacmap_proba),
    ]

    _names = []
    _accuracies = []
    _precisions = []
    _recalls = []
    _f1s = []
    _aucs = []

    for _name, _pred, _proba in _models:
        _names.append(_name)
        _accuracies.append(accuracy_score(y_test, _pred))
        _precisions.append(precision_score(y_test, _pred))
        _recalls.append(recall_score(y_test, _pred))
        _f1s.append(f1_score(y_test, _pred))
        _aucs.append(roc_auc_score(y_test, _proba))

    summary_df = pl.DataFrame({
        "Model": _names,
        "Accuracy": _accuracies,
        "Precision": _precisions,
        "Recall": _recalls,
        "F1": _f1s,
        "ROC AUC": _aucs,
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
    rf_pacmap_pred,
    rf_pca_pred,
    rf_raw_pred,
    y_test,
):
    _models = [
        ("Raw (10-dim)", rf_raw_pred),
        ("PCA (2-dim)", rf_pca_pred),
        ("PaCMAP (2-dim)", rf_pacmap_pred),
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### ROC Curves
    """)
    return


@app.cell
def _(
    plt,
    rf_pacmap_proba,
    rf_pca_proba,
    rf_raw_proba,
    roc_auc_score,
    roc_curve,
    y_test,
):
    _models = [
        ("Raw (10-dim)", rf_raw_proba),
        ("PCA (2-dim)", rf_pca_proba),
        ("PaCMAP (2-dim)", rf_pacmap_proba),
    ]

    _fig, _ax = plt.subplots(figsize=(8, 5))

    for _name, _proba in _models:
        _fpr, _tpr, _ = roc_curve(y_test, _proba)
        _auc = roc_auc_score(y_test, _proba)
        _ax.plot(
            _fpr,
            _tpr,
            label=f"{_name} (AUC = {_auc:.3f})",
            linewidth=2,
        )

    _ax.plot(
        [0, 1],
        [0, 1],
        "k--",
        label="Random (AUC = 0.500)",
        linewidth=1,
    )

    _ax.set_xlabel("False Positive Rate")
    _ax.set_ylabel("True Positive Rate")
    _ax.set_title("ROC Curve Comparison")
    _ax.legend(loc="lower right")

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
