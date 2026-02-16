import marimo

__generated_with = "0.19.8"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    import matplotlib.pyplot as plt
    import numpy as np
    import optuna
    import polars as pl
    import seaborn as sns
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import (
        ConfusionMatrixDisplay,
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
        roc_curve,
    )
    from sklearn.model_selection import (
        StratifiedKFold,
        cross_val_score,
        train_test_split,
    )
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.tree import DecisionTreeClassifier

    from tqdm.auto import tqdm

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sns.set_style("white")
    return (
        ConfusionMatrixDisplay,
        DecisionTreeClassifier,
        OneHotEncoder,
        StratifiedKFold,
        accuracy_score,
        cross_val_score,
        f1_score,
        mo,
        np,
        optuna,
        pl,
        plt,
        precision_score,
        recall_score,
        roc_auc_score,
        roc_curve,
        sns,
        tqdm,
        train_test_split,
    )


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
    # Define features to use
    numeric_features = ["age", "balance", "duration", "campaign"]
    categorical_features = ["marital", "education", "housing"]
    target_column = "y"

    # Sample data
    model_data = raw_data.select(
        numeric_features + categorical_features + [target_column]
    ).sample(
        n=10_000,
        seed=42,
        shuffle=True,
    )

    # Split features and target
    X = model_data.drop(target_column)
    y = model_data[target_column].to_numpy()

    # Train/test split with stratification
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
    # One-hot encode categorical features
    # drop="first" avoids multicollinearity for statsmodels
    encoder = OneHotEncoder(
        sparse_output=False,
        handle_unknown="ignore",
        drop="first",
    )

    # Fit on training data only (accepts polars DataFrames directly)
    encoder.fit(X_train.select(categorical_features))

    # Transform returns polars DataFrames with set_output
    encoder.set_output(transform="polars")
    _X_train_cat = encoder.transform(X_train.select(categorical_features))
    _X_test_cat = encoder.transform(X_test.select(categorical_features))

    # Combine numeric + encoded categorical using polars concat
    X_train_encoded = pl.concat([
        X_train.select(numeric_features),
        _X_train_cat,
    ], how="horizontal").to_numpy()

    X_test_encoded = pl.concat([
        X_test.select(numeric_features),
        _X_test_cat,
    ], how="horizontal").to_numpy()

    # Build feature names for interpretability
    encoded_cat_names = list(encoder.get_feature_names_out(categorical_features))
    all_feature_names = numeric_features + encoded_cat_names

    print(f"Encoded feature count: {len(all_feature_names)}")
    print(f"Features: {all_feature_names}")
    return X_test_encoded, X_train_encoded


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Assignment: Build a Random Forest from Scratch

    In this assignment you will implement a random forest classifier **from scratch** using
    `DecisionTreeClassifier` as the base learner, tune it with Optuna, and compare it against an
    Optuna-optimized single decision tree.

    ### Task 1 — From-Scratch Random Forest

    Build a function that implements the core random forest algorithm:

    1. **Bootstrap sampling** — For each tree, draw a random sample *with replacement* from the
       training data (same size as the original training set).
    2. **Fit individual trees** — Train a `DecisionTreeClassifier` on each bootstrap sample.
    3. **Aggregate predictions** — For each test observation, collect the predicted *probabilities*
       from every tree and average them to produce the ensemble's prediction.

    ### Task 2 — Optuna Hyperparameter Tuning

    Use Optuna to optimize hyperparameters for **both** models. Use 5-fold cross-validated AUC
    (`scoring="roc_auc"`) as the objective.

    **Single decision tree — suggested parameter ranges:**

    | Parameter            | Range / Choices                |
    |----------------------|-------------------------------|
    | `max_depth`          | 2 – 20 (int)                  |
    | `min_samples_split`  | 2 – 20 (int)                  |
    | `min_samples_leaf`   | 1 – 10 (int)                  |
    | `max_features`       | `"sqrt"`, `"log2"`, `None`    |

    **From-scratch random forest — suggested parameter ranges:**

    All of the single-tree parameters above, plus:

    | Parameter            | Range / Choices                |
    |----------------------|-------------------------------|
    | `n_estimators`       | 50 – 300 (int)                |

    ### Task 3 — Compare Performance

    Evaluate both tuned models on the **test set** and compare:

    - ROC AUC
    - Any other metrics you find informative (accuracy, precision, recall, F1, etc.)
    """)
    return


@app.cell
def _(X_test_encoded, X_train_encoded, y_test, y_train):
    print(f' - {X_train_encoded.shape = }')
    print(f' - {X_test_encoded.shape = }')
    print(f' - {y_train.shape = }')
    print(f' - {y_test.shape = }')
    return


@app.cell
def _(DecisionTreeClassifier, np, tqdm):
    def get_bootstrap_samples(
        x_array: np.ndarray,
        y_array: np.ndarray,
        seed: int = 42,
    ):
        _n = len(x_array)

        np.random.seed(seed)
        selected_indices = np.random.choice(
            a=range(_n),
            replace=True,
            size=_n
        )

        return x_array[selected_indices], y_array[selected_indices]


    def fit_random_forest(
        x_array: np.ndarray,
        y_array: np.ndarray,
        n_estimators: int = 25,
        dt_params: dict | None = None,
        show_progress: bool = True,
    ):
        if dt_params is None:
            dt_params = {}

        random_forest_trees = {}
        for _idx in tqdm(range(n_estimators), disable=not show_progress):

            bootstrap_x, bootstrap_y = get_bootstrap_samples(
                x_array=x_array,
                y_array=y_array,
                seed=_idx,
            )
            _tree_params = {**dt_params, 'random_state': _idx}

            dt = DecisionTreeClassifier(**_tree_params)
            dt.fit(bootstrap_x, bootstrap_y)
            random_forest_trees[_idx] = dt

        return random_forest_trees

    def predict_random_forest(
        rf_trees: dict,
        x_array: np.ndarray,
        voting: str = "soft",
        threshold: float = 0.5,
    ):
        if voting == "soft":
            _all_proba = [
                _tree.predict_proba(x_array)[:, 1]
                for _tree in rf_trees.values()
            ]
            probabilities = np.mean(_all_proba, axis=0)
            predictions = (probabilities >= threshold).astype(int)
        elif voting == "hard":
            _all_preds = [
                _tree.predict(x_array)
                for _tree in rf_trees.values()
            ]
            probabilities = np.mean(_all_preds, axis=0)
            predictions = (probabilities >= threshold).astype(int)
        else:
            raise ValueError(f"voting must be 'soft' or 'hard', got '{voting}'")

        return predictions, probabilities

    return fit_random_forest, predict_random_forest


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Prediction Aggregation Strategies

    A random forest aggregates predictions from individual trees. Two common strategies:

    - **Hard voting**: Each tree casts a class label (0 or 1). The ensemble prediction is the
      majority vote. The "probability" is the proportion of trees that voted for class 1.
    - **Soft voting**: Each tree outputs a probability for class 1. The ensemble averages these
      probabilities, then applies a threshold (default 0.5) to produce the class label.

    Soft voting is generally preferred because it preserves probability information, which is
    important for metrics like AUC that depend on ranking observations by predicted risk.
    """)
    return


@app.cell
def _(X_train_encoded, fit_random_forest, y_train):
    random_forest_trees = fit_random_forest(
        x_array=X_train_encoded,
        y_array=y_train,
        n_estimators=25,
    )
    return (random_forest_trees,)


@app.cell
def _(X_test_encoded, predict_random_forest, random_forest_trees):
    rf_soft_pred, rf_soft_proba = predict_random_forest(
        rf_trees=random_forest_trees,
        x_array=X_test_encoded,
        voting="soft",
    )
    print(f"Soft voting — predicted positive rate: {rf_soft_pred.mean():.2%}")
    return (rf_soft_pred,)


@app.cell
def _(X_test_encoded, predict_random_forest, random_forest_trees):
    rf_hard_pred, rf_hard_vote_proportions = predict_random_forest(
        rf_trees=random_forest_trees,
        x_array=X_test_encoded,
        voting="hard",
    )
    print(f"Hard voting — predicted positive rate: {rf_hard_pred.mean():.2%}")
    return (rf_hard_pred,)


@app.cell
def _(np, rf_hard_pred, rf_soft_pred):
    _agreement = np.mean(rf_soft_pred == rf_hard_pred)
    _n_disagree = np.sum(rf_soft_pred != rf_hard_pred)
    print(f"Hard vs soft voting agreement: {_agreement:.2%} ({_n_disagree} disagreements)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Optuna Hyperparameter Tuning

    We use [Optuna](https://optuna.org/) to search for the best hyperparameters. Optuna uses
    Tree-structured Parzen Estimators (TPE) to efficiently explore the parameter space.

    We tune two models:

    1. **Single decision tree** — 100 trials, 5-fold CV AUC via `cross_val_score`
    2. **From-scratch random forest** — 30 trials, 5-fold CV AUC via manual `StratifiedKFold`
       (since our RF is not a scikit-learn estimator)
    """)
    return


@app.cell
def _(
    DecisionTreeClassifier,
    X_test_encoded,
    X_train_encoded,
    cross_val_score,
    np,
    optuna,
    y_train,
):
    def _dt_objective(_trial):
        _params = {
            "max_depth": _trial.suggest_int("max_depth", 2, 20),
            "min_samples_split": _trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": _trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": _trial.suggest_categorical(
                "max_features", ["sqrt", "log2", None],
            ),
            "random_state": 42,
        }
        _clf = DecisionTreeClassifier(**_params)
        _scores = cross_val_score(
            estimator=_clf,
            X=X_train_encoded,
            y=y_train,
            cv=5,
            scoring="roc_auc",
        )
        return np.mean(_scores)

    _dt_study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    _dt_study.optimize(
        _dt_objective,
        n_trials=30,
        show_progress_bar=True,
        n_jobs=8,
    )

    print(f"Best parameters: {_dt_study.best_params}")
    print(f"Best CV AUC: {_dt_study.best_value:.4f}")

    # Refit best model on full training data
    _dt_best = DecisionTreeClassifier(
        **_dt_study.best_params,
        random_state=42,
    )
    _dt_best.fit(X_train_encoded, y_train)

    dt_optuna_pred = _dt_best.predict(X_test_encoded)
    dt_optuna_proba = _dt_best.predict_proba(X_test_encoded)[:, 1]
    return dt_optuna_pred, dt_optuna_proba


@app.cell
def _(
    StratifiedKFold,
    X_test_encoded,
    X_train_encoded,
    fit_random_forest,
    np,
    optuna,
    predict_random_forest,
    roc_auc_score,
    y_train,
):
    def _rf_objective(_trial):
        _params = {
            "max_depth": _trial.suggest_int("max_depth", 2, 20),
            "min_samples_split": _trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": _trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": _trial.suggest_categorical(
                "max_features", ["sqrt", "log2", None],
            ),
        }
        _n_estimators = _trial.suggest_int("n_estimators", 50, 300)

        _skf = StratifiedKFold(
            n_splits=5,
            shuffle=False,
            random_state=42,
        )
        _fold_aucs = []
        for _train_idx, _val_idx in _skf.split(X_train_encoded, y_train):
            _X_fold_train = X_train_encoded[_train_idx]
            _y_fold_train = y_train[_train_idx]
            _X_fold_val = X_train_encoded[_val_idx]
            _y_fold_val = y_train[_val_idx]

            _trees = fit_random_forest(
                x_array=_X_fold_train,
                y_array=_y_fold_train,
                n_estimators=_n_estimators,
                dt_params=_params,
                show_progress=False,
            )
            _, _proba = predict_random_forest(
                rf_trees=_trees,
                x_array=_X_fold_val,
                voting="soft",
            )
            _fold_aucs.append(roc_auc_score(_y_fold_val, _proba))

        return np.mean(_fold_aucs)

    _rf_study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    _rf_study.optimize(
        _rf_objective,
        n_trials=30,
        show_progress_bar=True,
        n_jobs=8
    )

    print(f"Best parameters: {_rf_study.best_params}")
    print(f"Best CV AUC: {_rf_study.best_value:.4f}")

    # Refit on full training data
    _best_params = {
        k: v for k, v in _rf_study.best_params.items()
        if k != "n_estimators"
    }
    rf_optuna_trees = fit_random_forest(
        x_array=X_train_encoded,
        y_array=y_train,
        n_estimators=_rf_study.best_params["n_estimators"],
        dt_params=_best_params,
        show_progress=True,
    )

    rf_optuna_pred, rf_optuna_proba = predict_random_forest(
        rf_trees=rf_optuna_trees,
        x_array=X_test_encoded,
        voting="soft",
    )
    return rf_optuna_pred, rf_optuna_proba


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Evaluation

    We compare the Optuna-tuned single decision tree against the from-scratch random forest
    on the held-out test set using ROC curves, confusion matrices, and a summary metrics table.
    """)
    return


@app.cell
def _(
    dt_optuna_proba,
    plt,
    rf_optuna_proba,
    roc_auc_score,
    roc_curve,
    sns,
    y_test,
):
    _models = [
        ("DT", dt_optuna_proba, "steelblue"),
        ("RF", rf_optuna_proba, "coral"),
    ]

    sns.set_style('whitegrid')
    _fig, _ax = plt.subplots(figsize=(9, 5))

    for _name, _proba, _color in _models:
        _fpr, _tpr, _ = roc_curve(y_test, _proba)
        _auc = roc_auc_score(y_test, _proba)
        _ax.plot(
            _fpr,
            _tpr,
            label=f"{_name} (AUC = {_auc:.3f})",
            color=_color,
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
    _ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.01))


    plt.tight_layout()
    plt.show()

    dt_test_auc = roc_auc_score(y_test, dt_optuna_proba)
    rf_test_auc = roc_auc_score(y_test, rf_optuna_proba)
    return dt_test_auc, rf_test_auc


@app.cell
def _(
    ConfusionMatrixDisplay,
    dt_optuna_pred,
    plt,
    rf_optuna_pred,
    sns,
    y_test,
):
    _models = [
        ("DT (Optuna)", dt_optuna_pred, "Blues"),
        ("RF (From-Scratch, Optuna)", rf_optuna_pred, "Oranges"),
    ]

    sns.set_style('white')
    _fig, _axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(10, 4.5),
    )

    for _i, (_name, _pred, _cmap) in enumerate(_models):
        ConfusionMatrixDisplay.from_predictions(
            y_true=y_test,
            y_pred=_pred,
            ax=_axes[_i],
            cmap=_cmap,
        )
        _axes[_i].set_title(_name)

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(
    accuracy_score,
    dt_optuna_pred,
    dt_test_auc,
    f1_score,
    pl,
    precision_score,
    recall_score,
    rf_optuna_pred,
    rf_test_auc,
    y_test,
):
    _models = [
        ("DT (Optuna)", dt_optuna_pred, dt_test_auc),
        ("RF (From-Scratch, Optuna)", rf_optuna_pred, rf_test_auc),
    ]

    _names = []
    _accuracies = []
    _precisions = []
    _recalls = []
    _f1s = []
    _aucs = []

    for _name, _pred, _auc in _models:
        _names.append(_name)
        _accuracies.append(accuracy_score(y_test, _pred))
        _precisions.append(precision_score(y_test, _pred))
        _recalls.append(recall_score(y_test, _pred))
        _f1s.append(f1_score(y_test, _pred))
        _aucs.append(_auc)

    summary_df = pl.DataFrame({
        "Model": _names,
        "Accuracy": _accuracies,
        "Precision": _precisions,
        "Recall": _recalls,
        "F1": _f1s,
        "AUC": _aucs,
    })

    summary_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Interpretation

    The random forest should outperform the single decision tree on AUC. This improvement comes
    from **variance reduction via bagging**: each tree sees a different bootstrap sample, so their
    errors are partially uncorrelated. Averaging these diverse predictions produces a smoother,
    more stable decision boundary.

    **Trade-off**: The from-scratch RF is slower to train (many trees x many Optuna trials) and
    harder to interpret than a single tree. In practice, `sklearn.ensemble.RandomForestClassifier`
    handles this efficiently with parallel fitting and C-optimized internals.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
