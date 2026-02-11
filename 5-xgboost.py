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
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.preprocessing import OneHotEncoder
    from xgboost import XGBClassifier

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sns.set_style("whitegrid")
    return (
        ConfusionMatrixDisplay,
        OneHotEncoder,
        RandomForestClassifier,
        XGBClassifier,
        accuracy_score,
        cross_val_score,
        f1_score,
        mo,
        np,
        optuna,
        permutation_importance,
        pl,
        plt,
        precision_score,
        recall_score,
        roc_auc_score,
        roc_curve,
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
    return X_test_encoded, X_train_encoded, all_feature_names


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## XGBoost

    XGBoost (eXtreme Gradient Boosting) builds an ensemble of decision trees **sequentially**,
    where each new tree corrects the errors of the previous ensemble. This differs from random
    forests, which build trees **independently** in parallel.

    **How it works**: At each step, XGBoost fits a new tree to the negative gradient of the loss
    function (hence "gradient boosting"). The final prediction is the sum of all trees' predictions.

    **Hyperparameters tuned with Optuna** (6):

    - `learning_rate` (0.01–0.3): Step size for each boosting round. Smaller values require more trees but often generalize better.
    - `n_estimators` (50–500): Number of boosting rounds (trees in the ensemble).
    - `max_depth` (3–10): Maximum depth of each tree. Controls model complexity.
    - `min_child_weight` (1–10): Minimum sum of instance weight in a child node. Higher values prevent overfitting.
    - `subsample` (0.5–1.0): Fraction of training samples used per tree (row subsampling).
    - `colsample_bytree` (0.5–1.0): Fraction of features used per tree (column subsampling).
    """)
    return


@app.cell
def _(
    XGBClassifier,
    X_test_encoded,
    X_train_encoded,
    cross_val_score,
    np,
    optuna,
    y_train,
):
    def _xgb_objective(_trial):
        _params = {
            "learning_rate": _trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": _trial.suggest_int("n_estimators", 50, 500),
            "max_depth": _trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": _trial.suggest_int("min_child_weight", 1, 10),
            "subsample": _trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": _trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "eval_metric": "logloss",
            "random_state": 42,
            "n_jobs": 1,
        }
        _clf = XGBClassifier(**_params)
        _scores = cross_val_score(
            estimator=_clf,
            X=X_train_encoded,
            y=y_train,
            cv=5,
            scoring="roc_auc",
        )
        return np.mean(_scores)

    _xgb_study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    _xgb_study.optimize(
        _xgb_objective,
        n_trials=100,
        show_progress_bar=True,
    )

    print(f"Best parameters: {_xgb_study.best_params}")
    print(f"Best CV AUC score: {_xgb_study.best_value:.4f}")

    # Refit best model on full training data
    xgb_model = XGBClassifier(
        **_xgb_study.best_params,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    xgb_model.fit(X_train_encoded, y_train)

    # Predict on test set
    xgb_proba = xgb_model.predict_proba(X_test_encoded)[:, 1]
    xgb_pred = xgb_model.predict(X_test_encoded)
    return xgb_model, xgb_pred, xgb_proba


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Random Forest: Optuna

    We use Optuna to tune the random forest with fewer trials than XGBoost since
    each trial is more computationally expensive due to the ensemble. This serves as
    our baseline for comparison with XGBoost.
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
    def _rf_objective(_trial):
        _params = {
            "n_estimators": _trial.suggest_int("n_estimators", 50, 300),
            "max_depth": _trial.suggest_int("max_depth", 3, 20),
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
            X=X_train_encoded,
            y=y_train,
            cv=5,
            scoring="roc_auc",
        )
        return np.mean(_scores)

    _rf_study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    _rf_study.optimize(
        _rf_objective,
        n_trials=50,
        show_progress_bar=True,
    )

    print(f"Best parameters: {_rf_study.best_params}")
    print(f"Best CV AUC score: {_rf_study.best_value:.4f}")

    # Refit best model on full training data
    rf_optuna_model = RandomForestClassifier(
        **_rf_study.best_params,
        random_state=42,
        n_jobs=1,
    )
    rf_optuna_model.fit(X_train_encoded, y_train)

    # Predict on test set
    rf_optuna_proba = rf_optuna_model.predict_proba(X_test_encoded)[:, 1]
    rf_optuna_pred = rf_optuna_model.predict(X_test_encoded)
    return rf_optuna_model, rf_optuna_pred, rf_optuna_proba


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Confusion Matrix Display

    The confusion matrix shows how predictions compare to actual labels. Rows are actual classes,
    columns are predicted classes. Diagonal entries are correct predictions.
    """)
    return


@app.cell
def _(ConfusionMatrixDisplay, plt, rf_optuna_pred, xgb_pred, y_test):
    _models = [
        ("XGBoost (Optuna)", xgb_pred),
        ("RF (Optuna)", rf_optuna_pred),
    ]

    _fig, _axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(10, 4),
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
    ## ROC Curves

    The ROC curve plots True Positive Rate vs False Positive Rate at all classification thresholds.
    A model with perfect discrimination has AUC = 1.0. The diagonal represents random guessing.
    """)
    return


@app.cell
def _(plt, rf_optuna_proba, roc_auc_score, roc_curve, xgb_proba, y_test):
    _models = [
        ("XGBoost (Optuna)", xgb_proba),
        ("RF (Optuna)", rf_optuna_proba),
    ]

    _fig, _ax = plt.subplots(figsize=(8, 6))

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

    # Store AUC values for summary table
    xgb_auc = roc_auc_score(y_test, xgb_proba)
    rf_optuna_auc = roc_auc_score(y_test, rf_optuna_proba)
    return rf_optuna_auc, xgb_auc


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Permutation Importance

    Permutation importance measures feature importance by shuffling each feature and measuring
    the decrease in model performance. Features whose shuffling causes large performance drops
    are considered important.
    """)
    return


@app.cell
def _(
    X_test_encoded,
    all_feature_names,
    np,
    permutation_importance,
    plt,
    xgb_model,
    y_test,
):
    # Calculate permutation importance for XGBoost
    _perm_importance = permutation_importance(
        estimator=xgb_model,
        X=X_test_encoded,
        y=y_test,
        n_repeats=10,
        random_state=42,
        scoring="roc_auc",
    )

    # Sort by importance
    _sorted_idx = np.argsort(_perm_importance.importances_mean)

    _fig, _ax = plt.subplots(figsize=(8, 6))

    _ax.barh(
        range(len(_sorted_idx)),
        _perm_importance.importances_mean[_sorted_idx],
        xerr=_perm_importance.importances_std[_sorted_idx],
        color="steelblue",
        edgecolor="k",
    )
    _ax.set_yticks(range(len(_sorted_idx)))
    _ax.set_yticklabels([all_feature_names[_i] for _i in _sorted_idx])
    _ax.set_xlabel("Mean Importance (decrease in AUC)")
    _ax.set_title("Permutation Importance: XGBoost (Optuna)")

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(
    X_test_encoded,
    all_feature_names,
    np,
    permutation_importance,
    plt,
    rf_optuna_model,
    y_test,
):
    # Calculate permutation importance for random forest
    _perm_importance = permutation_importance(
        estimator=rf_optuna_model,
        X=X_test_encoded,
        y=y_test,
        n_repeats=10,
        random_state=42,
        scoring="roc_auc",
    )

    # Sort by importance
    _sorted_idx = np.argsort(_perm_importance.importances_mean)

    _fig, _ax = plt.subplots(figsize=(8, 6))

    _ax.barh(
        range(len(_sorted_idx)),
        _perm_importance.importances_mean[_sorted_idx],
        xerr=_perm_importance.importances_std[_sorted_idx],
        color="coral",
        edgecolor="k",
    )
    _ax.set_yticks(range(len(_sorted_idx)))
    _ax.set_yticklabels([all_feature_names[_i] for _i in _sorted_idx])
    _ax.set_xlabel("Mean Importance (decrease in AUC)")
    _ax.set_title("Permutation Importance: Random Forest (Optuna)")

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    Final comparison of both models on the test set.
    """)
    return


@app.cell
def _(
    accuracy_score,
    f1_score,
    pl,
    precision_score,
    recall_score,
    rf_optuna_auc,
    rf_optuna_pred,
    xgb_auc,
    xgb_pred,
    y_test,
):
    _models = [
        ("XGBoost (Optuna)", xgb_pred, xgb_auc),
        ("RF (Optuna)", rf_optuna_pred, rf_optuna_auc),
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


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
