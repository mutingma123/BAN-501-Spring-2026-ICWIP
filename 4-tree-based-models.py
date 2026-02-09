import marimo

__generated_with = "0.19.4"
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
    from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.tree import DecisionTreeClassifier

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sns.set_style("whitegrid")
    return (
        ConfusionMatrixDisplay,
        DecisionTreeClassifier,
        GridSearchCV,
        OneHotEncoder,
        RandomForestClassifier,
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
        sm,
        smf,
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
    ## Logistic Regression

    We fit a logistic regression model using statsmodels. This provides detailed statistical output
    including coefficients, standard errors, and p-values.
    """)
    return


@app.cell
def _(
    X_test,
    X_train,
    categorical_features,
    numeric_features,
    smf,
    y_train,
    y_test,
):
    # Create training DataFrame with target column
    _train_df = X_train.to_pandas()
    _train_df["y"] = y_train

    # Create test DataFrame
    _test_df = X_test.to_pandas()
    _test_df["y"] = y_test

    # Build formula: statsmodels auto-detects string columns as categorical
    _formula = "y ~ " + " + ".join(numeric_features + categorical_features)
    print(f"Formula: {_formula}\n")

    # Fit logistic regression using formula API
    logit_model = smf.logit(
        formula=_formula,
        data=_train_df,
    ).fit(disp=False)

    print(logit_model.summary())

    # Predict on test set
    logreg_proba = logit_model.predict(_test_df)
    logreg_pred = (logreg_proba >= 0.5).astype(int)
    return logreg_pred, logreg_proba


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Decision Tree: GridSearchCV

    Decision trees partition the feature space into regions using a series of binary splits.
    They are interpretable and can capture non-linear relationships without feature engineering.

    We use `GridSearchCV` to exhaustively search over a grid of hyperparameters, selecting the
    combination that maximizes cross-validated AUC.
    """)
    return


@app.cell
def _(
    DecisionTreeClassifier,
    GridSearchCV,
    X_test_encoded,
    X_train_encoded,
    y_train,
):
    # Define parameter grid for hyperparameter tuning
    _param_grid = {
        "max_depth": [3, 5, 7, 10, 15],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
    }

    # Set up GridSearchCV
    _grid_search = GridSearchCV(
        estimator=DecisionTreeClassifier(random_state=42),
        param_grid=_param_grid,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1,
    )

    # Fit with cross-validation
    _grid_search.fit(X_train_encoded, y_train)

    print(f"Best parameters: {_grid_search.best_params_}")
    print(f"Best CV AUC score: {_grid_search.best_score_:.4f}")

    # Use the best model
    dt_grid_model = _grid_search.best_estimator_

    # Predict on test set
    dt_grid_proba = dt_grid_model.predict_proba(X_test_encoded)[:, 1]
    dt_grid_pred = dt_grid_model.predict(X_test_encoded)

    print(f"Tree depth: {dt_grid_model.get_depth()}")
    print(f"Number of leaves: {dt_grid_model.get_n_leaves()}")
    return dt_grid_model, dt_grid_pred, dt_grid_proba


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Decision Tree: Optuna

    [Optuna](https://optuna.org/) uses Bayesian optimization (Tree-structured Parzen Estimator)
    to search the hyperparameter space more efficiently than grid search. Instead of evaluating
    every combination, Optuna learns from previous trials to focus on promising regions.
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
        n_trials=100,
        show_progress_bar=True,
    )

    print(f"Best parameters: {_dt_study.best_params}")
    print(f"Best CV AUC score: {_dt_study.best_value:.4f}")

    # Refit best model on full training data
    dt_optuna_model = DecisionTreeClassifier(
        **_dt_study.best_params,
        random_state=42,
    )
    dt_optuna_model.fit(X_train_encoded, y_train)

    # Predict on test set
    dt_optuna_proba = dt_optuna_model.predict_proba(X_test_encoded)[:, 1]
    dt_optuna_pred = dt_optuna_model.predict(X_test_encoded)

    print(f"Tree depth: {dt_optuna_model.get_depth()}")
    print(f"Number of leaves: {dt_optuna_model.get_n_leaves()}")
    return dt_optuna_model, dt_optuna_pred, dt_optuna_proba


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Random Forest: GridSearchCV

    A random forest is an ensemble of decision trees, each trained on a bootstrap sample of the
    data with a random subset of features considered at each split. This reduces variance and
    typically improves generalization over a single tree.
    """)
    return


@app.cell
def _(
    GridSearchCV,
    RandomForestClassifier,
    X_test_encoded,
    X_train_encoded,
    y_train,
):
    # Define parameter grid
    _param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [5, 7, 10, 15],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    }

    # Set up GridSearchCV
    # n_jobs=1 on estimator, n_jobs=-1 on GridSearchCV to avoid nested parallelism
    _grid_search = GridSearchCV(
        estimator=RandomForestClassifier(
            random_state=42,
            n_jobs=1,
        ),
        param_grid=_param_grid,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1,
    )

    # Fit with cross-validation
    _grid_search.fit(X_train_encoded, y_train)

    print(f"Best parameters: {_grid_search.best_params_}")
    print(f"Best CV AUC score: {_grid_search.best_score_:.4f}")

    # Use the best model
    rf_grid_model = _grid_search.best_estimator_

    # Predict on test set
    rf_grid_proba = rf_grid_model.predict_proba(X_test_encoded)[:, 1]
    rf_grid_pred = rf_grid_model.predict(X_test_encoded)
    return rf_grid_model, rf_grid_pred, rf_grid_proba


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Random Forest: Optuna

    We use Optuna to tune the random forest with fewer trials than the decision tree since
    each trial is more computationally expensive due to the ensemble.
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
def _(
    ConfusionMatrixDisplay,
    dt_grid_pred,
    dt_optuna_pred,
    logreg_pred,
    plt,
    rf_grid_pred,
    rf_optuna_pred,
    y_test,
):
    _models = [
        ("Logistic Regression", logreg_pred),
        ("DT (GridSearchCV)", dt_grid_pred),
        ("DT (Optuna)", dt_optuna_pred),
        ("RF (GridSearchCV)", rf_grid_pred),
        ("RF (Optuna)", rf_optuna_pred),
    ]

    _fig, _axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(15, 8),
    )

    for _i, (_name, _pred) in enumerate(_models):
        _ax = _axes[_i // 3, _i % 3]
        ConfusionMatrixDisplay.from_predictions(
            y_true=y_test,
            y_pred=_pred,
            ax=_ax,
            cmap="Blues",
        )
        _ax.set_title(_name)

    # Hide the empty 6th subplot
    _axes[1, 2].set_visible(False)

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
def _(
    dt_grid_proba,
    dt_optuna_proba,
    logreg_proba,
    plt,
    rf_grid_proba,
    rf_optuna_proba,
    roc_auc_score,
    roc_curve,
    y_test,
):
    _models = [
        ("Logistic Regression", logreg_proba),
        ("DT (GridSearchCV)", dt_grid_proba),
        ("DT (Optuna)", dt_optuna_proba),
        ("RF (GridSearchCV)", rf_grid_proba),
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
    logreg_auc = roc_auc_score(y_test, logreg_proba)
    dt_grid_auc = roc_auc_score(y_test, dt_grid_proba)
    dt_optuna_auc = roc_auc_score(y_test, dt_optuna_proba)
    rf_grid_auc = roc_auc_score(y_test, rf_grid_proba)
    rf_optuna_auc = roc_auc_score(y_test, rf_optuna_proba)
    return dt_grid_auc, dt_optuna_auc, logreg_auc, rf_grid_auc, rf_optuna_auc


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Permutation Importance

    Permutation importance measures feature importance by shuffling each feature and measuring
    the decrease in model performance. Features whose shuffling causes large performance drops
    are considered important.

    We show permutation importance for the Optuna-tuned decision tree and random forest models.
    """)
    return


@app.cell
def _(
    X_test_encoded,
    all_feature_names,
    dt_optuna_model,
    np,
    permutation_importance,
    plt,
    y_test,
):
    # Calculate permutation importance for decision tree
    _perm_importance = permutation_importance(
        estimator=dt_optuna_model,
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
    _ax.set_title("Permutation Importance: Decision Tree (Optuna)")

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

    Final comparison of all models on the test set.
    """)
    return


@app.cell
def _(
    accuracy_score,
    dt_grid_auc,
    dt_grid_pred,
    dt_optuna_auc,
    dt_optuna_pred,
    f1_score,
    logreg_auc,
    logreg_pred,
    pl,
    precision_score,
    recall_score,
    rf_grid_auc,
    rf_grid_pred,
    rf_optuna_auc,
    rf_optuna_pred,
    y_test,
):
    _models = [
        ("Logistic Regression", logreg_pred, logreg_auc),
        ("DT (GridSearchCV)", dt_grid_pred, dt_grid_auc),
        ("DT (Optuna)", dt_optuna_pred, dt_optuna_auc),
        ("RF (GridSearchCV)", rf_grid_pred, rf_grid_auc),
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
