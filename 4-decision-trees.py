import marimo

__generated_with = "0.19.4"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    import seaborn as sns
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import (
        ConfusionMatrixDisplay,
        accuracy_score,
        precision_score,
        recall_score,
        roc_auc_score,
        roc_curve,
    )
    from sklearn.model_selection import GridSearchCV, train_test_split
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.tree import DecisionTreeClassifier

    sns.set_style("whitegrid")
    return (
        ConfusionMatrixDisplay,
        DecisionTreeClassifier,
        GridSearchCV,
        OneHotEncoder,
        accuracy_score,
        mo,
        np,
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
    ## Decision Tree

    Decision trees partition the feature space into regions using a series of binary splits.
    They are interpretable and can capture non-linear relationships without feature engineering.
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
    param_grid = {
        "max_depth": [3, 4, 5, 6, 7, 10, 15],
        "min_samples_split": [2, 3, 4, 5, 6, 7, 8, 9, 10],
    }

    # Set up GridSearchCV
    grid_search = GridSearchCV(
        estimator=DecisionTreeClassifier(random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1,
    )

    # Fit with cross-validation
    grid_search.fit(X_train_encoded, y_train)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV AUC score: {grid_search.best_score_:.4f}")

    # Use the best model
    dt_model = grid_search.best_estimator_

    # Predict on test set
    dt_proba = dt_model.predict_proba(X_test_encoded)[:, 1]
    dt_pred = dt_model.predict(X_test_encoded)

    print(f"Tree depth: {dt_model.get_depth()}")
    print(f"Number of leaves: {dt_model.get_n_leaves()}")
    return dt_model, dt_pred, dt_proba


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Confusion Matrix Display

    The confusion matrix shows how predictions compare to actual labels. Rows are actual classes,
    columns are predicted classes. Diagonal entries are correct predictions.
    """)
    return


@app.cell
def _(ConfusionMatrixDisplay, dt_pred, logreg_pred, plt, y_test):
    _fig, (_ax1, _ax2) = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(10, 4),
    )

    ConfusionMatrixDisplay.from_predictions(
        y_true=y_test,
        y_pred=logreg_pred,
        ax=_ax1,
        cmap="Blues",
    )
    _ax1.set_title("Logistic Regression")

    ConfusionMatrixDisplay.from_predictions(
        y_true=y_test,
        y_pred=dt_pred,
        ax=_ax2,
        cmap="Blues",
    )
    _ax2.set_title("Decision Tree")

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
def _(dt_proba, logreg_proba, plt, roc_auc_score, roc_curve, y_test):
    # Calculate ROC curves
    logreg_fpr, logreg_tpr, _ = roc_curve(y_test, logreg_proba)
    dt_fpr, dt_tpr, _ = roc_curve(y_test, dt_proba)

    # Calculate AUC scores
    logreg_auc = roc_auc_score(y_test, logreg_proba)
    dt_auc = roc_auc_score(y_test, dt_proba)

    # Plot
    _fig, _ax = plt.subplots(figsize=(8, 6))

    _ax.plot(
        logreg_fpr,
        logreg_tpr,
        label=f"Logistic Regression (AUC = {logreg_auc:.3f})",
        linewidth=2,
    )
    _ax.plot(
        dt_fpr,
        dt_tpr,
        label=f"Decision Tree (AUC = {dt_auc:.3f})",
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
    return dt_auc, logreg_auc


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
    dt_model,
    np,
    permutation_importance,
    plt,
    y_test,
):
    # Calculate permutation importance
    perm_importance = permutation_importance(
        estimator=dt_model,
        X=X_test_encoded,
        y=y_test,
        n_repeats=10,
        random_state=42,
        scoring="roc_auc",
    )

    # Sort by importance
    sorted_idx = np.argsort(perm_importance.importances_mean)

    _fig, _ax = plt.subplots(figsize=(8, 6))

    _ax.barh(
        range(len(sorted_idx)),
        perm_importance.importances_mean[sorted_idx],
        xerr=perm_importance.importances_std[sorted_idx],
        color="steelblue",
        edgecolor="k",
    )
    _ax.set_yticks(range(len(sorted_idx)))
    _ax.set_yticklabels([all_feature_names[i] for i in sorted_idx])
    _ax.set_xlabel("Mean Importance (decrease in AUC)")
    _ax.set_title("Permutation Importance (Decision Tree)")

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
    dt_auc,
    dt_pred,
    logreg_auc,
    logreg_pred,
    pl,
    precision_score,
    recall_score,
    y_test,
):
    summary_df = pl.DataFrame({
        "Model": [
            "Logistic Regression",
            "Decision Tree (tuned)",
        ],
        "Accuracy": [
            accuracy_score(y_test, logreg_pred),
            accuracy_score(y_test, dt_pred),
        ],
        "Precision": [
            precision_score(y_test, logreg_pred),
            precision_score(y_test, dt_pred),
        ],
        "Recall": [
            recall_score(y_test, logreg_pred),
            recall_score(y_test, dt_pred),
        ],
        "AUC": [
            logreg_auc,
            dt_auc,
        ],
    })

    summary_df
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
