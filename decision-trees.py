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
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GridSearchCV, train_test_split
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

    sns.set_style("whitegrid")
    return (
        DecisionTreeClassifier,
        GridSearchCV,
        OneHotEncoder,
        RandomForestClassifier,
        export_text,
        mo,
        np,
        pl,
        plot_tree,
        plt,
        roc_auc_score,
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
    y_test,
    y_train,
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
    return


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
        "max_depth": [3, 4],
        "min_samples_split": [6, 7, 8, 9, 10],
    }

    # Set up GridSearchCV
    grid_search = GridSearchCV(
        estimator=DecisionTreeClassifier(random_state=42),
        param_grid=param_grid,
        cv=5,
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
    return (dt_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Tree Visualization

    The `export_text` function displays the decision tree as a text-based representation.
    Each line shows a decision rule: the feature, threshold, and which branch to follow.
    Leaf nodes show the predicted class.
    """)
    return


@app.cell
def _(all_feature_names, dt_model, export_text):
    print(export_text(
        decision_tree=dt_model,
        feature_names=all_feature_names,
    ))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Tree Diagram

    The `plot_tree` function renders the decision tree as a diagram. Each node shows:
    - The split condition (feature and threshold)
    - Gini impurity
    - Number of samples
    - Class distribution
    """)
    return


@app.cell
def _(all_feature_names, dt_model, plot_tree, plt):
    _fig, _ax = plt.subplots(figsize=(10, 4))
    plot_tree(
        decision_tree=dt_model,
        feature_names=all_feature_names,
        class_names=["No", "Yes"],
        filled=True,
        rounded=True,
        ax=_ax,
        fontsize=8,
    )
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Bootstrap Sampling

    A **bootstrap sample** is created by sampling $n$ observations **with replacement** from a
    dataset of $n$ observations. Because sampling is done with replacement:

    - Some observations appear **more than once**
    - Some observations are **left out entirely**
    - On average, each bootstrap sample contains about **63.2%** of the unique original observations
      (the rest are duplicates)

    Observations not selected in a given bootstrap sample are called **out-of-bag (OOB)** samples.
    These can be used as a built-in validation set — no need for a separate holdout.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## From Bootstrap Samples to Diverse Trees

    Each bootstrap sample produces a **different** training set, so fitting the same decision tree
    algorithm to each one yields **different splits and tree structures**.

    Random forests add a second source of diversity: at each split, only a **random subset** of
    features is considered as candidates. For classification, the default is $\sqrt{p}$ features
    (where $p$ is the total number of features).

    $$\text{Same algorithm} + \text{different data} + \text{random feature subsets} \rightarrow \textbf{diverse trees}$$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Aggregation: Majority Voting

    Once we have an ensemble of diverse trees, each tree **votes** on the predicted class for a
    new observation. The final prediction is the **majority vote** across all trees.

    For probability estimates, the forest **averages** the predicted probabilities from each tree.
    This averaging reduces variance: individual trees may make errors, but uncorrelated errors
    **cancel out** through aggregation.

    This process — **B**ootstrap samples + **Agg**regation — is called **bagging**
    (Bootstrap AGGregatING).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Gini $\rightarrow$ Trees $\rightarrow$ Bootstrap $\rightarrow$ Random Forests

    Putting it all together:

    1. **Gini impurity** ($1 - \sum p_i^2$) quantifies class mixing; lower = purer
    2. **Decision trees** greedily select splits that minimize weighted Gini
    3. **Bootstrap sampling** creates diverse training sets for each tree
    4. **Random feature subsets** at each split add further diversity
    5. **Aggregation** (majority voting) combines diverse trees into a robust ensemble

    > Diversity through bootstrapping and random feature selection transforms high-variance trees
    > into a low-variance ensemble.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Random Forest

    Key hyperparameters:

    - `n_estimators`: Number of trees in the forest (more trees = more stable, but slower)
    - `max_depth`: Maximum depth of each tree (controls overfitting)
    - `min_samples_split`: Minimum samples required to split an internal node
    - `max_features`: Number of features considered at each split (`"sqrt"` for classification)
    """)
    return


@app.cell
def _(
    RandomForestClassifier,
    X_test_encoded,
    X_train_encoded,
    roc_auc_score,
    y_test,
    y_train,
):
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=7,
        min_samples_split=10,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    rf_model.fit(X_train_encoded, y_train)

    _rf_proba = rf_model.predict_proba(X_test_encoded)[:, 1]
    _rf_auc = roc_auc_score(y_test, _rf_proba)
    print(f"Random Forest Test AUC: {_rf_auc:.4f}")
    return (rf_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Feature Importance

    Gini-based feature importance measures the total reduction in Gini impurity brought by each
    feature across all trees in the forest. Features that frequently produce large purity gains
    rank higher.
    """)
    return


@app.cell
def _(all_feature_names, np, plt, rf_model):
    _importances = rf_model.feature_importances_
    _sorted_idx = np.argsort(_importances)

    _fig, _ax = plt.subplots(figsize=(8, 6))

    _ax.barh(
        range(len(_sorted_idx)),
        _importances[_sorted_idx],
        color="steelblue",
        edgecolor="k",
    )
    _ax.set_yticks(range(len(_sorted_idx)))
    _ax.set_yticklabels([all_feature_names[_i] for _i in _sorted_idx])
    _ax.set_xlabel("Gini Importance")
    _ax.set_title("Feature Importance (Random Forest)")

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
