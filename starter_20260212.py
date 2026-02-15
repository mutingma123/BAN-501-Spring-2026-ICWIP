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
    from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.tree import DecisionTreeClassifier

    from tqdm.auto import tqdm

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sns.set_style("whitegrid")
    return (
        DecisionTreeClassifier,
        OneHotEncoder,
        mo,
        np,
        pl,
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
    ):
        if dt_params is None:
            dt_params = {}

        random_forest_trees = {}
        for _idx in tqdm(range(n_estimators)):

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

    return (fit_random_forest,)


@app.cell
def _(X_train_encoded, fit_random_forest, y_train):
    random_forest_trees = fit_random_forest(
        x_array=X_train_encoded,
        y_array=y_train,
        n_estimators=25,
    )
    return (random_forest_trees,)


@app.cell
def _(X_test_encoded, np, random_forest_trees, tqdm):
    _all_proba = []
    for _tree_idx, _current_tree in tqdm(random_forest_trees.items()):
        _proba = _current_tree.predict_proba(X_test_encoded)[:, 1]
        _all_proba.append(_proba)

    positive_probabilities = np.mean(_all_proba, axis=0)
    return (positive_probabilities,)


@app.cell
def _(positive_probabilities):
    positive_probabilities
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
