import marimo

__generated_with = "0.19.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    import seaborn as sns
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
        roc_curve,
    )
    from sklearn.inspection import permutation_importance
    from sklearn.tree import DecisionTreeClassifier, plot_tree

    sns.set_style("whitegrid")
    return (
        DecisionTreeClassifier,
        RandomForestClassifier,
        accuracy_score,
        confusion_matrix,
        f1_score,
        mo,
        np,
        permutation_importance,
        pl,
        plot_tree,
        plt,
        precision_score,
        recall_score,
        roc_auc_score,
        roc_curve,
        sns,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Decision Trees and Random Forests

    ## Tree-Based Methods vs. Logistic Regression

    While logistic regression models the relationship between features and outcome using a linear
    combination passed through a sigmoid function, **decision trees** take a fundamentally different
    approach: they recursively partition the feature space into regions and assign predictions
    based on the majority class in each region.

    **Key differences:**
    - Logistic regression assumes a specific functional form (linear in log-odds)
    - Decision trees make no assumptions about the relationship between features and outcome
    - Trees can capture non-linear relationships and interactions automatically
    - Logistic regression provides coefficients and odds ratios; trees provide feature importance
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How Decision Trees Work

    A decision tree builds a model by recursively splitting the data based on feature values:

    1. **Start** with all training data at the root node
    2. **Find the best split**: For each feature, find the threshold that best separates the classes
    3. **Split the data** into two child nodes based on the best feature and threshold
    4. **Repeat** recursively for each child node until a stopping criterion is met
    5. **Assign predictions** based on the majority class in each leaf node

    The "best split" is determined by maximizing the reduction in **impurity** (how mixed the
    classes are in a node).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Impurity Measures: Gini and Entropy

    Two common measures of node impurity for classification:

    **Gini Impurity:**
    $$G = 1 - \sum_{k=1}^{K} p_k^2$$

    where $p_k$ is the proportion of class $k$ in the node.
    - $G = 0$: Pure node (all samples from one class)
    - $G = 0.5$: Maximum impurity for binary classification (50/50 split)

    **Entropy (Information Gain):**
    $$H = -\sum_{k=1}^{K} p_k \log_2(p_k)$$

    - $H = 0$: Pure node
    - $H = 1$: Maximum entropy for binary classification

    Both measures give similar results in practice. Gini is computationally faster and is the
    default in scikit-learn.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Dataset: Bank Marketing

    We use the same bank marketing dataset as in the logistic regression notebook. The goal is
    to predict whether a client will subscribe to a term deposit (`y = 1`) or not (`y = 0`).

    This allows direct comparison between logistic regression and tree-based methods on the
    same prediction task.
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
    ## Feature Selection and Encoding

    We select the same 7 features as in logistic regression:

    **Numeric features** (4): `age`, `balance`, `duration`, `campaign`

    **Categorical features** (3): `marital`, `education`, `housing`

    For scikit-learn's tree models, categorical features must be encoded as numeric values.
    We use **one-hot encoding** (dummy variables) to convert categorical features into binary
    columns, similar to what statsmodels does automatically with formulas.
    """)
    return


@app.cell
def _(np, pl, raw_data):
    # Define features to use
    numeric_features = ["age", "balance", "duration", "campaign"]
    categorical_features = ["marital", "education", "housing"]

    # Stratified sample of 10,000 rows
    np.random.seed(42)
    _sample_size = 10000

    # Split by target, sample proportionally
    _class_0 = raw_data.filter(pl.col("y") == 0)
    _class_1 = raw_data.filter(pl.col("y") == 1)

    _n_class_1 = int(_sample_size * len(_class_1) / len(raw_data))
    _n_class_0 = _sample_size - _n_class_1

    _sample_0 = _class_0.sample(
        n=_n_class_0,
        seed=42,
    )
    _sample_1 = _class_1.sample(
        n=_n_class_1,
        seed=42,
    )

    model_data = pl.concat([_sample_0, _sample_1]).sample(
        fraction=1.0,
        seed=42,
    )

    print(f"Sampled data shape: {model_data.shape[0]:,} rows")
    print(f"Target distribution in sample:")
    print(model_data.group_by("y").len().sort("y"))
    return categorical_features, model_data, numeric_features


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Train/Test Split

    We split the data into training (80%) and test (20%) sets with stratification to maintain
    the same class proportions in both sets. We also prepare the feature matrices for
    scikit-learn, which requires numeric arrays.
    """)
    return


@app.cell
def _(categorical_features, model_data, np, numeric_features, pl):
    # Train/test split (stratified)
    np.random.seed(42)

    _class_0 = model_data.filter(pl.col("y") == 0)
    _class_1 = model_data.filter(pl.col("y") == 1)

    _n_train_0 = int(0.8 * len(_class_0))
    _n_train_1 = int(0.8 * len(_class_1))

    _train_0 = _class_0.sample(n=_n_train_0, seed=42)
    _train_1 = _class_1.sample(n=_n_train_1, seed=42)

    _test_0 = _class_0.join(_train_0, on="id", how="anti")
    _test_1 = _class_1.join(_train_1, on="id", how="anti")

    train_data = pl.concat([_train_0, _train_1]).sample(fraction=1.0, seed=42)
    test_data = pl.concat([_test_0, _test_1]).sample(fraction=1.0, seed=42)

    # Create dummy variables for categorical features
    _train_numeric = train_data.select(numeric_features).to_numpy()
    _train_dummies = train_data.select(categorical_features).to_dummies()
    _test_numeric = test_data.select(numeric_features).to_numpy()
    _test_dummies = test_data.select(categorical_features).to_dummies()

    # Combine numeric and dummy features
    X_train = np.hstack([_train_numeric, _train_dummies.to_numpy()])
    X_test = np.hstack([_test_numeric, _test_dummies.to_numpy()])
    y_train = train_data["y"].to_numpy()
    y_test = test_data["y"].to_numpy()

    # Store feature names for visualization
    feature_names = numeric_features + _train_dummies.columns

    print(f"Training set: {len(X_train):,} samples, {X_train.shape[1]} features")
    print(f"Test set: {len(X_test):,} samples")
    print(f"\nFeature names: {feature_names}")
    return X_test, X_train, feature_names, test_data, train_data, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Decision Tree Parameters

    Key parameters for `DecisionTreeClassifier`:

    | Parameter | Description | Default |
    |-----------|-------------|---------|
    | `max_depth` | Maximum depth of the tree | None (fully grown) |
    | `criterion` | Impurity measure: "gini" or "entropy" | "gini" |
    | `min_samples_split` | Minimum samples required to split a node | 2 |
    | `min_samples_leaf` | Minimum samples required in a leaf node | 1 |

    **Important:** Without constraints, a decision tree will grow until each leaf is pure,
    perfectly memorizing the training data. This leads to **overfitting**.
    """)
    return


@app.cell
def _(DecisionTreeClassifier, X_train, y_train):
    # Fit a shallow decision tree for visualization
    dt_shallow = DecisionTreeClassifier(
        max_depth=3,
        criterion="gini",
        random_state=42,
    )
    dt_shallow.fit(X_train, y_train)

    print(f"Tree depth: {dt_shallow.get_depth()}")
    print(f"Number of leaves: {dt_shallow.get_n_leaves()}")
    return (dt_shallow,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Reading a Decision Tree Diagram

    Each node in the tree shows:
    - **Feature and threshold**: The split condition (e.g., "duration <= 225.5")
    - **Gini**: The impurity at that node (lower = more pure)
    - **samples**: Number of training samples reaching this node
    - **value**: Count of each class [class 0, class 1]
    - **class**: The predicted class if this were a leaf (majority class)

    The color intensity indicates the dominant class: darker blue = more class 0 (No),
    darker orange = more class 1 (Yes).
    """)
    return


@app.cell
def _(dt_shallow, feature_names, plot_tree, plt):
    # Visualize the decision tree
    _fig, _ax = plt.subplots(figsize=(16, 8))

    plot_tree(
        decision_tree=dt_shallow,
        feature_names=feature_names,
        class_names=["No", "Yes"],
        filled=True,
        rounded=True,
        ax=_ax,
        fontsize=9,
    )

    _ax.set_title("Decision Tree (max_depth=3)")

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Feature Importance

    Unlike logistic regression which provides coefficients, decision trees provide **feature
    importance** scores. These are calculated as the **mean decrease in impurity** (MDI):

    For each feature, sum the impurity reduction across all splits using that feature,
    weighted by the number of samples reaching each node. The importances are then normalized
    to sum to 1.

    **Interpretation:**
    - Higher importance = feature contributes more to reducing impurity
    - Importance does not indicate direction of effect (positive/negative)
    - Correlated features may have split importance between them
    """)
    return


@app.cell
def _(dt_shallow, feature_names, np, plt, sns):
    # Feature importance from shallow tree
    _importances = dt_shallow.feature_importances_
    _indices = np.argsort(_importances)[::-1]

    _fig, _ax = plt.subplots(figsize=(8, 5))

    sns.barplot(
        x=_importances[_indices],
        y=[feature_names[_i] for _i in _indices],
        hue=[feature_names[_i] for _i in _indices],
        ax=_ax,
        palette="Blues_r",
        edgecolor="black",
        legend=False,
    )

    _ax.set_xlabel("Feature Importance (Mean Decrease in Impurity)")
    _ax.set_ylabel("Feature")
    _ax.set_title("Decision Tree Feature Importance (max_depth=3)")

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Overfitting: Shallow vs. Deep Trees

    Decision trees are prone to **overfitting** when allowed to grow too deep:
    - A fully grown tree can achieve 100% training accuracy by memorizing the data
    - But it will perform poorly on new, unseen data

    We demonstrate this by comparing a shallow tree (max_depth=3) with a deep tree
    (max_depth=20) on both training and test data.
    """)
    return


@app.cell
def _(DecisionTreeClassifier, X_test, X_train, y_test, y_train):
    # Compare shallow vs deep tree
    _dt_deep = DecisionTreeClassifier(
        max_depth=20,
        random_state=42,
    )
    _dt_deep.fit(X_train, y_train)

    _dt_shallow = DecisionTreeClassifier(
        max_depth=3,
        random_state=42,
    )
    _dt_shallow.fit(X_train, y_train)

    # Calculate accuracies
    _shallow_train_acc = _dt_shallow.score(X_train, y_train)
    _shallow_test_acc = _dt_shallow.score(X_test, y_test)
    _deep_train_acc = _dt_deep.score(X_train, y_train)
    _deep_test_acc = _dt_deep.score(X_test, y_test)

    print("Overfitting Demonstration: Shallow vs Deep Tree")
    print("-" * 50)
    print(f"{'Model':<20} {'Train Accuracy':<18} {'Test Accuracy':<18}")
    print("-" * 50)
    print(f"{'Shallow (depth=3)':<20} {_shallow_train_acc:<18.4f} {_shallow_test_acc:<18.4f}")
    print(f"{'Deep (depth=20)':<20} {_deep_train_acc:<18.4f} {_deep_test_acc:<18.4f}")
    print("-" * 50)
    print("\nNote: The deep tree has higher training accuracy but lower test accuracy,")
    print("indicating overfitting to the training data.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Ensemble Methods: Bootstrap Aggregating (Bagging)

    **Ensemble methods** combine multiple models to produce better predictions than any
    single model alone. The key insight is that averaging predictions from diverse models
    reduces variance without increasing bias.

    **Bootstrap Aggregating (Bagging):**
    1. Create multiple training sets by sampling **with replacement** from the original data
    2. Train a separate model on each bootstrap sample
    3. Combine predictions by voting (classification) or averaging (regression)

    Each bootstrap sample contains about 63% unique observations on average, leaving ~37%
    "out-of-bag" (OOB) samples that can be used for validation.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Random Forests

    A **Random Forest** is an ensemble of decision trees with two sources of randomness:

    1. **Bootstrap samples**: Each tree is trained on a different bootstrap sample (bagging)
    2. **Random feature subsets**: At each split, only a random subset of features is considered

    The feature randomization decorrelates the trees, making the ensemble more robust.
    If one feature dominates (like `duration` in our data), standard bagging would create
    very similar trees. Random feature selection forces trees to explore other features.

    **Key hyperparameters:**
    - `n_estimators`: Number of trees (more is generally better, but with diminishing returns)
    - `max_features`: Number of features to consider at each split (default: sqrt(n_features))
    - `oob_score`: If True, use out-of-bag samples to estimate generalization error
    """)
    return


@app.cell
def _(RandomForestClassifier, X_train, y_train):
    # Fit random forest
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        max_features="sqrt",
        oob_score=True,
        random_state=42,
        n_jobs=-1,
    )
    rf_model.fit(X_train, y_train)

    print(f"Random Forest trained with {rf_model.n_estimators} trees")
    print(f"Out-of-Bag Score: {rf_model.oob_score_:.4f}")
    print(f"\nNote: OOB score provides an unbiased estimate of test accuracy")
    print("without needing a separate validation set.")
    return (rf_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Aggregated Feature Importance

    Random forest feature importance is more **stable** than single tree importance because
    it averages across many trees. This reduces the variance that comes from any single
    tree's particular splits.

    The interpretation remains the same: mean decrease in impurity across all trees,
    averaged over the forest.
    """)
    return


@app.cell
def _(dt_shallow, feature_names, np, plt, rf_model):
    # Compare feature importance: single tree vs random forest
    _fig, _axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(12, 5),
    )

    # Single tree importance
    _tree_imp = dt_shallow.feature_importances_
    _tree_idx = np.argsort(_tree_imp)[::-1]

    _axes[0].barh(
        y=[feature_names[_i] for _i in _tree_idx],
        width=_tree_imp[_tree_idx],
        color="steelblue",
        edgecolor="black",
    )
    _axes[0].set_xlabel("Feature Importance")
    _axes[0].set_title("Single Decision Tree (depth=3)")
    _axes[0].invert_yaxis()

    # Random forest importance
    _rf_imp = rf_model.feature_importances_
    _rf_idx = np.argsort(_rf_imp)[::-1]

    _axes[1].barh(
        y=[feature_names[_i] for _i in _rf_idx],
        width=_rf_imp[_rf_idx],
        color="coral",
        edgecolor="black",
    )
    _axes[1].set_xlabel("Feature Importance")
    _axes[1].set_title("Random Forest (100 trees)")
    _axes[1].invert_yaxis()

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Permutation Importance vs. Mean Decrease in Impurity (MDI)

    The feature importance shown above uses **Mean Decrease in Impurity (MDI)**, which has
    some limitations:

    **MDI Limitations:**
    - Biased toward high-cardinality features (features with many unique values)
    - Computed on training data, so can reflect overfitting
    - Does not account for feature correlations well

    **Permutation Importance** is an alternative that addresses these issues:
    - **Model-agnostic**: Works with any model, not just trees
    - **Computed on held-out test data**: Measures actual predictive contribution
    - **Less biased**: Doesn't favor high-cardinality features
    - **Intuitive interpretation**: Measures how much performance drops when a feature is shuffled

    The algorithm:
    1. Compute baseline score on test data
    2. For each feature, randomly shuffle its values and compute the new score
    3. Importance = baseline score - shuffled score
    4. Repeat multiple times to get robust estimates
    """)
    return


@app.cell
def _(X_test, feature_names, np, permutation_importance, plt, rf_model, y_test):
    # Compute permutation importance on test data
    _perm_result = permutation_importance(
        estimator=rf_model,
        X=X_test,
        y=y_test,
        n_repeats=10,
        random_state=42,
        n_jobs=-1,
    )

    # Get MDI importance for comparison
    _mdi_imp = rf_model.feature_importances_

    # Sort by permutation importance
    _perm_idx = np.argsort(_perm_result.importances_mean)[::-1]

    # Create comparison plot
    _fig, _axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(12, 5),
    )

    # MDI importance (same order as permutation for comparison)
    _axes[0].barh(
        y=[feature_names[_i] for _i in _perm_idx],
        width=_mdi_imp[_perm_idx],
        color="coral",
        edgecolor="black",
    )
    _axes[0].set_xlabel("Feature Importance")
    _axes[0].set_title("MDI (Mean Decrease in Impurity)")
    _axes[0].invert_yaxis()

    # Permutation importance with error bars
    _axes[1].barh(
        y=[feature_names[_i] for _i in _perm_idx],
        width=_perm_result.importances_mean[_perm_idx],
        xerr=_perm_result.importances_std[_perm_idx],
        color="seagreen",
        edgecolor="black",
        capsize=3,
    )
    _axes[1].set_xlabel("Mean Accuracy Decrease")
    _axes[1].set_title("Permutation Importance (on test data)")
    _axes[1].invert_yaxis()

    plt.suptitle("Comparing Feature Importance Methods", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.show()

    print("Permutation importance shows how much model accuracy drops when each feature")
    print("is randomly shuffled. Error bars show variability across 10 repetitions.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Hyperparameter Guidelines

    | Parameter | Guideline | Notes |
    |-----------|-----------|-------|
    | `n_estimators` | Start with 100, increase if needed | More trees rarely hurt, but slow down training |
    | `max_depth` | 10-20 for most problems | Deeper trees capture more complexity but risk overfitting |
    | `max_features` | "sqrt" for classification | Forces tree diversity; try "log2" as alternative |
    | `min_samples_leaf` | 1-5 for small data, higher for large | Prevents very small leaf nodes |

    **Tuning strategy:**
    1. Start with defaults and evaluate OOB score
    2. Increase `n_estimators` until OOB score plateaus
    3. Tune `max_depth` and `max_features` via cross-validation
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model Evaluation

    We evaluate both the single decision tree and random forest using the same metrics
    as in logistic regression:
    - **Confusion matrix**: Shows true/false positives and negatives
    - **Accuracy, Precision, Recall, F1**: Standard classification metrics
    - **ROC curve and AUC**: Threshold-independent performance measure
    """)
    return


@app.cell
def _(
    X_test,
    confusion_matrix,
    dt_shallow,
    plt,
    rf_model,
    sns,
    y_test,
):
    # Predictions
    _dt_pred = dt_shallow.predict(X_test)
    _rf_pred = rf_model.predict(X_test)

    # Confusion matrices
    _dt_cm = confusion_matrix(y_true=y_test, y_pred=_dt_pred)
    _rf_cm = confusion_matrix(y_true=y_test, y_pred=_rf_pred)

    _fig, _axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(8, 4),
    )

    # Decision tree confusion matrix
    sns.heatmap(
        _dt_cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=_axes[0],
        xticklabels=["No (0)", "Yes (1)"],
        yticklabels=["No (0)", "Yes (1)"],
        cbar=False,
        linewidths=0.1,
        linecolor="k",
    )
    _axes[0].set_xlabel("Predicted")
    _axes[0].set_ylabel("Actual")
    _axes[0].set_title("Decision Tree")

    # Random forest confusion matrix
    sns.heatmap(
        _rf_cm,
        annot=True,
        fmt="d",
        cmap="Oranges",
        ax=_axes[1],
        xticklabels=["No (0)", "Yes (1)"],
        yticklabels=["No (0)", "Yes (1)"],
        cbar=False,
        linewidths=0.1,
        linecolor="k",
    )
    _axes[1].set_xlabel("Predicted")
    _axes[1].set_ylabel("Actual")
    _axes[1].set_title("Random Forest")

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Understanding Classification Metrics

    Before comparing the models, let's understand what each metric measures and when to prioritize each.

    **Precision** = TP / (TP + FP)
    - "Of all predicted positives, how many are actually positive?"
    - Prioritize when **false positives are costly**
    - Example: Spam filter (flagging legitimate email as spam is annoying)

    **Recall (Sensitivity)** = TP / (TP + FN)
    - "Of all actual positives, how many did we correctly identify?"
    - Prioritize when **false negatives are costly**
    - Example: Disease screening (missing a disease is dangerous)

    **F1 Score** = 2 × (Precision × Recall) / (Precision + Recall)
    - Harmonic mean of precision and recall
    - Useful when classes are imbalanced
    - Balances the trade-off between precision and recall

    **AUC (Area Under ROC Curve)**
    - Threshold-independent measure of ranking ability
    - Probability that a random positive ranks higher than a random negative
    - Useful for comparing models across all possible decision thresholds

    | Metric | Prioritize when... | Example use case |
    |--------|-------------------|------------------|
    | Precision | False positives are costly | Spam filtering, fraud alerts |
    | Recall | False negatives are costly | Disease screening, fraud detection |
    | F1 | Need balance between precision/recall | Imbalanced classification |
    | AUC | Comparing models across thresholds | Model selection |
    """)
    return


@app.cell
def _(
    X_test,
    accuracy_score,
    dt_shallow,
    f1_score,
    pl,
    precision_score,
    recall_score,
    rf_model,
    y_test,
):
    # Calculate metrics for both models
    _dt_pred = dt_shallow.predict(X_test)
    _rf_pred = rf_model.predict(X_test)

    metrics_comparison = pl.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
        "Decision Tree": [
            accuracy_score(y_true=y_test, y_pred=_dt_pred),
            precision_score(y_true=y_test, y_pred=_dt_pred),
            recall_score(y_true=y_test, y_pred=_dt_pred),
            f1_score(y_true=y_test, y_pred=_dt_pred),
        ],
        "Random Forest": [
            accuracy_score(y_true=y_test, y_pred=_rf_pred),
            precision_score(y_true=y_test, y_pred=_rf_pred),
            recall_score(y_true=y_test, y_pred=_rf_pred),
            f1_score(y_true=y_test, y_pred=_rf_pred),
        ],
    })

    print("Model Comparison: Decision Tree vs Random Forest")
    print("-" * 55)
    print(metrics_comparison)
    return (metrics_comparison,)


@app.cell
def _(X_test, dt_shallow, plt, rf_model, roc_auc_score, roc_curve, y_test):
    # ROC curves for both models
    _dt_prob = dt_shallow.predict_proba(X_test)[:, 1]
    _rf_prob = rf_model.predict_proba(X_test)[:, 1]

    _dt_fpr, _dt_tpr, _ = roc_curve(y_true=y_test, y_score=_dt_prob)
    _rf_fpr, _rf_tpr, _ = roc_curve(y_true=y_test, y_score=_rf_prob)

    _dt_auc = roc_auc_score(y_true=y_test, y_score=_dt_prob)
    _rf_auc = roc_auc_score(y_true=y_test, y_score=_rf_prob)

    _fig, _ax = plt.subplots(figsize=(6, 5))

    _ax.plot(
        _dt_fpr,
        _dt_tpr,
        color="steelblue",
        linewidth=2,
        label=f"Decision Tree (AUC = {_dt_auc:.3f})",
    )
    _ax.plot(
        _rf_fpr,
        _rf_tpr,
        color="coral",
        linewidth=2,
        label=f"Random Forest (AUC = {_rf_auc:.3f})",
    )

    # Diagonal reference line
    _ax.plot(
        [0, 1],
        [0, 1],
        color="gray",
        linestyle="--",
        label="Random classifier (AUC = 0.5)",
    )

    _ax.set_xlabel("False Positive Rate")
    _ax.set_ylabel("True Positive Rate (Recall)")
    _ax.set_title("ROC Curves: Decision Tree vs Random Forest")
    _ax.legend(loc="lower right")
    _ax.set_xlim(-0.02, 1.02)
    _ax.set_ylim(-0.02, 1.02)

    plt.tight_layout()
    plt.show()

    print(f"Decision Tree AUC: {_dt_auc:.4f}")
    print(f"Random Forest AUC: {_rf_auc:.4f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Comparison Summary

    | Aspect | Logistic Regression | Decision Tree | Random Forest |
    |--------|---------------------|---------------|---------------|
    | **Model type** | Linear (in log-odds) | Non-linear, rule-based | Ensemble of trees |
    | **Interpretability** | Coefficients, odds ratios | Visual tree, feature importance | Feature importance only |
    | **Handles non-linearity** | No (needs feature engineering) | Yes, automatically | Yes, automatically |
    | **Handles interactions** | No (needs explicit terms) | Yes, automatically | Yes, automatically |
    | **Overfitting risk** | Low | High (if unconstrained) | Low (due to averaging) |
    | **Training speed** | Fast | Fast | Slower (many trees) |
    | **Prediction speed** | Fast | Fast | Slower (many trees) |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Takeaways

    **Decision Trees:**
    - Intuitive, interpretable models that recursively partition the feature space
    - Prone to overfitting; require constraints (max_depth, min_samples_leaf)
    - Feature importance based on impurity reduction, not direction of effect
    - Can capture non-linear relationships and interactions automatically

    **Random Forests:**
    - Ensemble of decision trees with bootstrap sampling and random feature selection
    - More robust and accurate than single trees due to averaging
    - Out-of-bag (OOB) score provides free validation estimate
    - Feature importance is more stable than single tree importance

    **When to use each:**
    - **Logistic Regression**: When you need interpretable coefficients, odds ratios, and p-values
    - **Decision Tree**: When you need a simple, interpretable model with visual representation
    - **Random Forest**: When prediction accuracy is the priority and you have sufficient data
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
