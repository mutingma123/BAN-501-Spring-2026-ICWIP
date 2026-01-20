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
    from sklearn.model_selection import cross_val_score
    from sklearn.tree import DecisionTreeClassifier
    from xgboost import XGBClassifier
    import optuna

    sns.set_style("whitegrid")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return (
        DecisionTreeClassifier,
        RandomForestClassifier,
        XGBClassifier,
        accuracy_score,
        confusion_matrix,
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
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # XGBoost with Optuna Hyperparameter Optimization

    This notebook introduces **XGBoost** (eXtreme Gradient Boosting), a powerful gradient boosting
    algorithm, and demonstrates how to tune its hyperparameters using **Optuna**, a modern
    Bayesian optimization framework.

    ## From Bagging to Boosting

    In the previous notebook, we explored **Random Forests**, which use **bagging** (bootstrap
    aggregating) to create diverse trees that vote independently. Boosting takes a different
    approach: trees are built **sequentially**, with each new tree correcting the errors of
    the previous ensemble.

    | Approach | Bagging (Random Forest) | Boosting (XGBoost) |
    |----------|------------------------|-------------------|
    | Tree building | Parallel, independent | Sequential, dependent |
    | Focus | Reduce variance through averaging | Reduce bias by correcting errors |
    | Tree depth | Typically deeper trees | Typically shallow trees (weak learners) |
    | Learning | Each tree sees bootstrap sample | Each tree fits residual errors |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Gradient Boosting Concept

    Gradient boosting builds an **additive model** where each new tree fits the negative gradient
    (residuals) of the loss function:

    $$F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)$$

    Where:
    - $F_m(x)$ is the ensemble prediction after $m$ trees
    - $h_m(x)$ is the new tree fitted to the residuals
    - $\eta$ is the learning rate (shrinkage factor)

    **Key insight**: Each tree is a "weak learner" that makes small corrections. The learning
    rate $\eta$ controls how much each tree contributes, preventing overfitting.

    **Example**: If the first tree predicts 0.3 for a sample with true label 1, the residual
    is 0.7. The next tree tries to predict this 0.7 residual, and so on.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## XGBoost: Regularized Gradient Boosting

    XGBoost improves on basic gradient boosting with several enhancements:

    **1. Regularization**: The objective function includes L1 and L2 penalties on leaf weights:
    $$\mathcal{L} = \sum_{i} l(y_i, \hat{y}_i) + \sum_{k} \Omega(f_k)$$
    where $\Omega(f) = \gamma T + \frac{1}{2}\lambda \|w\|^2$

    **2. Tree pruning**: Uses a "max_depth" parameter and prunes trees that don't improve
    the objective, rather than growing to maximum depth then pruning.

    **3. Handling missing values**: XGBoost learns the optimal direction to send missing
    values at each split during training.

    **4. Column subsampling**: Like random forests, XGBoost can sample features at each
    tree or each split level, reducing correlation between trees.

    **5. Parallel computation**: While trees are built sequentially, the split finding
    within each tree is parallelized.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Dataset: Bank Marketing

    We use the same bank marketing dataset and preprocessing as in the decision trees notebook.
    This allows direct comparison between models.
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
    ## Feature Selection and Sampling

    We use the same 7 features and stratified sample of 10,000 rows:

    **Numeric features** (4): `age`, `balance`, `duration`, `campaign`

    **Categorical features** (3): `marital`, `education`, `housing`
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

    We split the data into training (80%) and test (20%) sets with stratification.
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
    ## Baseline XGBoost Model

    We first train an XGBoost model with default parameters to establish a baseline
    for comparison after hyperparameter tuning.
    """)
    return


@app.cell
def _(XGBClassifier, X_train, y_train):
    # Train baseline XGBoost with default parameters
    xgb_baseline = XGBClassifier(
        random_state=42,
        eval_metric="logloss",
    )
    xgb_baseline.fit(X_train, y_train)

    print("Baseline XGBoost trained with default parameters:")
    print(f"  n_estimators: {xgb_baseline.n_estimators}")
    print(f"  max_depth: {xgb_baseline.max_depth}")
    print(f"  learning_rate: {xgb_baseline.learning_rate}")
    return (xgb_baseline,)


@app.cell
def _(
    X_test,
    accuracy_score,
    f1_score,
    pl,
    precision_score,
    recall_score,
    roc_auc_score,
    xgb_baseline,
    y_test,
):
    # Evaluate baseline model
    _baseline_pred = xgb_baseline.predict(X_test)
    _baseline_prob = xgb_baseline.predict_proba(X_test)[:, 1]

    baseline_metrics = pl.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "AUC"],
        "Baseline XGBoost": [
            accuracy_score(y_true=y_test, y_pred=_baseline_pred),
            precision_score(y_true=y_test, y_pred=_baseline_pred),
            recall_score(y_true=y_test, y_pred=_baseline_pred),
            f1_score(y_true=y_test, y_pred=_baseline_pred),
            roc_auc_score(y_true=y_test, y_score=_baseline_prob),
        ],
    })

    print("Baseline XGBoost Performance:")
    print("-" * 40)
    print(baseline_metrics)
    return (baseline_metrics,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## XGBoost Hyperparameters

    XGBoost has many hyperparameters that can significantly affect model performance.
    Here are the key parameters we will tune:

    | Parameter | Description | Range |
    |-----------|-------------|-------|
    | `n_estimators` | Number of boosting rounds (trees) | 50-300 |
    | `max_depth` | Maximum tree depth | 3-10 |
    | `learning_rate` | Step size shrinkage (eta) | 0.01-0.3 |
    | `min_child_weight` | Minimum sum of instance weight in a child | 1-10 |
    | `subsample` | Row sampling ratio per tree | 0.5-1.0 |
    | `colsample_bytree` | Column sampling ratio per tree | 0.5-1.0 |
    | `gamma` | Minimum loss reduction for split | 0-5 |
    | `reg_alpha` | L1 regularization (lasso) | 0-1 |
    | `reg_lambda` | L2 regularization (ridge) | 0-1 |

    **Trade-offs:**
    - More trees (`n_estimators`) with lower `learning_rate` often performs better but trains slower
    - Lower `max_depth` and higher regularization prevent overfitting
    - `subsample` and `colsample_bytree` add stochasticity to reduce overfitting
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Introduction to Optuna

    **Optuna** is a hyperparameter optimization framework that uses **Bayesian optimization**
    to efficiently search the parameter space.

    **Key concepts:**
    - **Study**: A single optimization session
    - **Trial**: One evaluation of the objective function with a specific parameter set
    - **TPE Sampler**: Tree-structured Parzen Estimator, the default algorithm that models
      the probability of good vs bad parameters

    **How TPE works:**
    1. Run some initial random trials
    2. Build probability models: $p(x|y < y^*)$ and $p(x|y \geq y^*)$
    3. Sample parameters that maximize the ratio $p(x|y < y^*) / p(x|y \geq y^*)$
    4. This focuses search on promising regions while still exploring

    **Advantages over grid search:**
    - Explores promising regions more densely
    - Handles continuous, discrete, and conditional parameters
    - Can be stopped early if results are satisfactory
    - Provides insights into parameter importance
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Defining the Objective Function

    The objective function defines what Optuna optimizes. It:
    1. Receives a `trial` object with suggested parameters
    2. Trains and evaluates a model
    3. Returns a score to maximize (or minimize)

    We use **3-fold cross-validation AUC** as our objective to get a robust estimate
    of model performance and avoid overfitting to a single train/test split.
    """)
    return


@app.cell
def _(XGBClassifier, X_train, cross_val_score, y_train):
    def objective(trial):
        """Optuna objective function for XGBoost hyperparameter optimization."""

        # Suggest hyperparameters
        _params = {
            "n_estimators": trial.suggest_int(
                name="n_estimators",
                low=50,
                high=300,
                step=50,
            ),
            "max_depth": trial.suggest_int(
                name="max_depth",
                low=3,
                high=10,
            ),
            "learning_rate": trial.suggest_float(
                name="learning_rate",
                low=0.01,
                high=0.3,
                log=True,
            ),
            "min_child_weight": trial.suggest_int(
                name="min_child_weight",
                low=1,
                high=10,
            ),
            "subsample": trial.suggest_float(
                name="subsample",
                low=0.5,
                high=1.0,
            ),
            "colsample_bytree": trial.suggest_float(
                name="colsample_bytree",
                low=0.5,
                high=1.0,
            ),
            "gamma": trial.suggest_float(
                name="gamma",
                low=0,
                high=5,
            ),
            "reg_alpha": trial.suggest_float(
                name="reg_alpha",
                low=0,
                high=1,
            ),
            "reg_lambda": trial.suggest_float(
                name="reg_lambda",
                low=0,
                high=1,
            ),
            "random_state": 42,
            "eval_metric": "logloss",
        }

        # Create and evaluate model with cross-validation
        _model = XGBClassifier(**_params)

        _scores = cross_val_score(
            estimator=_model,
            X=X_train,
            y=y_train,
            cv=3,
            scoring="roc_auc",
            n_jobs=-1,
        )

        return _scores.mean()
    return (objective,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Running the Optimization Study

    We run 50 trials to find the best hyperparameters. Each trial:
    1. Samples a new set of hyperparameters using TPE
    2. Trains an XGBoost model with 3-fold cross-validation
    3. Records the mean AUC score

    The study learns from previous trials to focus on promising parameter regions.
    """)
    return


@app.cell
def _(objective, optuna):
    # Create and run the Optuna study
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    study.optimize(
        func=objective,
        n_trials=50,
        show_progress_bar=True,
    )

    print(f"\nBest trial:")
    print(f"  AUC: {study.best_trial.value:.4f}")
    print(f"\nBest hyperparameters:")
    for _key, _value in study.best_trial.params.items():
        if isinstance(_value, float):
            print(f"  {_key}: {_value:.4f}")
        else:
            print(f"  {_key}: {_value}")
    return (study,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Optimization History

    This plot shows how the optimization progressed over trials:
    - **Blue dots**: Individual trial AUC scores
    - **Red line**: Best AUC found so far

    A good optimization shows the best line improving over time, indicating that
    Optuna is finding better parameter combinations.
    """)
    return


@app.cell
def _(np, plt, study):
    # Plot optimization history
    _fig, _ax = plt.subplots(figsize=(10, 5))

    _trials = study.trials
    _trial_numbers = [_t.number for _t in _trials]
    _trial_values = [_t.value for _t in _trials]

    # Calculate running best
    _best_so_far = np.maximum.accumulate(_trial_values)

    # Plot individual trials
    _ax.scatter(
        _trial_numbers,
        _trial_values,
        alpha=0.6,
        color="steelblue",
        s=40,
        label="Trial AUC",
        edgecolor="white",
        linewidth=0.5,
    )

    # Plot best so far line
    _ax.plot(
        _trial_numbers,
        _best_so_far,
        color="coral",
        linewidth=2,
        label="Best AUC so far",
    )

    _ax.set_xlabel("Trial Number")
    _ax.set_ylabel("Cross-Validation AUC")
    _ax.set_title("Optuna Optimization History")
    _ax.legend(loc="lower right")
    _ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"Initial AUC (Trial 0): {_trial_values[0]:.4f}")
    print(f"Final Best AUC: {max(_trial_values):.4f}")
    print(f"Improvement: {max(_trial_values) - _trial_values[0]:.4f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Hyperparameter Importance

    Optuna can estimate which hyperparameters have the most impact on the objective
    using **fANOVA** (functional ANOVA). This helps understand:
    - Which parameters are worth tuning carefully
    - Which parameters have little effect and can use defaults

    Higher importance means the parameter has more influence on the AUC score.
    """)
    return


@app.cell
def _(optuna, plt, study):
    # Get parameter importances using fANOVA
    _importances = optuna.importance.get_param_importances(study)

    _fig, _ax = plt.subplots(figsize=(8, 5))

    _params = list(_importances.keys())
    _values = list(_importances.values())

    _ax.barh(
        y=_params[::-1],
        width=_values[::-1],
        color="steelblue",
        edgecolor="black",
    )

    _ax.set_xlabel("Importance (fANOVA)")
    _ax.set_title("Hyperparameter Importance")

    plt.tight_layout()
    plt.show()

    print("Hyperparameter Importance (fANOVA):")
    print("-" * 40)
    for _param, _imp in _importances.items():
        print(f"  {_param}: {_imp:.4f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Training the Optimized Model

    Now we train a final XGBoost model using the best hyperparameters found by Optuna.
    This model is trained on the full training set (not cross-validated) for deployment.
    """)
    return


@app.cell
def _(XGBClassifier, X_train, study, y_train):
    # Train optimized model with best parameters
    _best_params = study.best_trial.params.copy()
    _best_params["random_state"] = 42
    _best_params["eval_metric"] = "logloss"

    xgb_optimized = XGBClassifier(**_best_params)
    xgb_optimized.fit(X_train, y_train)

    print("Optimized XGBoost trained with best parameters:")
    for _key, _value in study.best_trial.params.items():
        if isinstance(_value, float):
            print(f"  {_key}: {_value:.4f}")
        else:
            print(f"  {_key}: {_value}")
    return (xgb_optimized,)


@app.cell
def _(
    X_test,
    accuracy_score,
    baseline_metrics,
    f1_score,
    pl,
    precision_score,
    recall_score,
    roc_auc_score,
    xgb_optimized,
    y_test,
):
    # Evaluate optimized model
    _optimized_pred = xgb_optimized.predict(X_test)
    _optimized_prob = xgb_optimized.predict_proba(X_test)[:, 1]

    optimized_metrics = pl.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "AUC"],
        "Baseline XGBoost": baseline_metrics["Baseline XGBoost"],
        "Optimized XGBoost": [
            accuracy_score(y_true=y_test, y_pred=_optimized_pred),
            precision_score(y_true=y_test, y_pred=_optimized_pred),
            recall_score(y_true=y_test, y_pred=_optimized_pred),
            f1_score(y_true=y_test, y_pred=_optimized_pred),
            roc_auc_score(y_true=y_test, y_score=_optimized_prob),
        ],
    })

    print("Baseline vs Optimized XGBoost:")
    print("-" * 55)
    print(optimized_metrics)
    return (optimized_metrics,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## XGBoost Feature Importance

    XGBoost provides feature importance based on the **gain** each feature contributes
    across all trees. This measures how much each feature improves the objective when
    used for splitting.
    """)
    return


@app.cell
def _(feature_names, np, plt, xgb_optimized):
    # Feature importance from optimized model
    _importances = xgb_optimized.feature_importances_
    _indices = np.argsort(_importances)[::-1]

    _fig, _ax = plt.subplots(figsize=(8, 5))

    _ax.barh(
        y=[feature_names[_i] for _i in _indices][::-1],
        width=_importances[_indices][::-1],
        color="coral",
        edgecolor="black",
    )

    _ax.set_xlabel("Feature Importance (Gain)")
    _ax.set_title("XGBoost Feature Importance (Optimized Model)")

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Confusion Matrices: Baseline vs Optimized

    Comparing confusion matrices shows how hyperparameter tuning affects the
    classification decisions at the default threshold of 0.5.
    """)
    return


@app.cell
def _(
    X_test,
    confusion_matrix,
    plt,
    sns,
    xgb_baseline,
    xgb_optimized,
    y_test,
):
    # Confusion matrices for baseline and optimized
    _baseline_pred = xgb_baseline.predict(X_test)
    _optimized_pred = xgb_optimized.predict(X_test)

    _baseline_cm = confusion_matrix(y_true=y_test, y_pred=_baseline_pred)
    _optimized_cm = confusion_matrix(y_true=y_test, y_pred=_optimized_pred)

    _fig, _axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(8, 4),
    )

    # Baseline confusion matrix
    sns.heatmap(
        _baseline_cm,
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
    _axes[0].set_title("Baseline XGBoost")

    # Optimized confusion matrix
    sns.heatmap(
        _optimized_cm,
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
    _axes[1].set_title("Optimized XGBoost")

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Comparison with Decision Tree and Random Forest

    To put XGBoost's performance in context, we train Decision Tree and Random Forest
    models with the same data and compare all four models (including baseline and
    optimized XGBoost).
    """)
    return


@app.cell
def _(DecisionTreeClassifier, RandomForestClassifier, X_train, y_train):
    # Train Decision Tree and Random Forest for comparison
    dt_model = DecisionTreeClassifier(
        max_depth=5,
        random_state=42,
    )
    dt_model.fit(X_train, y_train)

    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    rf_model.fit(X_train, y_train)

    print("Comparison models trained:")
    print(f"  Decision Tree: max_depth={dt_model.max_depth}")
    print(f"  Random Forest: n_estimators={rf_model.n_estimators}, max_depth={rf_model.max_depth}")
    return dt_model, rf_model


@app.cell
def _(
    X_test,
    accuracy_score,
    dt_model,
    f1_score,
    pl,
    precision_score,
    recall_score,
    rf_model,
    roc_auc_score,
    xgb_baseline,
    xgb_optimized,
    y_test,
):
    # Calculate metrics for all models
    _models = {
        "Decision Tree": dt_model,
        "Random Forest": rf_model,
        "XGBoost (Baseline)": xgb_baseline,
        "XGBoost (Optimized)": xgb_optimized,
    }

    _results = {"Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "AUC"]}

    for _name, _model in _models.items():
        _pred = _model.predict(X_test)
        _prob = _model.predict_proba(X_test)[:, 1]

        _results[_name] = [
            accuracy_score(y_true=y_test, y_pred=_pred),
            precision_score(y_true=y_test, y_pred=_pred),
            recall_score(y_true=y_test, y_pred=_pred),
            f1_score(y_true=y_test, y_pred=_pred),
            roc_auc_score(y_true=y_test, y_score=_prob),
        ]

    full_comparison = pl.DataFrame(_results)

    print("Full Model Comparison:")
    print("-" * 80)
    print(full_comparison)
    return (full_comparison,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## ROC Curves: All Models

    The ROC curve plots the trade-off between true positive rate (recall) and
    false positive rate across all classification thresholds. Higher AUC indicates
    better discrimination between classes.
    """)
    return


@app.cell
def _(
    X_test,
    dt_model,
    plt,
    rf_model,
    roc_auc_score,
    roc_curve,
    xgb_baseline,
    xgb_optimized,
    y_test,
):
    # ROC curves for all models
    _models = {
        "Decision Tree": (dt_model, "steelblue", "-"),
        "Random Forest": (rf_model, "coral", "-"),
        "XGBoost (Baseline)": (xgb_baseline, "green", "--"),
        "XGBoost (Optimized)": (xgb_optimized, "purple", "-"),
    }

    _fig, _ax = plt.subplots(figsize=(7, 6))

    for _name, (_model, _color, _linestyle) in _models.items():
        _prob = _model.predict_proba(X_test)[:, 1]
        _fpr, _tpr, _ = roc_curve(y_true=y_test, y_score=_prob)
        _auc = roc_auc_score(y_true=y_test, y_score=_prob)

        _ax.plot(
            _fpr,
            _tpr,
            color=_color,
            linestyle=_linestyle,
            linewidth=2,
            label=f"{_name} (AUC = {_auc:.3f})",
        )

    # Diagonal reference line
    _ax.plot(
        [0, 1],
        [0, 1],
        color="gray",
        linestyle="--",
        label="Random (AUC = 0.5)",
    )

    _ax.set_xlabel("False Positive Rate")
    _ax.set_ylabel("True Positive Rate (Recall)")
    _ax.set_title("ROC Curves: Model Comparison")
    _ax.legend(loc="lower right")
    _ax.set_xlim(-0.02, 1.02)
    _ax.set_ylim(-0.02, 1.02)
    _ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Metrics Comparison Bar Chart

    A grouped bar chart provides a visual comparison of all metrics across models.
    """)
    return


@app.cell
def _(full_comparison, np, plt):
    # Grouped bar chart of metrics
    _metrics = full_comparison["Metric"].to_list()
    _models = [_c for _c in full_comparison.columns if _c != "Metric"]

    _x = np.arange(len(_metrics))
    _width = 0.2
    _colors = ["steelblue", "coral", "lightgreen", "mediumpurple"]

    _fig, _ax = plt.subplots(figsize=(10, 5))

    for _i, (_model, _color) in enumerate(zip(_models, _colors)):
        _values = full_comparison[_model].to_list()
        _offset = (_i - len(_models) / 2 + 0.5) * _width
        _ax.bar(
            _x + _offset,
            _values,
            _width,
            label=_model,
            color=_color,
            edgecolor="black",
        )

    _ax.set_xlabel("Metric")
    _ax.set_ylabel("Score")
    _ax.set_title("Model Comparison: Classification Metrics")
    _ax.set_xticks(_x)
    _ax.set_xticklabels(_metrics)
    _ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
    )
    _ax.set_ylim(0, 1)
    _ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary and Key Takeaways

    **XGBoost:**
    - Gradient boosting builds trees sequentially, each correcting previous errors
    - Built-in regularization (L1, L2, gamma) helps prevent overfitting
    - Many hyperparameters require tuning for optimal performance
    - Generally achieves state-of-the-art results on tabular data

    **Optuna for Hyperparameter Optimization:**
    - Bayesian optimization (TPE) is more efficient than grid/random search
    - Cross-validation prevents overfitting to a single train/test split
    - fANOVA analysis reveals which parameters matter most
    - 50 trials is often sufficient for good results; more trials may help

    **When to use each model:**

    | Model | Best for | Trade-offs |
    |-------|----------|------------|
    | Decision Tree | Interpretability, visualization | Prone to overfitting |
    | Random Forest | Quick baseline, robust performance | Less interpretable than single tree |
    | XGBoost | Maximum predictive accuracy | Requires hyperparameter tuning |

    **Practical recommendations:**
    1. Start with Random Forest for a quick baseline
    2. Use XGBoost with Optuna when performance matters
    3. Focus tuning on high-importance parameters (learning_rate, max_depth, n_estimators)
    4. Use cross-validation to get robust performance estimates
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
