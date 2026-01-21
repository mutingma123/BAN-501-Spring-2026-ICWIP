import marimo

__generated_with = "0.19.4"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import pathlib

    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    import seaborn as sns
    import statsmodels.api as sm
    from sklearn.model_selection import KFold, train_test_split

    sns.set_style("whitegrid")
    return KFold, mo, np, pathlib, pl, plt, sm, sns, train_test_split


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Regularized Regression Techniques

    ## The Bias-Variance Tradeoff

    In predictive modeling, we face a fundamental tradeoff:
    - **Low bias**: Model fits training data well (complex models)
    - **Low variance**: Model generalizes to new data (simple models)

    Ordinary Least Squares (OLS) minimizes training error without constraint, which can lead
    to **overfitting**—especially when:
    - Number of features is large relative to observations
    - Features are highly correlated (multicollinearity)
    - The model captures noise rather than signal

    ## Regularization: A Solution

    Regularization adds a **penalty term** to the loss function, discouraging overly complex
    models. This introduces some bias but reduces variance, often improving prediction on
    new data.

    This notebook covers three regularization techniques:
    1. **Ridge Regression** (L2 penalty)
    2. **Lasso Regression** (L1 penalty)
    3. **Elastic Net** (combination of L1 and L2)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Mathematical Background

    ### OLS Objective
    Standard OLS minimizes the **Residual Sum of Squares (RSS)**:

    $$\text{RSS} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 = \sum_{i=1}^{n} \left(y_i - \beta_0 - \sum_{j=1}^{p} \beta_j x_{ij}\right)^2$$

    ### Ridge Regression (L2 Penalty)
    Adds the squared magnitude of coefficients:

    $$\text{Ridge} = \text{RSS} + \lambda \sum_{j=1}^{p} \beta_j^2$$

    - Shrinks coefficients toward zero but **never exactly zero**
    - Good when many features contribute to the outcome
    - Handles multicollinearity well

    ### Lasso Regression (L1 Penalty)
    Adds the absolute magnitude of coefficients:

    $$\text{Lasso} = \text{RSS} + \lambda \sum_{j=1}^{p} |\beta_j|$$

    - Can shrink coefficients **exactly to zero** (feature selection)
    - Produces sparse, interpretable models
    - May arbitrarily select one feature among correlated groups

    ### Elastic Net (Combined Penalty)
    Combines both L1 and L2 penalties:

    $$\text{Elastic Net} = \text{RSS} + \lambda \left[(1-\alpha)\frac{1}{2}\sum_{j=1}^{p} \beta_j^2 + \alpha \sum_{j=1}^{p} |\beta_j|\right]$$

    - $\alpha = 1$: Pure Lasso
    - $\alpha = 0$: Pure Ridge
    - $0 < \alpha < 1$: Blends both penalties
    - Handles correlated feature groups better than Lasso alone
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load the Ames Housing Dataset

    We use the Ames Housing dataset, which contains 1,460 observations and 81 features
    describing residential properties in Ames, Iowa. Our target variable is `SalePrice`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Feature Preparation

    To better demonstrate regularization benefits, we use **all available features** from the
    Ames Housing dataset, including both numeric and categorical variables. This creates a
    high-dimensional scenario (~200 features) where regularization clearly outperforms OLS.

    **Preprocessing steps:**
    1. Drop columns with ≥25% missing values
    2. Drop the Id column
    3. Impute remaining missing values (median for numeric, mode for categorical)
    4. One-hot encode all categorical columns (drop first category to avoid collinearity)

    This results in approximately 200 features with ~1,168 training samples—a ~6:1 sample-to-feature
    ratio where regularization provides clear benefits.
    """)
    return


@app.cell
def _(pathlib, pl):
    # Load data
    _data_filepath = pathlib.Path("data/regression/train.parquet")
    raw_data = pl.read_parquet(_data_filepath)
    print(f"Dataset shape: {raw_data.shape[0]} rows x {raw_data.shape[1]} columns")

    # Drop columns with ≥25% missing values
    MISSING_VALUE_THRESHOLD = 0.25
    _missing_value_proportions = (raw_data.null_count() / len(raw_data)).to_dicts()[0]
    _columns_to_drop = [
        _key for _key, _val in _missing_value_proportions.items()
        if _val >= MISSING_VALUE_THRESHOLD
    ]
    raw_data = raw_data.drop(_columns_to_drop)
    print(f"Dropped columns (≥25% missing): {_columns_to_drop}")

    # Impute numeric columns with median
    _numeric_fill_list = [
        pl.col(_col).fill_null(pl.col(_col).median())
        for _col in raw_data.columns
        if raw_data[_col].dtype.is_numeric()
    ]
    raw_data = raw_data.with_columns(_numeric_fill_list)

    # Impute categorical columns with mode
    _categorical_fill_list = [
        pl.col(_col).fill_null(pl.col(_col).mode().first())
        for _col in raw_data.columns
        if not raw_data[_col].dtype.is_numeric()
    ]
    raw_data = raw_data.with_columns(_categorical_fill_list)

    # One-hot encode categorical columns
    model_data = raw_data.to_dummies(
        columns=[
            _col for _col in raw_data.columns
            if not raw_data[_col].dtype.is_numeric()
        ],
        drop_first=True,
    )

    # Drop Id column
    model_data = model_data.drop("Id")

    target_column = "SalePrice"
    feature_columns = [c for c in model_data.columns if c != target_column]

    print(f"Total features after encoding: {len(feature_columns)}")
    print(f"Samples: {len(model_data)}")
    return feature_columns, model_data, target_column


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data Preparation

    We perform the following steps:
    1. **Train/test split**: 80% training, 20% test (held out for final evaluation)
    2. **Standardization**: Scale features to have mean=0, std=1

    ### Why Standardize Features?

    Standardization (z-score normalization) transforms each feature to have mean=0 and
    standard deviation=1:

    $$x_{\text{scaled}} = \frac{x - \mu}{\sigma}$$

    This is necessary for regularized regression because:

    1. **Equal penalty across features**: The regularization penalty $\lambda \sum \beta_j^2$
       (Ridge) or $\lambda \sum |\beta_j|$ (Lasso) treats all coefficients equally. Without
       standardization, features measured in larger units (e.g., square feet vs. number of
       rooms) would have smaller coefficients and thus be penalized less.

    2. **Comparable coefficient magnitudes**: After standardization, each coefficient
       represents the change in y per one standard deviation change in that feature,
       making coefficients directly comparable.

    3. **Numerical stability**: Gradient-based optimization converges faster when features
       are on similar scales.

    **Trade-off**: Coefficients on standardized data are not directly interpretable in
    original units. We will demonstrate how to transform them back after fitting.
    """)
    return


@app.cell
def _(feature_columns, model_data, pl, target_column, train_test_split):
    # Train/test split using sklearn
    train_data, test_data = train_test_split(
        model_data,
        test_size=0.2,
        random_state=42,
    )

    # Calculate standardization parameters from training data
    _train_means = train_data.select(feature_columns).mean()
    _train_stds = train_data.select(feature_columns).std()

    # Identify columns with zero or very small std (drop these)
    _valid_cols = [
        col for col in feature_columns
        if _train_stds[col][0] is not None and _train_stds[col][0] > 1e-10
    ]
    _dropped_cols = set(feature_columns) - set(_valid_cols)
    if _dropped_cols:
        print(f"Dropped {len(_dropped_cols)} constant columns")

    # Standardize features using polars expressions
    _standardize_exprs = [
        ((pl.col(col) - _train_means[col][0]) / _train_stds[col][0]).alias(col)
        for col in _valid_cols
    ]

    _train_scaled = train_data.select(_standardize_exprs)
    _test_scaled = test_data.select(_standardize_exprs)

    # Add intercept column (1.0 for all rows)
    _train_scaled = _train_scaled.with_columns(pl.lit(1.0).alias("intercept"))
    _test_scaled = _test_scaled.with_columns(pl.lit(1.0).alias("intercept"))

    # Reorder columns: intercept first, then features
    _col_order = ["intercept"] + _valid_cols

    # Extract numpy arrays for statsmodels
    X_train = _train_scaled.select(_col_order).to_numpy()
    X_test = _test_scaled.select(_col_order).to_numpy()
    y_train = train_data.select(target_column).to_numpy().flatten()
    y_test = test_data.select(target_column).to_numpy().flatten()

    # Update feature_columns to only include valid columns
    feature_columns_final = _valid_cols

    print(f"Training samples: {len(y_train)}")
    print(f"Test samples: {len(y_test)}")
    print(f"Features (including intercept): {X_train.shape[1]}")
    return X_test, X_train, feature_columns_final, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # Part 1: OLS Baseline

    Before exploring regularization, we establish a baseline using standard OLS regression.
    This gives us a reference point for comparing regularized models.
    """)
    return


@app.cell
def _(X_train, feature_columns_final, sm, y_train):
    # Fit OLS model
    ols_model = sm.OLS(
        endog=y_train,
        exog=X_train,
    ).fit()

    # Display summary (truncated for high-dimensional data)
    print(f"OLS R² (training): {ols_model.rsquared:.4f}")
    print(f"OLS Adjusted R² (training): {ols_model.rsquared_adj:.4f}")
    print(f"Number of coefficients: {len(ols_model.params)}")

    # Extract coefficients (skip intercept for feature comparison)
    ols_coefs = dict(zip(feature_columns_final, ols_model.params[1:]))
    return (ols_model,)


@app.cell
def _(X_test, np, ols_model, y_test):
    # Evaluate OLS on test set
    ols_predictions = ols_model.predict(X_test)
    ols_mse = np.mean((y_test - ols_predictions) ** 2)
    ols_rmse = np.sqrt(ols_mse)
    ols_r2 = 1 - np.sum((y_test - ols_predictions) ** 2) / np.sum(
        (y_test - np.mean(y_test)) ** 2
    )

    print(f"OLS Test Performance:")
    print(f"  RMSE: ${ols_rmse:,.0f}")
    print(f"  R²: {ols_r2:.4f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # Part 2: Ridge Regression (L2 Penalty)

    Ridge regression adds an L2 penalty that shrinks coefficients toward zero.
    In statsmodels, we use `fit_regularized()` with `L1_wt=0` for pure L2 penalty.

    Key characteristics:
    - Shrinks all coefficients but keeps them non-zero
    - Reduces variance at the cost of introducing bias
    - Handles multicollinearity by distributing weight among correlated features
    """)
    return


@app.cell
def _(X_train, feature_columns_final, np, sm, y_train):
    # Test different alpha values for Ridge
    # Note: statsmodels alpha values need to be small relative to data scale
    ridge_alphas = [0.0001, 0.001, 0.005, 0.01, 0.05]
    ridge_coefs_by_alpha = {}

    for _alpha in ridge_alphas:
        _model = sm.OLS(
            endog=y_train,
            exog=X_train,
        )
        _fit = _model.fit_regularized(
            alpha=_alpha,
            L1_wt=0,  # L1_wt=0 means pure L2 (Ridge)
        )
        ridge_coefs_by_alpha[_alpha] = _fit.params[1:]  # Skip intercept

    # Display summary statistics for high-dimensional data
    print("Ridge Coefficient Summary by Alpha:")
    print("-" * 75)
    print(f"{'Alpha':<12} {'Mean |Coef|':<15} {'Max |Coef|':<15} {'Min |Coef|':<15}")
    print("-" * 75)

    for _alpha in ridge_alphas:
        _coefs = ridge_coefs_by_alpha[_alpha]
        _abs_coefs = np.abs(_coefs)
        print(f"{_alpha:<12} {np.mean(_abs_coefs):<15.2f} {np.max(_abs_coefs):<15.2f} {np.min(_abs_coefs):<15.4f}")

    # Select moderate alpha for comparison
    ridge_alpha_default = 0.001
    ridge_coefs = dict(zip(feature_columns_final, ridge_coefs_by_alpha[ridge_alpha_default]))
    return (ridge_alpha_default,)


@app.cell
def _(X_test, X_train, np, ridge_alpha_default, sm, y_test, y_train):
    # Fit ridge model with default alpha and evaluate
    ridge_model = sm.OLS(
        endog=y_train,
        exog=X_train,
    ).fit_regularized(
        alpha=ridge_alpha_default,
        L1_wt=0,
    )

    ridge_predictions = ridge_model.predict(X_test)
    ridge_mse = np.mean((y_test - ridge_predictions) ** 2)
    ridge_rmse = np.sqrt(ridge_mse)
    ridge_r2 = 1 - np.sum((y_test - ridge_predictions) ** 2) / np.sum(
        (y_test - np.mean(y_test)) ** 2
    )

    print(f"Ridge (alpha={ridge_alpha_default}) Test Performance:")
    print(f"  RMSE: ${ridge_rmse:,.0f}")
    print(f"  R²: {ridge_r2:.4f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # Part 3: Lasso Regression (L1 Penalty)

    Lasso (Least Absolute Shrinkage and Selection Operator) uses an L1 penalty.
    In statsmodels, we use `fit_regularized()` with `L1_wt=1` for pure L1 penalty.

    Key characteristics:
    - Can shrink coefficients **exactly to zero** (automatic feature selection)
    - Produces sparse, more interpretable models
    - May arbitrarily select one feature among highly correlated groups
    """)
    return


@app.cell
def _(X_train, feature_columns_final, np, sm, y_train):
    # Test different alpha values for Lasso
    # Larger alphas needed to see feature selection effect
    lasso_alphas = [1, 10, 50, 100, 500]
    lasso_coefs_by_alpha = {}

    for _alpha in lasso_alphas:
        _model = sm.OLS(
            endog=y_train,
            exog=X_train,
        )
        _fit = _model.fit_regularized(
            alpha=_alpha,
            L1_wt=1,  # L1_wt=1 means pure L1 (Lasso)
        )
        lasso_coefs_by_alpha[_alpha] = _fit.params[1:]  # Skip intercept

    # Display summary showing feature selection effect
    print("Lasso Feature Selection by Alpha:")
    print("-" * 80)
    print(f"{'Alpha':<12} {'# Non-zero':<15} {'# Zeroed':<15} {'Mean |Non-zero|':<20}")
    print("-" * 80)

    for _alpha in lasso_alphas:
        _coefs = lasso_coefs_by_alpha[_alpha]
        _n_nonzero = sum(abs(_c) >= 1 for _c in _coefs)
        _n_zeros = len(_coefs) - _n_nonzero
        _nonzero_mean = np.mean([abs(_c) for _c in _coefs if abs(_c) >= 1]) if _n_nonzero > 0 else 0
        print(f"{_alpha:<12} {_n_nonzero:<15} {_n_zeros:<15} {_nonzero_mean:<20.2f}")

    print(f"\nTotal features: {len(feature_columns_final)}")

    # Select moderate alpha for comparison (shows some feature selection)
    lasso_alpha_default = 50
    lasso_coefs = dict(zip(feature_columns_final, lasso_coefs_by_alpha[lasso_alpha_default]))
    return (lasso_alpha_default,)


@app.cell
def _(X_test, X_train, lasso_alpha_default, np, sm, y_test, y_train):
    # Fit lasso model with default alpha and evaluate
    lasso_model = sm.OLS(
        endog=y_train,
        exog=X_train,
    ).fit_regularized(
        alpha=lasso_alpha_default,
        L1_wt=1,
    )

    lasso_predictions = lasso_model.predict(X_test)
    lasso_mse = np.mean((y_test - lasso_predictions) ** 2)
    lasso_rmse = np.sqrt(lasso_mse)
    lasso_r2 = 1 - np.sum((y_test - lasso_predictions) ** 2) / np.sum(
        (y_test - np.mean(y_test)) ** 2
    )

    print(f"Lasso (alpha={lasso_alpha_default}) Test Performance:")
    print(f"  RMSE: ${lasso_rmse:,.0f}")
    print(f"  R²: {lasso_r2:.4f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # Part 4: Elastic Net (Combined L1 + L2 Penalty)

    Elastic Net combines the L1 and L2 penalties, controlled by the `L1_wt` parameter:
    - `L1_wt = 0`: Pure Ridge (L2 only)
    - `L1_wt = 1`: Pure Lasso (L1 only)
    - `0 < L1_wt < 1`: Mix of both penalties

    Key characteristics:
    - Gets feature selection from L1 (like Lasso)
    - Gets grouped selection from L2 (keeps correlated features together)
    - Often performs better when features are correlated
    """)
    return


@app.cell
def _(X_train, feature_columns_final, np, sm, y_train):
    # Test different L1_wt values with moderate alpha
    # Note: Alpha needs to be smaller to show useful coefficients across L1_wt range
    enet_alpha = 0.01
    enet_l1_weights = [0.0, 0.25, 0.5, 0.75, 1.0]

    # Store coefficients for each L1_wt
    enet_coefs_by_l1wt = {}

    for _l1wt in enet_l1_weights:
        _model = sm.OLS(
            endog=y_train,
            exog=X_train,
        )
        _fit = _model.fit_regularized(
            alpha=enet_alpha,
            L1_wt=_l1wt,
        )
        enet_coefs_by_l1wt[_l1wt] = _fit.params[1:]  # Skip intercept

    # Display summary showing how L1_wt affects sparsity
    print(f"Elastic Net: Effect of L1_wt (alpha={enet_alpha}):")
    print("(L1_wt=0 is Ridge, L1_wt=1 is Lasso)")
    print("-" * 75)
    print(f"{'L1_wt':<12} {'# Non-zero':<15} {'# Zeroed':<15} {'Mean |Coef|':<15}")
    print("-" * 75)

    for _l1wt in enet_l1_weights:
        _coefs = enet_coefs_by_l1wt[_l1wt]
        _n_nonzero = sum(abs(_c) >= 1 for _c in _coefs)
        _n_zeros = len(_coefs) - _n_nonzero
        _mean_abs = np.mean(np.abs(_coefs))
        print(f"{_l1wt:<12} {_n_nonzero:<15} {_n_zeros:<15} {_mean_abs:<15.2f}")

    # Select L1_wt=0.5 for comparison
    enet_l1wt_default = 0.5
    enet_coefs = dict(zip(feature_columns_final, enet_coefs_by_l1wt[enet_l1wt_default]))
    return enet_alpha, enet_l1wt_default


@app.cell
def _(X_test, X_train, enet_alpha, enet_l1wt_default, np, sm, y_test, y_train):
    # Fit elastic net model and evaluate
    enet_model = sm.OLS(
        endog=y_train,
        exog=X_train,
    ).fit_regularized(
        alpha=enet_alpha,
        L1_wt=enet_l1wt_default,
    )

    enet_predictions = enet_model.predict(X_test)
    enet_mse = np.mean((y_test - enet_predictions) ** 2)
    enet_rmse = np.sqrt(enet_mse)
    enet_r2 = 1 - np.sum((y_test - enet_predictions) ** 2) / np.sum(
        (y_test - np.mean(y_test)) ** 2
    )

    print(f"Elastic Net (alpha={enet_alpha}, L1_wt={enet_l1wt_default}) Test Performance:")
    print(f"  RMSE: ${enet_rmse:,.0f}")
    print(f"  R²: {enet_r2:.4f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # Part 5: Cross-Validation for Hyperparameter Tuning

    The regularization strength (alpha) is a **hyperparameter** that controls the
    bias-variance tradeoff. Too small: overfitting. Too large: underfitting.

    Cross-validation helps select the optimal alpha by:
    1. Splitting training data into K folds
    2. For each alpha value, train on K-1 folds and evaluate on the held-out fold
    3. Average the scores across folds
    4. Select alpha with best average CV score
    """)
    return


@app.cell
def _(KFold, X_train, np, plt, sm, sns, y_train):
    # Cross-validation for Ridge
    ridge_cv_alphas = np.array([0.005, 0.01, 0.015, 0.02, 0.05])
    kfold = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    ridge_cv_scores = []
    for _alpha in ridge_cv_alphas:
        _fold_scores = []

        for _train_idx, _val_idx in kfold.split(X_train):
            # Extract fold data
            _X_tr = X_train[_train_idx]
            _X_val = X_train[_val_idx]
            _y_tr = y_train[_train_idx]
            _y_val = y_train[_val_idx]

            # Fit model
            _model = sm.OLS(
                endog=_y_tr,
                exog=_X_tr,
            ).fit_regularized(
                alpha=_alpha,
                L1_wt=0,
            )

            # Evaluate
            _pred = _model.predict(_X_val)
            _mse = np.mean((_y_val - _pred) ** 2)
            _fold_scores.append(_mse)

        ridge_cv_scores.append(np.mean(_fold_scores))

    ridge_cv_scores = np.array(ridge_cv_scores)
    ridge_best_alpha = ridge_cv_alphas[np.argmin(ridge_cv_scores)]
    ridge_best_cv_rmse = np.sqrt(np.min(ridge_cv_scores))

    # Plot CV results
    _fig, _ax = plt.subplots(figsize=(8, 4))
    sns.lineplot(
        x=ridge_cv_alphas,
        y=np.sqrt(ridge_cv_scores),
        marker="o",
        ax=_ax,
    )
    _ax.axvline(
        x=ridge_best_alpha,
        color="red",
        linestyle="--",
        label=f"Best alpha={ridge_best_alpha:.5f}",
    )
    _ax.set_xscale("log")
    _ax.set_xlabel("Alpha (log scale)")
    _ax.set_ylabel("CV RMSE ($)")
    _ax.set_title("Ridge Regression: Cross-Validation for Alpha Selection")
    _ax.legend()
    plt.tight_layout()
    plt.show()

    print(f"Best Ridge alpha: {ridge_best_alpha:.6f}")
    print(f"Best Ridge CV RMSE: ${ridge_best_cv_rmse:,.0f}")
    return kfold, ridge_best_alpha


@app.cell
def _(X_train, kfold, np, plt, sm, sns, y_train):
    # Cross-validation for Lasso
    lasso_cv_alphas = np.array([100, 300, 600, 1000, 1500])

    lasso_cv_scores = []
    for _alpha in lasso_cv_alphas:
        _fold_scores = []

        for _train_idx, _val_idx in kfold.split(X_train):
            # Extract fold data
            _X_tr = X_train[_train_idx]
            _X_val = X_train[_val_idx]
            _y_tr = y_train[_train_idx]
            _y_val = y_train[_val_idx]

            # Fit model
            _model = sm.OLS(
                endog=_y_tr,
                exog=_X_tr,
            ).fit_regularized(
                alpha=_alpha,
                L1_wt=1,
            )

            # Evaluate
            _pred = _model.predict(_X_val)
            _mse = np.mean((_y_val - _pred) ** 2)
            _fold_scores.append(_mse)

        lasso_cv_scores.append(np.mean(_fold_scores))

    lasso_cv_scores = np.array(lasso_cv_scores)
    lasso_best_alpha = lasso_cv_alphas[np.argmin(lasso_cv_scores)]
    lasso_best_cv_rmse = np.sqrt(np.min(lasso_cv_scores))

    # Plot CV results
    _fig, _ax = plt.subplots(figsize=(8, 4))
    sns.lineplot(
        x=lasso_cv_alphas,
        y=np.sqrt(lasso_cv_scores),
        marker="o",
        ax=_ax,
    )
    _ax.axvline(
        x=lasso_best_alpha,
        color="red",
        linestyle="--",
        label=f"Best alpha={lasso_best_alpha:.2f}",
    )
    _ax.set_xscale("log")
    _ax.set_xlabel("Alpha (log scale)")
    _ax.set_ylabel("CV RMSE ($)")
    _ax.set_title("Lasso Regression: Cross-Validation for Alpha Selection")
    _ax.legend()
    plt.tight_layout()
    plt.show()

    print(f"Best Lasso alpha: {lasso_best_alpha:.2f}")
    print(f"Best Lasso CV RMSE: ${lasso_best_cv_rmse:,.0f}")
    return (lasso_best_alpha,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # Part 6: Model Comparison

    Now we compare all methods using the CV-selected hyperparameters on the held-out
    test set. This gives us an unbiased estimate of model performance.
    """)
    return


@app.cell
def _(
    X_test,
    X_train,
    feature_columns_final,
    lasso_best_alpha,
    np,
    pl,
    ridge_best_alpha,
    sm,
    y_test,
    y_train,
):
    # Fit final models with CV-selected alphas
    final_ols = sm.OLS(endog=y_train, exog=X_train).fit()

    final_ridge = sm.OLS(endog=y_train, exog=X_train).fit_regularized(
        alpha=ridge_best_alpha,
        L1_wt=0,
    )

    final_lasso = sm.OLS(endog=y_train, exog=X_train).fit_regularized(
        alpha=lasso_best_alpha,
        L1_wt=1,
    )

    final_enet = sm.OLS(endog=y_train, exog=X_train).fit_regularized(
        alpha=0.01,  # Using moderate alpha for elastic net
        L1_wt=0.5,
    )

    # Evaluate on test set
    _n_features = len(feature_columns_final)
    _models = {
        "OLS": final_ols,
        "Ridge": final_ridge,
        "Lasso": final_lasso,
        "Elastic Net": final_enet,
    }

    results = []
    final_coefs = {}

    for _name, _model in _models.items():
        _pred = _model.predict(X_test)
        _mse = np.mean((y_test - _pred) ** 2)
        _rmse = np.sqrt(_mse)
        _r2 = 1 - np.sum((y_test - _pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
        _n_zeros = sum(abs(_c) < 1 for _c in _model.params[1:])

        results.append({
            "Model": _name,
            "RMSE": _rmse,
            "R²": _r2,
            "Non-zero Features": _n_features - _n_zeros,
        })
        final_coefs[_name] = dict(zip(feature_columns_final, _model.params[1:]))

    results_df = pl.DataFrame(results)

    print("Model Comparison on Test Set:")
    print(results_df)
    return (final_coefs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Coefficient Comparison (Original Scale)

    With ~250 features, we visualize regularization effects using scatter plots comparing
    OLS coefficients (x-axis) to regularized coefficients (y-axis). Points below the
    diagonal line (y=x) indicate shrinkage toward zero.
    """)
    return


@app.cell
def _(feature_columns_final, final_coefs, np, plt):
    _ols = np.array([final_coefs["OLS"][f] for f in feature_columns_final])
    _ridge = np.array([final_coefs["Ridge"][f] for f in feature_columns_final])
    _lasso = np.array([final_coefs["Lasso"][f] for f in feature_columns_final])

    _fig, _axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

    # Ridge vs OLS
    _axes[0].scatter(_ols, _ridge, alpha=0.6, edgecolor="black", linewidth=0.5)
    _lim = max(abs(_ols).max(), abs(_ridge).max()) * 1.1
    _axes[0].plot([-_lim, _lim], [-_lim, _lim], "k--", alpha=0.5, label="y = x (no shrinkage)")
    _axes[0].set_xlim(-_lim, _lim)
    _axes[0].set_ylim(-_lim, _lim)
    _axes[0].set_xlabel("OLS Coefficient")
    _axes[0].set_ylabel("Ridge Coefficient")
    _axes[0].set_title("Ridge Shrinkage")
    _axes[0].legend()
    _axes[0].axhline(0, color="gray", linewidth=0.5)
    _axes[0].axvline(0, color="gray", linewidth=0.5)

    # Lasso vs OLS
    _n_zeroed = sum(abs(_lasso) < 1)
    _axes[1].scatter(_ols, _lasso, alpha=0.6, edgecolor="black", linewidth=0.5)
    _axes[1].plot([-_lim, _lim], [-_lim, _lim], "k--", alpha=0.5, label="y = x (no shrinkage)")
    _axes[1].set_xlim(-_lim, _lim)
    _axes[1].set_ylim(-_lim, _lim)
    _axes[1].set_xlabel("OLS Coefficient")
    _axes[1].set_ylabel("Lasso Coefficient")
    _axes[1].set_title(f"Lasso Shrinkage ({_n_zeroed} features zeroed)")
    _axes[1].legend()
    _axes[1].axhline(0, color="gray", linewidth=0.5)
    _axes[1].axvline(0, color="gray", linewidth=0.5)

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # Key Takeaways

    ## When to Use Each Method

    | Method | Best For | Trade-offs |
    |--------|----------|------------|
    | **OLS** | Few features, no multicollinearity, interpretability | Can overfit with many features |
    | **Ridge** | Many features, multicollinearity, retain all predictors | No feature selection, all coefficients non-zero |
    | **Lasso** | Feature selection needed, sparse models, interpretability | May be unstable with correlated features |
    | **Elastic Net** | Correlated feature groups, want some feature selection | Two hyperparameters to tune |

    ## Summary of Results

    For this Ames Housing dataset with **~250 features** (numeric + one-hot encoded categoricals):
    - **OLS** fits the training data well but may overfit with the high feature-to-sample ratio (~5:1)
    - **Ridge** shrinks all coefficients toward zero, reducing variance
    - **Lasso** performs automatic feature selection, zeroing out many irrelevant features
    - **Elastic Net** combines both behaviors

    The high-dimensional setting demonstrates regularization's value: with many features relative to
    samples, regularization helps prevent overfitting and often improves test set performance. Lasso's
    feature selection capability is particularly useful for identifying the most predictive variables.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
