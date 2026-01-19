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
    import statsmodels.api as sm
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.preprocessing import StandardScaler

    sns.set_style("whitegrid")
    return KFold, StandardScaler, mo, np, pl, plt, sm, sns


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


@app.cell
def _(pl):
    raw_data = pl.read_parquet("data/regression/train.parquet")
    print(f"Dataset shape: {raw_data.shape[0]} rows x {raw_data.shape[1]} columns")
    raw_data.head()
    return (raw_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Feature Selection

    For clarity, we select a subset of numeric features that are commonly important
    predictors of house price. We avoid categorical features to keep the focus on
    regularization concepts.

    **Selected features:**
    - `GrLivArea`: Above grade living area (sq ft)
    - `TotalBsmtSF`: Total basement area (sq ft)
    - `OverallQual`: Overall material and finish quality (1-10)
    - `GarageCars`: Garage capacity in cars
    - `GarageArea`: Garage size (sq ft)
    - `1stFlrSF`: First floor area (sq ft)
    - `FullBath`: Number of full bathrooms
    - `YearBuilt`: Year of construction
    - `YearRemodAdd`: Year of remodel (same as construction if none)
    - `TotRmsAbvGrd`: Total rooms above grade (excludes bathrooms)
    - `Fireplaces`: Number of fireplaces
    - `LotArea`: Lot size (sq ft)
    """)
    return


@app.cell
def _(raw_data):
    feature_columns = [
        "GrLivArea",
        "TotalBsmtSF",
        "OverallQual",
        "GarageCars",
        "GarageArea",
        "1stFlrSF",
        "FullBath",
        "YearBuilt",
        "YearRemodAdd",
        "TotRmsAbvGrd",
        "Fireplaces",
        "LotArea",
    ]
    target_column = "SalePrice"

    # Select features and target, drop rows with any missing values
    model_data = raw_data.select(feature_columns + [target_column]).drop_nulls()
    print(f"Samples after removing nulls: {model_data.shape[0]}")
    model_data.head()
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
def _(StandardScaler, feature_columns, model_data, np, target_column):
    TEST_FRACTION = 0.2

    # Extract feature matrix and target
    X_raw = model_data.select(feature_columns).to_numpy()
    y = model_data.select(target_column).to_numpy().flatten()

    # Train/test split (manual to keep it simple and reproducible)
    np.random.seed(42)
    n_samples = len(y)
    n_test = int(TEST_FRACTION * n_samples)
    _indices = np.random.permutation(n_samples)
    train_idx = _indices[n_test:]
    test_idx = _indices[:n_test]

    X_train_raw = X_raw[train_idx]
    X_test_raw = X_raw[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    # Standardize features (fit on training data only)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    # Add constant term for statsmodels
    # statsmodels OLS does not automatically include an intercept term, unlike
    # scikit-learn. We manually add a column of ones so the model can estimate
    # an intercept (beta_0). Without this, the regression line would be forced
    # through the origin, which is inappropriate for most real-world data.
    X_train = np.column_stack([np.ones(len(X_train_scaled)), X_train_scaled])
    X_test = np.column_stack([np.ones(len(X_test_scaled)), X_test_scaled])

    print(f"Training samples: {len(y_train)}")
    print(f"Test samples: {len(y_test)}")
    print(f"Features (including intercept): {X_train.shape[1]}")
    return X_test, X_train, scaler, y_test, y_train


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
def _(X_train, feature_columns, sm, y_train):
    # Fit OLS model
    ols_model = sm.OLS(
        endog=y_train,
        exog=X_train,
    ).fit()

    # Display summary
    print(ols_model.summary())

    # Extract coefficients (skip intercept for feature comparison)
    ols_coefs = dict(zip(feature_columns, ols_model.params[1:]))
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
    ## Transforming Coefficients Back to Original Scale

    The coefficients from our standardized model tell us the change in sale price per
    one standard deviation change in each feature. To interpret coefficients in original
    units (e.g., "dollars per square foot"), we need to transform them back.

    ### The Math

    When we standardize: $x_{\text{scaled}} = \frac{x - \mu}{\sigma}$

    The model learns: $\hat{y} = \beta_0^{(s)} + \sum_j \beta_j^{(s)} \cdot x_j^{(s)}$

    Substituting the standardization:
    $$\hat{y} = \beta_0^{(s)} + \sum_j \beta_j^{(s)} \cdot \frac{x_j - \mu_j}{\sigma_j}$$

    Rearranging to original scale:
    $$\hat{y} = \underbrace{\left(\beta_0^{(s)} - \sum_j \frac{\beta_j^{(s)} \mu_j}{\sigma_j}\right)}_{\beta_0^{(\text{orig})}} + \sum_j \underbrace{\frac{\beta_j^{(s)}}{\sigma_j}}_{\beta_j^{(\text{orig})}} \cdot x_j$$

    Therefore:
    - **Original coefficient**: $\beta_j^{(\text{orig})} = \frac{\beta_j^{(\text{scaled})}}{\sigma_j}$
    - **Original intercept**: $\beta_0^{(\text{orig})} = \beta_0^{(\text{scaled})} - \sum_j \frac{\beta_j^{(\text{scaled})} \cdot \mu_j}{\sigma_j}$
    """)
    return


@app.cell
def _(feature_columns, np, ols_model, scaler):
    def transform_coefs_to_original_scale(model_params, scaler, feature_names):
        """
        Transform coefficients from standardized scale back to original scale.

        Parameters
        ----------
        model_params : array-like
            Model parameters where params[0] is the intercept and params[1:] are
            feature coefficients (on standardized scale).
        scaler : StandardScaler
            Fitted scaler with mean_ and scale_ attributes.
        feature_names : list
            Names of features (for output dictionary).

        Returns
        -------
        dict with 'intercept' and feature coefficients on original scale.
        """
        _intercept_scaled = model_params[0]
        _coefs_scaled = model_params[1:]

        # Transform coefficients: beta_orig = beta_scaled / std
        _coefs_orig = _coefs_scaled / scaler.scale_

        # Transform intercept: beta0_orig = beta0_scaled - sum(beta_scaled * mean / std)
        _intercept_orig = _intercept_scaled - np.sum(_coefs_scaled * scaler.mean_ / scaler.scale_)

        _result = {"intercept": _intercept_orig}
        for _name, _coef in zip(feature_names, _coefs_orig):
            _result[_name] = _coef
        return _result

    # Transform OLS coefficients to original scale
    ols_coefs_original = transform_coefs_to_original_scale(
        model_params=ols_model.params,
        scaler=scaler,
        feature_names=feature_columns,
    )

    # Display comparison
    print("OLS Coefficients: Standardized vs Original Scale")
    print("-" * 70)
    print(f"{'Feature':<15} {'Standardized':>15} {'Original':>15} {'Unit interpretation':<20}")
    print("-" * 70)
    print(f"{'Intercept':<15} {ols_model.params[0]:>15,.0f} {ols_coefs_original['intercept']:>15,.0f}")
    for _i, _feat in enumerate(feature_columns):
        _std_coef = ols_model.params[_i + 1]
        _orig_coef = ols_coefs_original[_feat]
        print(f"{_feat:<15} {_std_coef:>15,.0f} {_orig_coef:>15,.2f}")
    return (transform_coefs_to_original_scale,)


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
def _(X_train, feature_columns, sm, y_train):
    # Test different alpha values for Ridge
    # Note: statsmodels alpha values need to be small relative to data scale
    ridge_alphas = [0.0001, 0.001, 0.005, 0.01, 0.05]

    # Store coefficients for each alpha
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

    # Display coefficient comparison
    print("Ridge Coefficients by Alpha:")
    print("-" * 85)
    print(f"{'Feature':<15}", end="")
    for _alpha in ridge_alphas:
        print(f"{_alpha:>14}", end="")
    print()
    print("-" * 85)
    for _i, _feat in enumerate(feature_columns):
        print(f"{_feat:<15}", end="")
        for _alpha in ridge_alphas:
            print(f"{ridge_coefs_by_alpha[_alpha][_i]:>14.0f}", end="")
        print()

    # Select moderate alpha for comparison
    ridge_alpha_default = 0.001
    ridge_coefs = dict(zip(feature_columns, ridge_coefs_by_alpha[ridge_alpha_default]))
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
def _(X_train, feature_columns, sm, y_train):
    # Test different alpha values for Lasso
    # Larger alphas needed to see feature selection effect
    lasso_alphas = [1, 10, 50, 100, 500]

    # Store coefficients for each alpha
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

    # Display coefficient comparison
    print("Lasso Coefficients by Alpha (notice zeros appearing as alpha increases):")
    print("-" * 85)
    print(f"{'Feature':<15}", end="")
    for _alpha in lasso_alphas:
        print(f"{_alpha:>14}", end="")
    print()
    print("-" * 85)
    for _i, _feat in enumerate(feature_columns):
        print(f"{_feat:<15}", end="")
        for _alpha in lasso_alphas:
            _coef = lasso_coefs_by_alpha[_alpha][_i]
            if abs(_coef) < 1:
                print(f"{'0':>14}", end="")  # Show zeros clearly
            else:
                print(f"{_coef:>14.0f}", end="")
        print()

    # Count zeros for each alpha
    print("-" * 85)
    print(f"{'# Zeros':<15}", end="")
    for _alpha in lasso_alphas:
        _n_zeros = sum(abs(_c) < 1 for _c in lasso_coefs_by_alpha[_alpha])
        print(f"{_n_zeros:>14}", end="")
    print()

    # Select moderate alpha for comparison (shows some feature selection)
    lasso_alpha_default = 50
    lasso_coefs = dict(zip(feature_columns, lasso_coefs_by_alpha[lasso_alpha_default]))
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
def _(X_train, feature_columns, sm, y_train):
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

    # Display coefficient comparison
    print(f"Elastic Net Coefficients by L1_wt (alpha={enet_alpha}):")
    print(f"(L1_wt=0 is Ridge, L1_wt=1 is Lasso)")
    print("-" * 85)
    print(f"{'Feature':<15}", end="")
    for _l1wt in enet_l1_weights:
        print(f"{_l1wt:>14}", end="")
    print()
    print("-" * 85)
    for _i, _feat in enumerate(feature_columns):
        print(f"{_feat:<15}", end="")
        for _l1wt in enet_l1_weights:
            _coef = enet_coefs_by_l1wt[_l1wt][_i]
            if abs(_coef) < 1:
                print(f"{'0':>14}", end="")
            else:
                print(f"{_coef:>14.0f}", end="")
        print()

    # Select L1_wt=0.5 for comparison
    enet_l1wt_default = 0.5
    enet_coefs = dict(zip(feature_columns, enet_coefs_by_l1wt[enet_l1wt_default]))
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
    ridge_cv_alphas = np.logspace(-5, -1, 20)  # Range from 0.00001 to 0.1
    kfold = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    ridge_cv_scores = []
    for _alpha in ridge_cv_alphas:
        _fold_scores = []
        for _train_idx, _val_idx in kfold.split(X_train):
            _X_tr = X_train[_train_idx]
            _X_val = X_train[_val_idx]
            _y_tr = y_train[_train_idx]
            _y_val = y_train[_val_idx]

            _model = sm.OLS(
                endog=_y_tr,
                exog=_X_tr,
            ).fit_regularized(
                alpha=_alpha,
                L1_wt=0,
            )
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
    lasso_cv_alphas = np.logspace(-1, 4, 20)  # Range from 0.1 to 1000

    lasso_cv_scores = []
    for _alpha in lasso_cv_alphas:
        _fold_scores = []
        for _train_idx, _val_idx in kfold.split(X_train):
            _X_tr = X_train[_train_idx]
            _X_val = X_train[_val_idx]
            _y_tr = y_train[_train_idx]
            _y_val = y_train[_val_idx]

            _model = sm.OLS(
                endog=_y_tr,
                exog=_X_tr,
            ).fit_regularized(
                alpha=_alpha,
                L1_wt=1,
            )
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
    feature_columns,
    lasso_best_alpha,
    np,
    pl,
    ridge_best_alpha,
    scaler,
    sm,
    transform_coefs_to_original_scale,
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
    _models = {
        "OLS": final_ols,
        "Ridge": final_ridge,
        "Lasso": final_lasso,
        "Elastic Net": final_enet,
    }

    results = []
    final_coefs_original = {}
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
            "Non-zero Features": 12 - _n_zeros,
        })
        # Transform coefficients to original scale for interpretability
        _orig_coefs = transform_coefs_to_original_scale(
            model_params=_model.params,
            scaler=scaler,
            feature_names=feature_columns,
        )
        final_coefs_original[_name] = {
            _f: _orig_coefs[_f] for _f in feature_columns
        }

    results_df = pl.DataFrame(results)
    print("Model Comparison on Test Set:")
    print(results_df)
    return (final_coefs_original,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Coefficient Comparison (Original Scale)

    The coefficients below are transformed back to original scale for interpretability.
    Each coefficient represents the change in sale price (dollars) per one-unit change
    in that feature (e.g., dollars per square foot, dollars per year).
    """)
    return


@app.cell
def _(feature_columns, final_coefs_original, np, plt, sns):
    # Sort features by descending OLS coefficient value (original scale)
    _ols_values = [final_coefs_original["OLS"][_f] for _f in feature_columns]
    _sorted_indices = np.argsort(_ols_values)[::-1]  # Descending order
    _sorted_features = [feature_columns[_i] for _i in _sorted_indices]

    # Create coefficient comparison plot
    _fig, _ax = plt.subplots(figsize=(12, 6))

    _x = np.arange(len(_sorted_features))
    _width = 0.2
    _colors = sns.color_palette("husl", 4)

    for _i, (_name, _coefs) in enumerate(final_coefs_original.items()):
        _values = [_coefs[_f] for _f in _sorted_features]
        _ax.bar(
            _x + _i * _width,
            _values,
            width=_width,
            label=_name,
            color=_colors[_i],
            edgecolor="black",
            linewidth=0.5,
        )

    _ax.set_xlabel("Feature")
    _ax.set_ylabel("Coefficient Value ($/unit)")
    _ax.set_title("Coefficient Comparison Across Regularization Methods (Original Scale)")
    _ax.set_xticks(_x + _width * 1.5)
    _ax.set_xticklabels(_sorted_features, rotation=45, ha="right")
    _ax.legend()
    _ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Regularization Path

    The regularization path shows how coefficients change as the penalty strength increases.
    This visualization helps understand the shrinkage behavior of each method.
    """)
    return


@app.cell
def _(X_train, feature_columns, np, plt, scaler, sm, y_train):
    # Plot regularization paths for Ridge (original scale)
    _alphas = np.logspace(-5, 0, 50)
    _coef_paths = {_f: [] for _f in feature_columns}

    for _alpha in _alphas:
        _model = sm.OLS(endog=y_train, exog=X_train).fit_regularized(
            alpha=_alpha,
            L1_wt=0,
        )
        # Transform to original scale: beta_orig = beta_scaled / std
        for _i, _f in enumerate(feature_columns):
            _coef_orig = _model.params[_i + 1] / scaler.scale_[_i]
            _coef_paths[_f].append(_coef_orig)

    _fig, _ax = plt.subplots(figsize=(10, 5))
    for _f in feature_columns:
        _ax.plot(_alphas, _coef_paths[_f], label=_f)

    _ax.set_xscale("log")
    _ax.set_xlabel("Alpha (log scale)")
    _ax.set_ylabel("Coefficient Value ($/unit)")
    _ax.set_title("Ridge Regularization Path (Original Scale)")
    _ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    _ax.axhline(y=0, color="black", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(X_train, feature_columns, np, plt, scaler, sm, y_train):
    # Plot regularization paths for Lasso (original scale)
    _alphas = np.logspace(-1, 4, 50)
    _coef_paths = {_f: [] for _f in feature_columns}

    for _alpha in _alphas:
        _model = sm.OLS(endog=y_train, exog=X_train).fit_regularized(
            alpha=_alpha,
            L1_wt=1,
        )
        # Transform to original scale: beta_orig = beta_scaled / std
        for _i, _f in enumerate(feature_columns):
            _coef_orig = _model.params[_i + 1] / scaler.scale_[_i]
            _coef_paths[_f].append(_coef_orig)

    _fig, _ax = plt.subplots(figsize=(10, 5))
    for _f in feature_columns:
        _ax.plot(_alphas, _coef_paths[_f], label=_f)

    _ax.set_xscale("log")
    _ax.set_xlabel("Alpha (log scale)")
    _ax.set_ylabel("Coefficient Value ($/unit)")
    _ax.set_title("Lasso Regularization Path (Original Scale)")
    _ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    _ax.axhline(y=0, color="black", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _():
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

    For this Ames Housing dataset with 12 numeric features:
    - **OLS** provides a strong baseline with interpretable coefficients
    - **Ridge** slightly shrinks coefficients but keeps all features
    - **Lasso** can eliminate features, but optimal alpha shows most features are useful
    - **Elastic Net** provides a balanced approach

    The similar performance across methods suggests the selected features are all
    genuinely predictive of house prices, with limited multicollinearity issues.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
