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
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    import statsmodels.api as sm

    sns.set_style('whitegrid')
    return StandardScaler, mo, np, pathlib, pl, sm, train_test_split


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Regularized Regression: Ridge, Lasso, and Elastic Net

    This notebook demonstrates regularized regression techniques using the Ames Housing dataset.
    Regularization adds a penalty term to the loss function to prevent overfitting and handle
    multicollinearity.

    **Methods covered:**
    - **Ridge (L2)**: Shrinks coefficients toward zero but rarely sets them exactly to zero
    - **Lasso (L1)**: Can set coefficients exactly to zero, performing feature selection
    - **Elastic Net**: Combines L1 and L2 penalties for a balance of both approaches
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Data Loading and Preprocessing

    We load the Ames Housing dataset and prepare it for modeling:
    1. Drop columns with too many missing values
    2. Impute remaining missing values (median for numeric, mode for categorical)
    3. One-hot encode categorical variables
    """)
    return


@app.cell
def _(pathlib, pl):
    data_filepath = pathlib.Path('data/regression/train.parquet')
    raw_data = pl.read_parquet(data_filepath)

    # Drop columns where more than 25% of values are missing.
    # Columns with too many missing values provide little predictive value
    # and imputation becomes unreliable.
    MISSING_VALUE_THRESHOLD = 0.25

    missing_value_proportions = (
        raw_data.null_count()/len(raw_data)
    ).to_dicts()[0]

    _columns_to_drop = [
        _key for _key, _val in missing_value_proportions.items()
        if _val >= MISSING_VALUE_THRESHOLD
    ]
    raw_data = raw_data.drop(_columns_to_drop)

    # Impute missing numeric values with the column median.
    # Median is robust to outliers compared to mean.
    numeric_fill_list = [
        pl.col(_col).fill_null(pl.col(_col).median())
        for _col in raw_data.columns
        if raw_data[_col].dtype.is_numeric()
    ]
    raw_data = raw_data.with_columns(numeric_fill_list)

    # Impute missing categorical values with the most frequent value (mode).
    categorical_fill_list = [
        pl.col(_col).fill_null(pl.col(_col).mode().first())
        for _col in raw_data.columns
        if not raw_data[_col].dtype.is_numeric()
    ]

    raw_data = raw_data.with_columns(categorical_fill_list)

    # One-hot encode categorical variables.
    # drop_first=True avoids multicollinearity (the "dummy variable trap")
    # by removing one category per variable as the reference level.
    raw_data = raw_data.to_dummies(
        columns = [
            _col for _col in raw_data.columns if not raw_data[_col].dtype.is_numeric()
        ],
        drop_first=True,
    )

    raw_data = raw_data.drop('Id')
    return (raw_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Train/Test Split

    We hold out 20% of the data for testing. The `random_state` ensures reproducibility.
    """)
    return


@app.cell
def _(raw_data, train_test_split):
    TEST_SET_PROPORTION = 0.2

    train_data, test_data = train_test_split(
        raw_data,
        test_size=TEST_SET_PROPORTION,
        random_state=42,
    )
    return test_data, train_data


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Feature Scaling

    Regularized regression penalizes coefficient magnitudes, so features must be on
    the same scale. Otherwise, features with larger numeric ranges would be penalized
    more heavily.

    We use StandardScaler to center features (mean=0) and scale to unit variance (std=1).

    **Note:** The intercept column is excluded from scaling. Scaling would center it to 0,
    effectively removing it from the model.
    """)
    return


@app.cell
def _(StandardScaler, pl, test_data, train_data):
    # Columns to scale: all features except intercept and the target
    _feature_cols = [
        _col for _col in train_data.columns
        if _col not in ('intercept', 'SalePrice')
    ]

    scaler = StandardScaler()

    # Fit on training data, transform both train and test
    _scaled_features_train = scaler.fit_transform(train_data.select(_feature_cols))
    _scaled_features_test = scaler.transform(test_data.select(_feature_cols))

    # Reconstruct DataFrames with scaled features + unscaled intercept and target
    scaled_train_data = pl.DataFrame(
        _scaled_features_train,
        schema=_feature_cols,
    ).with_columns([
        pl.lit(1).alias('intercept'),
        pl.Series('SalePrice', train_data['SalePrice'].to_list()),
    ])

    scaled_test_data = pl.DataFrame(
        _scaled_features_test,
        schema=_feature_cols,
    ).with_columns([
        pl.lit(1).alias('intercept'),
        pl.Series('SalePrice', test_data['SalePrice'].to_list()),
    ])
    return scaled_test_data, scaled_train_data


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Regularization Parameters

    statsmodels uses `fit_regularized()` with two key parameters:

    - **`alpha`**: Regularization strength. Higher values = stronger penalty = smaller coefficients.
      With `alpha=0`, you get ordinary least squares (no regularization).

    - **`L1_wt`**: Weight between L1 (Lasso) and L2 (Ridge) penalties.
      - `L1_wt=0`: Pure Ridge (L2 penalty only)
      - `L1_wt=1`: Pure Lasso (L1 penalty only)
      - `0 < L1_wt < 1`: Elastic Net (mix of both)

    The penalty term added to the loss function is:
    ```
    alpha * (L1_wt * |coefficients| + (1 - L1_wt) * coefficients²)
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Ridge Regression (L1_wt=0)
    """)
    return


@app.cell
def _(pl, scaled_train_data, sm):
    target = 'SalePrice'
    endog = scaled_train_data[target].to_numpy()
    exog = scaled_train_data.select(
        pl.all().exclude(target)
    ).to_numpy()

    # Ridge: L1_wt=0 means pure L2 penalty
    # Shrinks coefficients toward zero but rarely sets them exactly to zero
    ridge_reg = sm.OLS(
        endog=endog,
        exog=exog,
    ).fit_regularized(
        L1_wt=0,  # 0 = Ridge (L2 penalty)
        alpha=10,
    )
    return endog, exog, ridge_reg, target


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Lasso Regression (L1_wt=1)
    """)
    return


@app.cell
def _(endog, exog, sm):
    # Lasso: L1_wt=1 means pure L1 penalty
    # Can shrink coefficients exactly to zero, performing automatic feature selection
    lasso_reg = sm.OLS(
        endog=endog,
        exog=exog,
    ).fit_regularized(
        L1_wt=1,  # 1 = Lasso (L1 penalty)
        alpha=10,
    )
    return (lasso_reg,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Elastic Net (L1_wt=0.5)
    """)
    return


@app.cell
def _(endog, exog, sm):
    # Elastic Net: L1_wt between 0 and 1 combines both penalties
    # Balances Ridge's stability with Lasso's feature selection
    elastic_reg = sm.OLS(
        endog=endog,
        exog=exog,
    ).fit_regularized(
        L1_wt=0.5,  # 0.5 = Equal mix of L1 and L2
        alpha=10,
    )
    return (elastic_reg,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Ordinary Least Squares (No Regularization)
    """)
    return


@app.cell
def _(endog, exog, sm):
    # Standard OLS without regularization for baseline comparison
    ols_reg = sm.OLS(
        endog=endog,
        exog=exog,
    ).fit()
    return (ols_reg,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Coefficient Comparison

    Compare how each regularization method affects the coefficients:
    - Ridge keeps all coefficients non-zero (just shrinks them)
    - Lasso sets many coefficients exactly to zero
    - Elastic Net falls in between
    """)
    return


@app.cell
def _(elastic_reg, lasso_reg, pl, ridge_reg, scaled_train_data, target):
    variable_names = scaled_train_data.select(
        pl.all().exclude(target)
    ).columns

    # Build comparison dataframe
    _coef_data = []
    for _col, _ridge, _lasso, _elastic in zip(
        variable_names,
        ridge_reg.params,
        lasso_reg.params,
        elastic_reg.params,
    ):
        _coef_data.append({
            'predictor': _col,
            'ridge': _ridge,
            'lasso': _lasso,
            'elastic_net': _elastic,
        })

    coef_df = pl.DataFrame(_coef_data)

    # Count non-zero coefficients for each method
    _ridge_nonzero = coef_df.filter(pl.col('ridge').abs() > 1e-10).height
    _lasso_nonzero = coef_df.filter(pl.col('lasso').abs() > 1e-10).height
    _elastic_nonzero = coef_df.filter(pl.col('elastic_net').abs() > 1e-10).height

    print(f"Non-zero coefficients:")
    print(f"  Ridge:       {_ridge_nonzero} / {len(variable_names)}")
    print(f"  Lasso:       {_lasso_nonzero} / {len(variable_names)}")
    print(f"  Elastic Net: {_elastic_nonzero} / {len(variable_names)}")
    return (coef_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Coefficients Lasso Zeroed

    Lasso's L1 penalty drives coefficients to exactly zero, effectively removing features
    from the model. Below are features that Ridge kept but Lasso eliminated:
    """)
    return


@app.cell
def _(coef_df, pl):
    # Show coefficients that Lasso set to zero but Ridge kept
    coef_df.filter(
        (pl.col('lasso').abs() < 1e-10) & (pl.col('ridge').abs() > 1e-10)
    ).select(['predictor', 'ridge', 'lasso']).sort(
        pl.col('ridge').abs(),
        descending=True,
    ).head(10)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Top Predictors by Coefficient Magnitude

    The features with largest absolute coefficients have the strongest linear relationship
    with sale price (after standardization):
    """)
    return


@app.cell
def _(coef_df, pl):
    # Show top 10 predictors by absolute coefficient magnitude (Ridge)
    coef_df.sort(
        pl.col('ridge').abs(),
        descending=True,
    ).head(10)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Test Set Evaluation

    Root Mean Square Error (RMSE) measures prediction accuracy in the same units as the target variable.
    Lower RMSE indicates better predictive performance.
    """)
    return


@app.cell
def _(
    elastic_reg,
    lasso_reg,
    np,
    ols_reg,
    pl,
    ridge_reg,
    scaled_test_data,
    target,
):
    # Extract test features and target
    _test_target = scaled_test_data[target].to_numpy()
    _test_features = scaled_test_data.select(pl.all().exclude(target)).to_numpy()

    # Compute predictions for each model
    _ols_pred = _test_features @ ols_reg.params
    _ridge_pred = _test_features @ ridge_reg.params
    _lasso_pred = _test_features @ lasso_reg.params
    _elastic_pred = _test_features @ elastic_reg.params

    # Calculate RMSE for each model
    _ols_rmse = np.sqrt(((_test_target - _ols_pred) ** 2).mean())
    _ridge_rmse = np.sqrt(((_test_target - _ridge_pred) ** 2).mean())
    _lasso_rmse = np.sqrt(((_test_target - _lasso_pred) ** 2).mean())
    _elastic_rmse = np.sqrt(((_test_target - _elastic_pred) ** 2).mean())

    print("Root Mean Square Error (Test Set):")
    print(f"  OLS:         ${_ols_rmse:,.0f}")
    print(f"  Ridge:       ${_ridge_rmse:,.0f}")
    print(f"  Lasso:       ${_lasso_rmse:,.0f}")
    print(f"  Elastic Net: ${_elastic_rmse:,.0f}")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
