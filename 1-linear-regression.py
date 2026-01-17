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
    import statsmodels.formula.api as smf

    sns.set_style('whitegrid')
    return mo, np, pl, plt, smf, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Synthetic Data Generation

    We generate synthetic data with a known linear relationship: $y = mx + b + \epsilon$

    - **True slope** ($m$): 10
    - **True intercept** ($b$): 100
    - **Noise** ($\epsilon$): Uniform random values in $[-50, 50]$

    This lets us verify our optimization finds parameters close to the true values.
    """)
    return


@app.cell
def _(np, plt, sns):
    m_true = 10
    b_true = 100
    N = 30
    jitter = 50

    np.random.seed(42)

    noise = np.random.uniform(
        low=-jitter,
        high=jitter,
        size=N,
    )
    x = list(range(1, N+1))
    x = np.array(x)
    y = m_true*x + b_true + noise

    _fig, _ax = plt.subplots(1, 1, figsize=(4, 3))

    sns.scatterplot(
        x=x,
        y=y,
        edgecolor='k',
    )
    sns.lineplot(
        x=x,
        y=m_true*x + b_true,
    )

    plt.show()
    return x, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Search Neighborhood

    We define a circular neighborhood around the current parameter estimates. At each
    iteration, we evaluate all neighbors on the circle and move to the one with lowest SSE.

    The circle is parameterized by angle $\theta \in [0, 2\pi]$:
    - $\Delta m = r \cos(\theta)$
    - $\Delta b = r \sin(\theta)$

    where $r$ is the step size (radius). Smaller radii give finer-grained search but
    require more iterations to converge.
    """)
    return


@app.cell
def _(np, plt, sns):
    neighbor_radius = 0.01
    num_neighbors = 100

    theta_values = np.linspace(
        start=0, 
        stop=2*np.pi, 
        num=num_neighbors+1
    )
    neighbor_m_values = neighbor_radius*np.cos(theta_values)
    neighbor_b_values = neighbor_radius*np.sin(theta_values)

    _fig, _ax = plt.subplots(1, 1, figsize=(4, 4))

    sns.scatterplot(
        x=neighbor_m_values,
        y=neighbor_b_values,
        edgecolor='k'
    )
    _ax.plot(
        0.0,
        0.0,
        marker="*",
        markersize=25,
        markeredgecolor='k',
        color='yellow',
    )

    plt.show()
    return neighbor_b_values, neighbor_m_values


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Hill-Descending Optimization

    This cell implements a local search algorithm (hill descending) to find optimal
    regression parameters by minimizing Sum of Squared Errors (SSE).

    **Algorithm:**
    1. Start with initial guesses for $\hat{m}$ and $\hat{b}$
    2. Evaluate SSE for all neighbors on the search circle
    3. Move to the neighbor with lowest SSE (if it improves)
    4. Repeat until no neighbor improves the current solution

    This approximates gradient descent: by sampling many directions on a circle,
    we find an approximate descent direction without computing derivatives.
    """)
    return


@app.cell
def _(neighbor_b_values, neighbor_m_values, np, x, y):
    m_hat = 0
    b_hat = y.min()

    y_hat = m_hat*x + b_hat
    sse = ((y - y_hat)**2).sum()
    best_sse = sse

    improvements_made = True
    while improvements_made:
        improvements_made = False
        neighbor_m_array = m_hat + neighbor_m_values
        neighbor_b_array = b_hat + neighbor_b_values

        # Broadcasting: reshape slopes from (101,) to (101, 1) for row-wise multiplication with x
        # reshape intercepts similarly; result is (101, 30) - one row per neighbor, one column per data point
        neighbor_predictions = (
            neighbor_m_array.reshape(-1, 1) * x
            + neighbor_b_array.reshape(-1, 1)
        )

        # Reshape y from (30,) to (1, 30) for element-wise subtraction across all neighbor predictions
        neighbor_errors = y.reshape(1, -1) - neighbor_predictions
        neighbor_squared_errors = neighbor_errors**2
        # Sum across columns (axis=1) to get one SSE value per neighbor
        neighbor_sse = neighbor_squared_errors.sum(axis=1)
        neighbor_min_sse = neighbor_sse.min()
        if  neighbor_min_sse < best_sse:
            improvements_made = True
            min_index = np.argmin(neighbor_sse)
            m_hat = neighbor_m_array[min_index]
            b_hat = neighbor_b_array[min_index]
            best_sse = neighbor_min_sse

    print(f' - {m_hat = :.3f}')
    print(f' - {b_hat = :.3f}')
    print(f' - {best_sse = :.3f}')
    return (best_sse,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model Evaluation: $R^2$

    $R^2$ (coefficient of determination) measures the proportion of variance in $y$
    explained by the model:

    $$R^2 = 1 - \frac{SSE_{model}}{SSE_{mean}}$$

    - $SSE_{model}$: Sum of squared errors from our fitted line
    - $SSE_{mean}$: Sum of squared errors from predicting $\bar{y}$ for all points (baseline)

    $R^2 = 1$ means perfect fit; $R^2 = 0$ means the model is no better than predicting the mean.
    """)
    return


@app.cell
def _(best_sse, y):
    mean_sse = ((y.mean() - y)**2).sum()
    1 - best_sse/mean_sse
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Verification with Statsmodels

    We use `statsmodels` OLS (Ordinary Least Squares) regression to verify our
    manual optimization. OLS has a closed-form solution that finds the optimal
    parameters analytically, so it serves as ground truth.

    The summary includes:
    - **Coefficients**: Should match our $\hat{m}$ and $\hat{b}$
    - **R-squared**: Should match our computed $R^2$
    - **Standard errors** and **p-values**: Statistical significance of coefficients
    """)
    return


@app.cell
def _(pl, smf, x, y):
    reg_data = pl.DataFrame({
        'x': x, 
        'y': y,
    })

    ols_reg = smf.ols(
        data=reg_data,
        formula = 'y ~ x'
    )
    ols_reg = ols_reg.fit()
    print(ols_reg.summary())
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
