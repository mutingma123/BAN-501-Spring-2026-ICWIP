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
    return np, plt, sns


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


@app.cell
def _(np, plt, sns):
    neighbor_radius = 1
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

    plt.show()
    return neighbor_b_values, neighbor_m_values


@app.cell
def _(neighbor_b_values, neighbor_m_values, np, x, y):
    m_hat = 0
    b_hat = y.min()

    y_hat = m_hat*x + b_hat
    sse = ((y - y_hat)**2).sum()
    best_sse = sse

    neighbor_min_sse = -np.inf
    while neighbor_min_sse < best_sse:
        neighbor_m_array = m_hat + neighbor_m_values
        neighbor_b_array = b_hat + neighbor_b_values

        neighbor_predictions = (
            neighbor_m_array.reshape(-1, 1) * x 
            + neighbor_b_array.reshape(-1, 1)
        )

        neighbor_errors = y.reshape(1, -1) - neighbor_predictions
        neighbor_squared_errors = neighbor_errors**2
        neighbor_sse = neighbor_squared_errors.sum(axis=1)
        neighbor_min_sse = neighbor_sse.min()
        if  neighbor_min_sse < best_sse:
            min_index = np.argmin(neighbor_sse)
            m_hat = neighbor_m_array[min_index]
            b_hat = neighbor_b_array[min_index]
            best_sse = neighbor_min_sse

    print(f' - {m_hat = :.3f}')
    print(f' - {b_hat = :.3f}')
    print(f' - {best_sse = :.3f}')
    return


@app.cell
def _():
    # m_hat = 0
    # b_hat = y.min()

    # y_hat = m_hat*x + b_hat
    # sse = ((y - y_hat)**2).sum()
    # best_sse = sse

    # improvements_made = True
    # while improvements_made:
    #     improvements_made = False
    #     neighbor_m_array = m_hat + neighbor_m_values
    #     neighbor_b_array = b_hat + neighbor_b_values

    #     neighbor_predictions = (
    #         neighbor_m_array.reshape(-1, 1) * x 
    #         + neighbor_b_array.reshape(-1, 1)
    #     )

    #     neighbor_errors = y.reshape(1, -1) - neighbor_predictions
    #     neighbor_squared_errors = neighbor_errors**2
    #     neighbor_sse = neighbor_squared_errors.sum(axis=1)
    #     neighbor_min_sse = neighbor_sse.min()
    #     if  neighbor_min_sse < best_sse:
    #         improvements_made = True
    #         min_index = np.argmin(neighbor_sse)
    #         m_hat = neighbor_m_array[min_index]
    #         b_hat = neighbor_b_array[min_index]
    #         best_sse = neighbor_min_sse

    # print(f' - {m_hat = :.3f}')
    # print(f' - {b_hat = :.3f}')
    # print(f' - {best_sse = :.3f}')
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
