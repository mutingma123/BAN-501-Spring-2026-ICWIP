import marimo

__generated_with = "0.19.4"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    import statsmodels.api as sm
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LinearRegression

    sns.set_style("whitegrid")
    return LinearRegression, make_classification, mo, np, plt, sm, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Building Intuition for Logistic Regression

    This notebook develops the mathematical foundations of logistic regression step by step.
    By the end, you'll understand:

    1. Why linear regression fails for binary outcomes
    2. How odds and log-odds transform probabilities
    3. Why the sigmoid function is the natural choice for classification
    4. How to interpret logistic regression coefficients
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Section 1: Why Not Linear Regression for Binary Outcomes?

    Let's start with a simple binary classification problem. We have one continuous
    predictor $x$ and a binary outcome $y \in \{0, 1\}$.

    What happens if we try to fit a linear regression?
    """)
    return


@app.cell
def _(LinearRegression, make_classification, np, plt, sns):
    # Generate binary classification data with 1 feature
    _X, y_binary = make_classification(
        n_samples=200,
        n_features=1,
        n_informative=1,
        n_redundant=0,
        n_clusters_per_class=1,
        flip_y=0.1,
        random_state=42,
    )
    x_feature = _X.flatten()

    # Fit linear regression
    lr_model = LinearRegression()
    lr_model.fit(
        X=_X,
        y=y_binary,
    )

    # Create predictions over extended range
    x_range = np.linspace(
        start=x_feature.min() - 1,
        stop=x_feature.max() + 1,
        num=200,
    )
    y_pred_linear = lr_model.predict(x_range.reshape(-1, 1))

    # Plot
    _fig, _ax = plt.subplots(figsize=(8, 6))

    sns.scatterplot(
        x=x_feature,
        y=y_binary,
        alpha=0.6,
        edgecolor="k",
        ax=_ax,
    )

    _ax.plot(
        x_range,
        y_pred_linear,
        color="red",
        linewidth=2,
        label="Linear regression",
    )

    _ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    _ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5)

    # Shade invalid prediction regions
    _ax.fill_between(
        x_range,
        y_pred_linear,
        1,
        where=(y_pred_linear > 1),
        alpha=0.3,
        color="red",
        label="Predictions > 1",
    )
    _ax.fill_between(
        x_range,
        y_pred_linear,
        0,
        where=(y_pred_linear < 0),
        alpha=0.3,
        color="blue",
        label="Predictions < 0",
    )

    _ax.set_xlabel("Feature (x)")
    _ax.set_ylabel("Outcome (y)")
    _ax.set_title("Linear Regression on Binary Data: The Problem")
    _ax.legend()

    plt.tight_layout()
    plt.gca()
    return x_feature, x_range, y_binary, y_pred_linear


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Problems with linear regression for binary outcomes:**

    1. **Predictions outside [0, 1]**: Linear regression can predict values less than 0
       or greater than 1, which don't make sense as probabilities.

    2. **Wrong functional form**: The relationship between $x$ and $P(Y=1)$ is often
       S-shaped, not linear.

    3. **Heteroscedasticity**: Binary outcomes violate the constant variance assumption.

    We need a model that:
    - Always predicts values between 0 and 1
    - Can capture S-shaped relationships
    - Is appropriate for binary data
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Section 2: From Probability to Odds

    To solve the bounded prediction problem, we'll transform probability through two steps.
    First, let's understand **odds**.

    **Probability**: $p = P(Y = 1)$, bounded to $[0, 1]$

    **Odds**: The ratio of success to failure probability:

    $$\text{odds} = \frac{p}{1-p} = \frac{P(Y=1)}{P(Y=0)}$$

    Odds range from $0$ to $\infty$:
    - $p = 0.5 \Rightarrow \text{odds} = 1$ (even odds)
    - $p = 0.8 \Rightarrow \text{odds} = 4$ (4 to 1 in favor)
    - $p = 0.2 \Rightarrow \text{odds} = 0.25$ (4 to 1 against)
    """)
    return


@app.cell
def _(np, plt):
    # Show the transformation from probability to odds
    p_values = np.linspace(
        start=0.001,
        stop=0.999,
        num=500,
    )
    odds_values = p_values / (1 - p_values)

    _fig, _axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(8, 3),
    )

    # Left plot: probability vs odds (full range)
    _axes[0].plot(
        p_values,
        odds_values,
        linewidth=2,
        color="steelblue",
    )
    _axes[0].set_xlabel("Probability (p)")
    _axes[0].set_ylabel("Odds = p / (1-p)")
    _axes[0].set_title("Probability to Odds Transformation")
    _axes[0].axhline(y=1, color="gray", linestyle="--", alpha=0.5, label="Even odds")
    _axes[0].axvline(x=0.5, color="gray", linestyle="--", alpha=0.5)
    _axes[0].set_ylim(0, 20)
    _axes[0].legend()

    # Right plot: zoomed in around p=0.5
    _axes[1].plot(
        p_values,
        odds_values,
        linewidth=2,
        color="steelblue",
    )
    _axes[1].set_xlabel("Probability (p)")
    _axes[1].set_ylabel("Odds = p / (1-p)")
    _axes[1].set_title("Zoomed View (odds 0-5)")
    _axes[1].axhline(y=1, color="gray", linestyle="--", alpha=0.5)
    _axes[1].axvline(x=0.5, color="gray", linestyle="--", alpha=0.5)
    _axes[1].set_ylim(0, 5)
    _axes[1].set_xlim(0, 0.9)

    plt.tight_layout()
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Key insight about odds:**

    - Odds transform the bounded range $[0, 1]$ to $[0, \infty)$
    - But we still have a lower bound at 0
    - We can't model a linear relationship on $(-\infty, \infty)$ yet

    We need one more transformation...
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Section 3: From Odds to Log-Odds (The Logit)

    The natural logarithm transforms $(0, \infty)$ to $(-\infty, \infty)$:

    $$\text{log-odds} = \log\left(\frac{p}{1-p}\right) = \text{logit}(p)$$

    This is called the **logit function**. It's the key transformation that makes
    logistic regression work.
    """)
    return


@app.cell
def _(np, plt):
    # Show odds to log-odds transformation
    _p_values = np.linspace(
        start=0.001,
        stop=0.999,
        num=500,
    )
    _odds_values = _p_values / (1 - _p_values)
    _log_odds_values = np.log(_odds_values)

    _fig, _axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(8, 3),
    )

    # Left plot: odds vs log-odds
    _axes[0].plot(
        _odds_values,
        _log_odds_values,
        linewidth=2,
        color="steelblue",
    )
    _axes[0].set_xlabel("Odds")
    _axes[0].set_ylabel("Log-odds = log(odds)")
    _axes[0].set_title("Odds to Log-Odds")
    _axes[0].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    _axes[0].axvline(x=1, color="gray", linestyle="--", alpha=0.5)
    _axes[0].set_xlim(0, 10)
    _axes[0].set_ylim(-4, 4)

    # Right plot: probability vs log-odds (the logit function)
    _axes[1].plot(
        _log_odds_values,
        _p_values,
        linewidth=2,
        color="steelblue",
    )
    _axes[1].set_ylabel("Probability (p)")
    _axes[1].set_xlabel("Logit(p) = log(p / (1-p))")
    _axes[1].set_title("The Logit Function")
    _axes[1].axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    _axes[1].axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    _axes[1].set_ylim(-0.1, 1.1)

    plt.tight_layout()
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Key insight about log-odds:**

    | Probability (p) | Odds | Log-odds |
    |:---------------:|:----:|:--------:|
    | 0.01 | 0.01 | -4.6 |
    | 0.10 | 0.11 | -2.2 |
    | 0.25 | 0.33 | -1.1 |
    | 0.50 | 1.00 | 0.0 |
    | 0.75 | 3.00 | 1.1 |
    | 0.90 | 9.00 | 2.2 |
    | 0.99 | 99.0 | 4.6 |

    - Log-odds are **symmetric** around 0
    - $p = 0.5$ corresponds to log-odds = 0
    - Log-odds range from $-\infty$ to $+\infty$
    - Now we can model: $\text{log-odds} = \beta_0 + \beta_1 x$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Section 4: The Logistic (Sigmoid) Function

    We've seen how to go from probability to log-odds. But for predictions, we need
    the **inverse**: from log-odds back to probability.

    The inverse of the logit function is the **logistic function** (also called sigmoid):

    $$\sigma(z) = \frac{1}{1 + e^{-z}} = \frac{e^z}{1 + e^z}$$

    where $z$ represents the log-odds (or any linear combination of predictors).
    """)
    return


@app.cell
def _(np, plt):
    # Show the sigmoid function
    z_values = np.linspace(
        start=-8,
        stop=8,
        num=500,
    )
    sigmoid_values = 1 / (1 + np.exp(-z_values))

    _fig, _ax = plt.subplots(figsize=(6, 4))

    _ax.plot(
        z_values,
        sigmoid_values,
        linewidth=3,
        color="steelblue",
        label=r"$\sigma(z) = \frac{1}{1 + e^{-z}}$",
    )

    # Add reference lines
    _ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    _ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
    _ax.axhline(y=1, color="gray", linestyle="-", alpha=0.3)
    _ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)

    # Annotate key points
    _ax.scatter([0], [0.5], color="red", s=100, zorder=5)
    _ax.annotate(
        "z=0 maps to p=0.5",
        xy=(0, 0.5),
        xytext=(2, 0.3),
        fontsize=10,
        arrowprops=dict(arrowstyle="->", color="red"),
    )

    _ax.set_xlabel("z (log-odds)")
    _ax.set_ylabel(r"$\sigma(z)$ (probability)")
    _ax.set_title("The Logistic (Sigmoid) Function")
    _ax.set_ylim(-0.05, 1.05)
    _ax.legend(fontsize=12)

    plt.tight_layout()
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Properties of the sigmoid function:**

    1. **Bounded output**: Always produces values in $(0, 1)$ - valid probabilities
    2. **S-shaped**: Captures the natural threshold behavior in classification
    3. **Symmetric**: $\sigma(-z) = 1 - \sigma(z)$
    4. **Smooth**: Differentiable everywhere (important for optimization)

    **Key correspondences:**
    - $z = 0 \Rightarrow \sigma(z) = 0.5$
    - $z \to +\infty \Rightarrow \sigma(z) \to 1$
    - $z \to -\infty \Rightarrow \sigma(z) \to 0$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Section 5: Putting It Together - The Logistic Regression Model

    Now we can define the logistic regression model:

    **Model equation (log-odds form):**
    $$\log\left(\frac{p}{1-p}\right) = \beta_0 + \beta_1 x$$

    **Equivalent form (probability):**
    $$P(Y=1|x) = \sigma(\beta_0 + \beta_1 x) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x)}}$$

    This gives us:
    - A linear model in the log-odds space
    - Predictions always between 0 and 1
    - The S-shaped relationship we need
    """)
    return


@app.cell
def _(plt, sm, sns, x_feature, x_range, y_binary, y_pred_linear):
    # Fit logistic regression using statsmodels
    _X_with_const = sm.add_constant(x_feature)
    logit_model = sm.Logit(
        endog=y_binary,
        exog=_X_with_const,
    )
    logit_result = logit_model.fit(disp=0)

    # Get predictions
    _X_range_const = sm.add_constant(x_range)
    y_pred_logistic = logit_result.predict(_X_range_const)

    # Plot comparison
    _fig, _axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(8, 4),
    )

    # Left: Linear regression
    sns.scatterplot(
        x=x_feature,
        y=y_binary,
        alpha=0.6,
        edgecolor="k",
        ax=_axes[0],
    )
    _axes[0].plot(
        x_range,
        y_pred_linear,
        color="red",
        linewidth=2,
        label="Linear regression",
    )
    _axes[0].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    _axes[0].axhline(y=1, color="gray", linestyle="--", alpha=0.5)
    _axes[0].set_xlabel("Feature (x)")
    _axes[0].set_ylabel("Outcome / Predicted probability")
    _axes[0].set_title("Linear Regression")
    _axes[0].set_ylim(-0.3, 1.3)
    _axes[0].legend()

    # Right: Logistic regression
    sns.scatterplot(
        x=x_feature,
        y=y_binary,
        alpha=0.6,
        edgecolor="k",
        ax=_axes[1],
    )
    _axes[1].plot(
        x_range,
        y_pred_logistic,
        color="green",
        linewidth=2,
        label="Logistic regression",
    )
    _axes[1].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    _axes[1].axhline(y=1, color="gray", linestyle="--", alpha=0.5)
    _axes[1].set_xlabel("Feature (x)")
    _axes[1].set_ylabel("Outcome / Predicted probability")
    _axes[1].set_title("Logistic Regression")
    _axes[1].set_ylim(-0.3, 1.3)
    _axes[1].legend()

    plt.tight_layout()
    plt.gca()
    return (logit_result,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Comparison:**

    - **Linear regression** (left): Predictions extend beyond [0, 1], straight line
    - **Logistic regression** (right): S-curve, predictions bounded to [0, 1]

    The logistic regression captures the natural transition from "mostly 0" to "mostly 1"
    as the feature value increases.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Section 6: Interpreting the Coefficients

    Let's look at our fitted model:
    """)
    return


@app.cell
def _(logit_result, mo, np):
    _beta_0 = logit_result.params[0]
    _beta_1 = logit_result.params[1]
    _odds_ratio = np.exp(_beta_1)
    _p_at_0 = 1 / (1 + np.exp(-_beta_0))
    _p_at_1 = 1 / (1 + np.exp(-(_beta_0 + _beta_1)))
    _delta_p = _p_at_1 - _p_at_0

    mo.md(f"""
    **Fitted model:**

    $$\\log\\left(\\frac{{p}}{{1-p}}\\right) = {_beta_0:.3f} + {_beta_1:.3f} \\cdot x$$

    **Interpreting $\\beta_1 = {_beta_1:.3f}$:**

    1. **Log-odds interpretation**: A one-unit increase in $x$ increases the log-odds of
       $Y=1$ by {_beta_1:.3f}.

    2. **Odds ratio interpretation**: $e^{{\\beta_1}} = e^{{{_beta_1:.3f}}} = {_odds_ratio:.3f}$

       A one-unit increase in $x$ multiplies the odds by {_odds_ratio:.3f}.

       {"This means the odds increase by a factor of " + f"{_odds_ratio:.2f}" if _beta_1 > 0 else "This means the odds decrease (multiply by a factor less than 1)"}.

    3. **Probability interpretation** (at $x=0$):
       - $P(Y=1 | x=0) = {_p_at_0:.3f}$
       - $P(Y=1 | x=1) = {_p_at_1:.3f}$
       - Change in probability = {_delta_p:+.3f}

       Note: Unlike log-odds, the probability change varies depending on the starting value of $x$.

    **Interpreting $\\beta_0 = {_beta_0:.3f}$:**

    When $x = 0$:
    - Log-odds = {_beta_0:.3f}
    - Odds = $e^{{{_beta_0:.3f}}}$ = {np.exp(_beta_0):.3f}
    - Probability = $\\sigma({_beta_0:.3f})$ = {1/(1+np.exp(-_beta_0)):.3f}
    """)
    return


@app.cell
def _(logit_result, np, plt):
    # Visualize coefficient interpretation
    _beta_0 = logit_result.params[0]
    _beta_1 = logit_result.params[1]

    _fig, _axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(8, 3),
    )

    # Left: Show effect on log-odds (linear)
    _x_vals = np.linspace(-3, 3, 100)
    _log_odds = _beta_0 + _beta_1 * _x_vals

    _axes[0].plot(
        _x_vals,
        _log_odds,
        linewidth=2,
        color="steelblue",
    )
    _axes[0].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    _axes[0].axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    _axes[0].set_xlabel("Feature (x)")
    _axes[0].set_ylabel("Log-odds")
    _axes[0].set_title(f"Linear in Log-odds Space\nSlope = {_beta_1:.2f}")

    # Annotate the slope
    _x1, _x2 = 0, 1
    _y1 = _beta_0 + _beta_1 * _x1
    _y2 = _beta_0 + _beta_1 * _x2
    _axes[0].annotate(
        "",
        xy=(_x2, _y2),
        xytext=(_x1, _y1),
        arrowprops=dict(arrowstyle="->", color="red", lw=2),
    )
    _axes[0].annotate(
        f"+1 unit x\n= +{_beta_1:.2f} log-odds",
        xy=(0.5, (_y1 + _y2) / 2),
        xytext=(1.5, _y1),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="red"),
    )

    # Right: Show effect on probability (non-linear)
    _probs = 1 / (1 + np.exp(-_log_odds))

    _axes[1].plot(
        _x_vals,
        _probs,
        linewidth=2,
        color="steelblue",
    )
    _axes[1].axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    _axes[1].axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    _axes[1].set_xlabel("Feature (x)")
    _axes[1].set_ylabel("Probability P(Y=1)")
    _axes[1].set_title("Non-linear in Probability Space")
    _axes[1].set_ylim(0, 1)

    # Annotate the probability change for the same x interval
    _p1 = 1 / (1 + np.exp(-(_beta_0 + _beta_1 * _x1)))  # prob at x=0
    _p2 = 1 / (1 + np.exp(-(_beta_0 + _beta_1 * _x2)))  # prob at x=1
    _axes[1].annotate(
        "",
        xy=(_x2, _p2),
        xytext=(_x1, _p1),
        arrowprops=dict(arrowstyle="->", color="red", lw=2),
    )
    _axes[1].annotate(
        f"+1 unit x\n= +{_p2 - _p1:.2f} prob\n(at x=0)",
        xy=(0.5, (_p1 + _p2) / 2),
        xytext=(1.5, _p1),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="red"),
    )

    plt.tight_layout()
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Key takeaways:**

    1. The coefficient $\beta_1$ has a direct interpretation in log-odds space (linear effect)
    2. The odds ratio $e^{\beta_1}$ is often more intuitive for communication
    3. The effect on probability is non-linear and depends on where you start
       - Near $p=0.5$, changes in $x$ have the largest effect on probability
       - Near $p=0$ or $p=1$, the same change in $x$ has less effect on probability

    This is why we often report odds ratios rather than probability changes - the odds ratio
    is constant regardless of the baseline probability.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    **The path to logistic regression:**

    1. **Problem**: Linear regression produces invalid probabilities for binary outcomes

    2. **Solution - Transform the probability:**
       - Probability $p \in [0, 1]$
       - Odds $= p/(1-p) \in [0, \infty)$
       - Log-odds $= \log(p/(1-p)) \in (-\infty, \infty)$

    3. **Model in log-odds space:**
       $$\text{logit}(p) = \beta_0 + \beta_1 x$$

    4. **Transform back for predictions:**
       $$p = \sigma(\beta_0 + \beta_1 x) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x)}}$$

    5. **Interpretation:**
       - $\beta_1$: change in log-odds per unit increase in $x$
       - $e^{\beta_1}$: multiplicative change in odds per unit increase in $x$
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
