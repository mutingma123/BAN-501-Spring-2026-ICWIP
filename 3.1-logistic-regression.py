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
    import statsmodels.formula.api as smf
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
        roc_curve,
    )
    from sklearn.model_selection import train_test_split

    sns.set_style("whitegrid")
    return mo, np, pl, plt, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Logistic Regression

    ## Why Not Linear Regression for Classification?

    When the outcome is binary (0 or 1), linear regression has problems:
    - Predictions can fall outside [0, 1], which doesn't make sense for probabilities
    - The relationship between features and a binary outcome is often non-linear
    - Linear regression assumes normally distributed errors, which doesn't hold for binary outcomes

    **Logistic regression** solves these issues by modeling the **probability** of the positive class
    using the **sigmoid function**, which always outputs values between 0 and 1.

    ## The Sigmoid Function

    The sigmoid (logistic) function transforms any real number into a probability:

    $$\sigma(z) = \frac{1}{1 + e^{-z}}$$

    where $z = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \ldots + \beta_p x_p$ (the linear combination of features).

    Properties:
    - As $z \to +\infty$, $\sigma(z) \to 1$
    - As $z \to -\infty$, $\sigma(z) \to 0$
    - At $z = 0$, $\sigma(z) = 0.5$

    The sigmoid function creates the characteristic S-shaped curve that maps the linear predictor
    to a probability.
    """)
    return


@app.cell
def _(np, plt, sns):
    # Visualize the sigmoid function
    _z = np.linspace(
        start=-10,
        stop=10,
        num=200,
    )
    _sigmoid = 1 / (1 + np.exp(-_z))

    _fig, _ax = plt.subplots(
        figsize=(8, 4),
    )

    sns.lineplot(
        x=_z,
        y=_sigmoid,
        ax=_ax,
        linewidth=2,
    )

    _ax.axhline(
        y=0.5,
        color="gray",
        linestyle="--",
        alpha=0.7,
    )
    _ax.axvline(
        x=0,
        color="gray",
        linestyle="--",
        alpha=0.7,
    )

    _ax.set_xlabel("z (linear predictor)")
    _ax.set_ylabel("σ(z) = P(Y=1)")
    _ax.set_title("The Sigmoid Function")
    _ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.show()
    return


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
    ## Target Variable Distribution

    The target variable `y` indicates whether the client subscribed to a term deposit:
    - `0`: Did not subscribe (majority class)
    - `1`: Subscribed (minority class)

    Understanding the class balance is crucial because:
    - A naive model predicting always "no" would achieve ~88% accuracy
    - We need metrics beyond accuracy to evaluate model performance
    """)
    return


@app.cell
def _(raw_data):
    # Calculate target distribution
    _target_counts = raw_data.group_by("y").len().sort("y")
    _target_counts
    return


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

    We sample 10,000 rows (stratified by target) for faster demonstration.
    """)
    return


@app.cell
def _():
    # Define features to use
    numeric_features = ["age", "balance", "duration", "campaign"]
    categorical_features = ["marital", "education", "housing"]
    target_column = "y"

    _sample_size = 10000
    return


if __name__ == "__main__":
    app.run()
