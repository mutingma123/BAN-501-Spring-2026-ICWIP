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
    return mo, pl, plt, smf, sns


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
def _(plt, raw_data, sns):
    # Calculate target distribution
    _target_counts = raw_data.group_by("y").len().sort("y")
    _total = _target_counts["len"].sum()
    _labels = ["No (0)", "Yes (1)"]
    _counts = _target_counts["len"].to_list()
    _percentages = [_c / _total * 100 for _c in _counts]

    _fig, _ax = plt.subplots(figsize=(6, 4))

    _bars = sns.barplot(
        x=_labels,
        y=_counts,
        hue=_labels,
        ax=_ax,
        palette=["steelblue", "coral"],
        edgecolor="black",
        legend=False,
    )

    for _i, (_count, _pct) in enumerate(zip(_counts, _percentages)):
        _ax.text(
            _i,
            _count + _total * 0.01,
            f"{_count:,}\n({_pct:.1f}%)",
            ha="center",
            fontsize=10,
        )

    _ax.set_xlabel("Subscribed to Term Deposit")
    _ax.set_ylabel("Count")
    _ax.set_title("Target Variable Distribution")
    _ax.set_ylim(0, 790_000)

    plt.tight_layout()
    plt.show()
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

    We sample 10,000 rows for faster demonstration.
    """)
    return


@app.cell
def _(raw_data):
    # Define features to use
    numeric_features = ["age", "balance", "duration", "campaign"]
    categorical_features = ["marital", "education", "housing"]
    target_column = "y"


    model_data = raw_data.select(
        numeric_features + categorical_features + [target_column]
    ).sample(
        n=10_000,
        seed=42,
        shuffle=True
    )

    model_data.group_by(
        "y"
    ).len().sort("y")
    return model_data, target_column


@app.cell
def _(model_data, plt, sns):
    _fig, _ax = plt.subplots(1, 1, figsize=(6, 4))

    sns.scatterplot(
        model_data,
        x='duration',
        y='y',
        alpha=0.01,
        edgecolor='k',
    )

    plt.show()
    return


@app.cell
def _(formula):
    formula
    return


@app.cell
def _(model_data, smf, target_column):
    feature_columns = set(model_data.columns) - set([target_column])
    feature_columns = list(feature_columns)

    feature_sum = ' + '.join(feature_columns)
    formula = f'{target_column} ~ {feature_sum}'

    log_reg = smf.logit(
        data=model_data.to_pandas(),
        formula=formula,
    ).fit()

    print(log_reg.summary())
    return formula, log_reg


@app.cell
def _(log_reg, model_data):
    log_reg.predict(model_data.to_pandas())
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
