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
    return (
        accuracy_score,
        confusion_matrix,
        f1_score,
        mo,
        np,
        pl,
        plt,
        precision_score,
        recall_score,
        roc_auc_score,
        roc_curve,
        smf,
        sns,
    )


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

    We sample 10,000 rows (stratified by target) for faster demonstration.
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
    return (model_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Fitting the Logistic Regression Model

    We use `statsmodels.formula.api.logit()` to fit a logistic regression model.

    The formula API allows us to specify the model using R-style formulas:
    - Numeric features are used as-is
    - Categorical features are automatically converted to dummy variables

    The model estimates coefficients that maximize the **likelihood** of observing the training data.
    """)
    return


@app.cell
def _(model_data, smf):
    # Define the formula
    formula = "y ~ age + balance + duration + campaign + marital + education + housing"

    # Fit logistic regression model
    logit_model = smf.logit(
        formula=formula,
        data=model_data.to_pandas(),
    )
    logit_results = logit_model.fit()

    print(logit_results.summary())
    return (logit_results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Interpreting Logistic Regression Coefficients

    Logistic regression coefficients are in **log-odds** (logit) scale:

    $$\log\left(\frac{P(Y=1)}{1-P(Y=1)}\right) = \beta_0 + \beta_1 x_1 + \ldots$$

    **Interpretation of a coefficient $\beta_j$:**
    - A one-unit increase in $x_j$ changes the **log-odds** by $\beta_j$ (holding other variables constant)
    - The **odds ratio** is $e^{\beta_j}$: the multiplicative change in odds for a one-unit increase

    **Odds ratio interpretation:**
    - OR = 1: No effect
    - OR > 1: Increases odds of Y=1
    - OR < 1: Decreases odds of Y=1

    For example, OR = 1.5 means the odds increase by 50% for each one-unit increase in the predictor.
    """)
    return


@app.cell
def _(logit_results, np, pl):
    # Create odds ratio table with confidence intervals
    _params = logit_results.params
    _conf_int = logit_results.conf_int()
    _pvalues = logit_results.pvalues

    odds_ratio_data = []
    for _name in _params.index:
        _coef = _params[_name]
        _or = np.exp(_coef)
        _ci_lower = np.exp(_conf_int.loc[_name, 0])
        _ci_upper = np.exp(_conf_int.loc[_name, 1])
        _pval = _pvalues[_name]

        odds_ratio_data.append({
            "Variable": _name,
            "Coefficient": _coef,
            "Odds Ratio": _or,
            "95% CI Lower": _ci_lower,
            "95% CI Upper": _ci_upper,
            "P-value": _pval,
        })

    odds_ratio_df = pl.DataFrame(odds_ratio_data)
    print("Odds Ratios with 95% Confidence Intervals:")
    print(odds_ratio_df)
    return (odds_ratio_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Interpreting Specific Coefficients

    **Duration (contact duration in seconds):**
    - Each additional second of call duration increases the odds of subscription
    - This is expected: longer calls indicate more engagement with the offer

    **Marital status:**
    - The reference category is "divorced"
    - Married and single individuals may have different odds compared to divorced

    **Housing loan:**
    - Having a housing loan (yes vs no) affects subscription odds
    - Clients with housing loans may have less disposable income for deposits

    **Note:** Duration is a "leaky" feature because it's only known after the call ends.
    In a real prediction scenario, you wouldn't know this before making the call.
    """)
    return


@app.cell
def _(np, odds_ratio_df, pl, plt):
    # Forest plot of odds ratios
    _plot_data = odds_ratio_df.filter(pl.col("Variable") != "Intercept")

    _fig, _ax = plt.subplots(figsize=(6, 4))

    _variables = _plot_data["Variable"].to_list()
    _odds_ratios = _plot_data["Odds Ratio"].to_list()
    _ci_lower = _plot_data["95% CI Lower"].to_list()
    _ci_upper = _plot_data["95% CI Upper"].to_list()

    _y_pos = np.arange(len(_variables))

    # Plot odds ratios as points
    _ax.scatter(
        _odds_ratios,
        _y_pos,
        s=80,
        color="steelblue",
        zorder=3,
    )

    # Plot confidence intervals as horizontal lines
    for _i, (_lower, _upper) in enumerate(zip(_ci_lower, _ci_upper)):
        _ax.hlines(
            y=_i,
            xmin=_lower,
            xmax=_upper,
            color="steelblue",
            linewidth=2,
        )

    # Reference line at OR = 1
    _ax.axvline(
        x=1,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label="OR = 1 (no effect)",
    )

    _ax.set_yticks(_y_pos)
    _ax.set_yticklabels(_variables)
    _ax.set_xlabel("Odds Ratio (log scale)")
    _ax.set_title("Odds Ratios with 95% Confidence Intervals")
    _ax.set_xscale("log")
    _ax.legend(loc="upper right")

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model Evaluation Metrics

    For binary classification, accuracy alone can be misleading (especially with imbalanced classes).
    We use several complementary metrics:

    | Metric | Formula | What it measures |
    |--------|---------|------------------|
    | **Accuracy** | (TP + TN) / Total | Overall correctness |
    | **Precision** | TP / (TP + FP) | Of predicted positives, how many are correct? |
    | **Recall** | TP / (TP + FN) | Of actual positives, how many did we find? |
    | **F1 Score** | 2 × (Precision × Recall) / (Precision + Recall) | Harmonic mean of precision and recall |

    Where:
    - TP = True Positives, TN = True Negatives
    - FP = False Positives, FN = False Negatives
    """)
    return


@app.cell
def _(confusion_matrix, logit_results, model_data, plt, sns):
    # Make predictions on test set
    y_test = model_data["y"].to_numpy()
    y_pred_prob = logit_results.predict(model_data.to_pandas())
    y_pred = (y_pred_prob >= 0.5).astype(int)

    # Confusion matrix
    _cm = confusion_matrix(
        y_true=y_test,
        y_pred=y_pred,
    )

    _fig, _ax = plt.subplots(figsize=(4, 4))

    sns.heatmap(
        _cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=_ax,
        xticklabels=["No (0)", "Yes (1)"],
        yticklabels=["No (0)", "Yes (1)"],
        cbar=False,
        linewidths=0.1,
        linecolor='k'
    )

    _ax.set_xlabel("Predicted")
    _ax.set_ylabel("Actual")
    _ax.set_title("Confusion Matrix")

    plt.tight_layout()
    plt.show()
    return y_pred, y_pred_prob, y_test


@app.cell
def _(accuracy_score, f1_score, precision_score, recall_score, y_pred, y_test):
    # Calculate evaluation metrics
    _accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
    _precision = precision_score(y_true=y_test, y_pred=y_pred)
    _recall = recall_score(y_true=y_test, y_pred=y_pred)
    _f1 = f1_score(y_true=y_test, y_pred=y_pred)

    print("Model Evaluation Metrics (threshold = 0.5):")
    print("-" * 40)
    print(f"Accuracy:  {_accuracy:.4f}")
    print(f"Precision: {_precision:.4f}")
    print(f"Recall:    {_recall:.4f}")
    print(f"F1 Score:  {_f1:.4f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## ROC Curve and AUC

    The **ROC (Receiver Operating Characteristic) curve** shows the tradeoff between:
    - **True Positive Rate (Recall)**: TP / (TP + FN)
    - **False Positive Rate**: FP / (FP + TN)

    at different classification thresholds.

    The **AUC (Area Under the Curve)** summarizes model performance:
    - AUC = 0.5: Random guessing (diagonal line)
    - AUC = 1.0: Perfect classification
    - AUC > 0.7: Generally considered acceptable
    - AUC > 0.8: Good discrimination
    - AUC > 0.9: Excellent discrimination
    """)
    return


@app.cell
def _(plt, roc_auc_score, roc_curve, y_pred_prob, y_test):
    # Calculate ROC curve and AUC
    _fpr, _tpr, _thresholds = roc_curve(
        y_true=y_test,
        y_score=y_pred_prob,
    )
    _auc = roc_auc_score(
        y_true=y_test,
        y_score=y_pred_prob,
    )

    _fig, _ax = plt.subplots(figsize=(5, 4))

    # ROC curve
    _ax.plot(
        _fpr,
        _tpr,
        color="steelblue",
        linewidth=2,
        label=f"ROC curve (AUC = {_auc:.3f})",
    )

    # Diagonal reference line (random classifier)
    _ax.plot(
        [0, 1],
        [0, 1],
        color="gray",
        linestyle="--",
        label="Random classifier (AUC = 0.5)",
    )

    _ax.set_xlabel("False Positive Rate")
    _ax.set_ylabel("True Positive Rate (Recall)")
    _ax.set_title("ROC Curve")
    _ax.legend(loc="lower right")
    _ax.set_xlim(-0.02, 1.02)
    _ax.set_ylim(-0.02, 1.02)

    plt.tight_layout()
    plt.show()

    print(f"AUC Score: {_auc:.4f}")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
