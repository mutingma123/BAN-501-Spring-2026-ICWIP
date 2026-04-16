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
    from sklearn.metrics import confusion_matrix, roc_auc_score

    sns.set_style("whitegrid")
    return confusion_matrix, mo, np, pl, plt, roc_auc_score, smf, sns


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
        shuffle=True,
    )

    model_data.group_by(
        "y"
    ).len().sort("y")
    return model_data, target_column


@app.cell
def _(model_data, smf, target_column):
    feature_columns = list(set(model_data.columns) - {target_column})

    feature_sum = ' + '.join(feature_columns)
    formula = f'{target_column} ~ {feature_sum}'
    print(formula)

    log_reg = smf.logit(
        data=model_data.to_pandas(),
        formula=formula,
    ).fit()

    print(log_reg.summary())
    return (log_reg,)


@app.cell
def _(log_reg, model_data):
    actuals = model_data['y'].to_numpy()
    predicted_probabilities = log_reg.predict(exog=model_data.to_pandas()).values
    return actuals, predicted_probabilities


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## From Probabilities to Labels

    Logistic regression outputs **probabilities** (values between 0 and 1), but we often need
    binary **labels** (0 or 1) to make decisions. Converting probabilities to labels requires
    choosing a **threshold**.

    - If `P(y=1) >= threshold` → predict 1
    - If `P(y=1) < threshold` → predict 0

    The default threshold of 0.5 seems natural, but it's not always the best choice. The
    threshold you select determines the **types of errors** your model makes:

    - **Lower threshold**: More positive predictions → catches more true positives but also
      more false positives
    - **Higher threshold**: Fewer positive predictions → misses some true positives but has
      fewer false positives

    This is a **business decision**, not a purely statistical one.
    """)
    return


@app.cell
def _(actuals, confusion_matrix, predicted_probabilities):
    # Apply threshold of 0.5 to convert probabilities to labels
    _threshold = 0.5
    _predicted_labels = (predicted_probabilities >= _threshold).astype(int)

    # Compute confusion matrix components
    tn, fp, fn, tp = confusion_matrix(
        y_true=actuals,
        y_pred=_predicted_labels,
    ).ravel()

    # Manual computation of metrics
    accuracy = (_predicted_labels == actuals).mean()
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1 = (2 * recall * precision) / (recall + precision)

    print(f"Threshold: {_threshold}")
    print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"Accuracy:  {accuracy:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"F1 Score:  {f1:.3f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Understanding the Metrics

    The **confusion matrix** breaks down predictions into four categories:

    |                    | Predicted Negative | Predicted Positive |
    |--------------------|-------------------|-------------------|
    | **Actual Negative** | True Negative (TN) | False Positive (FP) |
    | **Actual Positive** | False Negative (FN) | True Positive (TP) |

    From these, we compute:

    - **Accuracy** = (TP + TN) / Total: overall correctness, but misleading with imbalanced classes
    - **Recall** (Sensitivity) = TP / (TP + FN): of all actual positives, how many did we catch?
    - **Precision** = TP / (TP + FP): of all positive predictions, how many were correct?
    - **F1 Score** = 2 × (Precision × Recall) / (Precision + Recall): harmonic mean of precision and recall

    For this bank marketing problem:
    - **High recall** means we contact most potential subscribers (but waste effort on non-subscribers)
    - **High precision** means most people we contact will subscribe (but we miss some potential subscribers)
    """)
    return


@app.cell
def _(actuals, confusion_matrix, np, pl, predicted_probabilities):
    # Compute metrics across a range of thresholds
    _threshold_array = np.arange(0.002, 1.00, 0.002)

    _metrics_list = []
    for _threshold in _threshold_array:
        _predicted_labels = (predicted_probabilities >= _threshold).astype(int)

        _tn, _fp, _fn, _tp = confusion_matrix(
            y_true=actuals,
            y_pred=_predicted_labels,
        ).ravel()

        _recall = _tp / (_tp + _fn) if (_tp + _fn) > 0 else 0
        _precision = _tp / (_tp + _fp) if (_tp + _fp) > 0 else 0
        _f1 = (2 * _recall * _precision) / (_recall + _precision) if (_recall + _precision) > 0 else 0

        _tpr = _tp / (_tp + _fn) if (_tp + _fn) > 0 else 0
        _fpr = _fp / (_tn + _fp) if (_tn + _fp) > 0 else 0

        _metrics_list.append({
            'threshold': round(_threshold, 3),
            'recall': _recall,
            'precision': _precision,
            'f1': _f1,
            'tpr': _tpr,
            'fpr': _fpr,
        })

    metrics_df = pl.DataFrame(_metrics_list)
    metrics_df
    return (metrics_df,)


@app.cell
def _(metrics_df, np, plt, sns):
    _fig, _ax = plt.subplots(figsize=(5, 4))

    # ROC curve from our computed metrics
    sns.lineplot(
        data=metrics_df.to_pandas(),
        x='fpr',
        y='tpr',
        ax=_ax,
        label='ROC Curve',
    )

    # Diagonal reference line (random classifier)
    _diag = np.linspace(0, 1, 100)
    _ax.plot(
        _diag,
        _diag,
        linestyle='--',
        color='gray',
        label='Random Classifier',
    )

    _ax.set_xlim(0, 1.0)
    _ax.set_ylim(0, 1.0)
    _ax.set_xlabel('False Positive Rate')
    _ax.set_ylabel('True Positive Rate')
    _ax.set_title('ROC Curve')
    _ax.legend()

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(actuals, predicted_probabilities, roc_auc_score):
    auc = roc_auc_score(
        y_true=actuals,
        y_score=predicted_probabilities,
    )
    print(f"ROC AUC Score: {auc:.3f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Choosing a Threshold

    The ROC curve shows the trade-off between true positive rate (recall) and false positive rate
    across all possible thresholds. The AUC summarizes overall model performance, but **selecting
    a threshold requires thinking about costs**.

    Consider the bank marketing example:

    | Error Type | What Happens | Cost |
    |------------|--------------|------|
    | False Positive | Call someone who won't subscribe | Wasted call center time |
    | False Negative | Miss a potential subscriber | Lost revenue opportunity |

    **If acquiring new customers is valuable**, you might lower the threshold to catch more
    potential subscribers, accepting that some calls will be wasted.

    **If call center capacity is limited**, you might raise the threshold to focus on the
    highest-probability leads, accepting that you'll miss some subscribers.

    The "best" threshold depends on:
    - Relative costs of each error type
    - Business constraints (budget, capacity)
    - Downstream actions (how expensive is a positive prediction?)

    A threshold of 0.5 is a reasonable default, but it's rarely optimal for any specific
    business problem.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
