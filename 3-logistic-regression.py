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
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
        roc_curve,
    )

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
    # Logistic Regression

    ## Why Not Linear Regression for Classification?

    When the outcome is binary (0 or 1), linear regression has problems:
    - Predictions can fall outside [0, 1], which doesn't make sense for probabilities
    - The relationship between features and a binary outcome is often non-linear
    - Linear regression assumes normally distributed errors, which doesn't hold for binary outcomes

    **Logistic regression** solves these issues by modeling the **probability** of the positive class
    using the **sigmoid function**, which always outputs values between 0 and 1.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
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
def _(np, pl, raw_data):
    # Define features to use
    numeric_features = ["age", "balance", "duration", "campaign"]
    categorical_features = ["marital", "education", "housing"]
    target_column = "y"

    # Stratified sample of 10,000 rows
    np.random.seed(42)
    _sample_size = 10000

    # Split by target, sample proportionally
    _class_0 = raw_data.filter(pl.col("y") == 0)
    _class_1 = raw_data.filter(pl.col("y") == 1)

    _n_class_1 = int(_sample_size * len(_class_1) / len(raw_data))
    _n_class_0 = _sample_size - _n_class_1

    _sample_0 = _class_0.sample(
        n=_n_class_0,
        seed=42,
    )
    _sample_1 = _class_1.sample(
        n=_n_class_1,
        seed=42,
    )

    model_data = pl.concat([_sample_0, _sample_1]).sample(
        fraction=1.0,
        seed=42,
    )

    print(f"Sampled data shape: {model_data.shape[0]:,} rows")
    print(f"Target distribution in sample:")
    print(model_data.group_by("y").len().sort("y"))
    return categorical_features, model_data, numeric_features


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exploratory Data Analysis

    Before fitting the model, we examine how features relate to the target variable.
    This helps us understand which features might be predictive of subscription.
    """)
    return


@app.cell
def _(model_data, numeric_features, plt, sns):
    # Boxplots of numeric features by target
    _fig, _axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(8, 6),
    )
    _axes = _axes.flatten()

    for _i, _feature in enumerate(numeric_features):
        sns.boxplot(
            data=model_data.to_pandas(),
            x="y",
            y=_feature,
            hue="y",
            ax=_axes[_i],
            palette=["steelblue", "coral"],
            legend=False,
        )
        _axes[_i].set_xlabel("Subscribed (y)")
        _axes[_i].set_title(f"{_feature} by Subscription Status")

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(categorical_features, model_data, pl, plt, sns):
    # Subscription rates by categorical features
    _fig, _axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(12, 4),
    )

    for _i, _feature in enumerate(categorical_features):
        _grouped = model_data.group_by(_feature).agg([
            pl.col("y").mean().alias("subscription_rate"),
            pl.len().alias("count"),
        ]).sort(_feature)

        sns.barplot(
            data=_grouped.to_pandas(),
            x=_feature,
            y="subscription_rate",
            hue=_feature,
            ax=_axes[_i],
            palette="Blues_d",
            edgecolor="black",
            legend=False,
        )
        _axes[_i].set_xlabel(_feature)
        _axes[_i].set_ylabel("Subscription Rate")
        _axes[_i].set_title(f"Subscription Rate by {_feature}")
        _axes[_i].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Train/Test Split

    We split the data into training (80%) and test (20%) sets:
    - **Training set**: Used to fit the model
    - **Test set**: Used to evaluate model performance on unseen data

    The split is stratified to maintain the same class proportions in both sets.
    """)
    return


@app.cell
def _(model_data, np, pl):
    # Train/test split (stratified)
    np.random.seed(42)

    _class_0 = model_data.filter(pl.col("y") == 0)
    _class_1 = model_data.filter(pl.col("y") == 1)

    _n_train_0 = int(0.8 * len(_class_0))
    _n_train_1 = int(0.8 * len(_class_1))

    _train_0 = _class_0.sample(n=_n_train_0, seed=42)
    _train_1 = _class_1.sample(n=_n_train_1, seed=42)

    _test_0 = _class_0.join(_train_0, on="id", how="anti")
    _test_1 = _class_1.join(_train_1, on="id", how="anti")

    train_data = pl.concat([_train_0, _train_1]).sample(fraction=1.0, seed=42)
    test_data = pl.concat([_test_0, _test_1]).sample(fraction=1.0, seed=42)

    print(f"Training set: {len(train_data):,} rows")
    print(f"Test set: {len(test_data):,} rows")
    print(f"\nTraining target distribution:")
    print(train_data.group_by("y").len().sort("y"))
    return test_data, train_data


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
def _(smf, train_data):
    # Define the formula
    formula = "y ~ age + balance + duration + campaign + marital + education + housing"

    # Fit logistic regression model
    logit_model = smf.logit(
        formula=formula,
        data=train_data.to_pandas(),
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
def _(confusion_matrix, logit_results, plt, sns, test_data):
    # Make predictions on test set
    y_test = test_data["y"].to_numpy()
    y_pred_prob = logit_results.predict(test_data.to_pandas())
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model Summary Statistics

    Logistic regression provides several statistics to assess model fit:

    | Statistic | Description |
    |-----------|-------------|
    | **Log-Likelihood** | Higher (less negative) is better; measures fit to training data |
    | **Pseudo R²** | Analogous to R² in linear regression; measures improvement over null model |
    | **AIC** | Akaike Information Criterion; lower is better; penalizes complexity |
    | **BIC** | Bayesian Information Criterion; lower is better; stronger complexity penalty |

    **McFadden's Pseudo R²:**
    $$R^2_{McFadden} = 1 - \frac{\text{Log-Likelihood (fitted)}}{\text{Log-Likelihood (null)}}$$

    The null model only includes the intercept (predicts the overall mean probability).
    """)
    return


@app.cell
def _(logit_results):
    # Display model summary statistics
    print("Model Summary Statistics:")
    print("-" * 40)
    print(f"Log-Likelihood:     {logit_results.llf:.2f}")
    print(f"Null Log-Likelihood: {logit_results.llnull:.2f}")
    print(f"Pseudo R² (McFadden): {logit_results.prsquared:.4f}")
    print(f"AIC:                {logit_results.aic:.2f}")
    print(f"BIC:                {logit_results.bic:.2f}")
    print(f"\nNumber of observations: {int(logit_results.nobs)}")
    print(f"Degrees of freedom (model): {int(logit_results.df_model)}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Takeaways: Logistic vs Linear Regression

    | Aspect | Linear Regression | Logistic Regression |
    |--------|-------------------|---------------------|
    | **Outcome** | Continuous | Binary (0/1) |
    | **Output** | Predicted value | Probability [0, 1] |
    | **Function** | $y = \beta_0 + \beta_1 x$ | $P(y=1) = \sigma(\beta_0 + \beta_1 x)$ |
    | **Loss function** | Sum of squared errors | Log-likelihood (cross-entropy) |
    | **Coefficients** | Change in y per unit x | Change in log-odds per unit x |
    | **Effect size** | Direct interpretation | Odds ratios ($e^\beta$) |
    | **Model fit** | R², Adjusted R² | Pseudo R², AIC, BIC |
    | **Assumptions** | Normal errors, homoscedasticity | Binary outcome, independence |

    **When to use logistic regression:**
    - Binary outcome variable (yes/no, success/failure, 0/1)
    - Goal is to estimate probability of an event
    - Want interpretable coefficients (odds ratios)
    - Need statistical inference (p-values, confidence intervals)
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
