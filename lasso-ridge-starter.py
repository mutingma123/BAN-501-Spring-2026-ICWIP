import marimo

__generated_with = "0.19.4"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    import pathlib

    import matplotlib.pyplot as plt
    import polars as pl
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    import seaborn as sns

    sns.set_style('whitegrid')
    return pathlib, pl, train_test_split


@app.cell
def _(pathlib, pl):
    data_filepath = pathlib.Path('data/regression/train.parquet')
    raw_data = pl.read_parquet(data_filepath)
    raw_data.shape

    MISSING_VALUE_THRESHOLD = 0.25

    missing_value_proportions = (
        raw_data.null_count()/len(raw_data)
    ).to_dicts()[0]

    _columns_to_drop = [
        _key for _key, _val in missing_value_proportions.items()
        if _val >= MISSING_VALUE_THRESHOLD
    ]
    raw_data = raw_data.drop(_columns_to_drop)

    numeric_fill_list = [
        pl.col(_col).fill_null(pl.col(_col).median())
        for _col in raw_data.columns
        if raw_data[_col].dtype.is_numeric()
    ]
    raw_data = raw_data.with_columns(numeric_fill_list)

    categorical_fill_list = [
        pl.col(_col).fill_null(pl.col(_col).mode().first())
        for _col in raw_data.columns
        if not raw_data[_col].dtype.is_numeric()
    ]

    raw_data = raw_data.with_columns(categorical_fill_list)

    raw_data = raw_data.to_dummies(
        columns = [
            _col for _col in raw_data.columns if not raw_data[_col].dtype.is_numeric()
        ],
        drop_first = True,
    )

    raw_data = raw_data.drop('Id')
    return (raw_data,)


@app.cell
def _(raw_data, train_test_split):
    TEST_SET_PROPORTION = 0.2

    train_data, test_data = train_test_split(
        raw_data, 
        test_size=TEST_SET_PROPORTION,
        random_state=42,
    )

    train_data.shape
    test_data.shape
    return (train_data,)


@app.cell
def _(train_data):
    train_data
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
