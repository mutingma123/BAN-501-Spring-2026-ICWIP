import marimo

__generated_with = "0.19.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    import matplotlib.pyplot as plt
    import pacmap
    import polars as pl
    import seaborn as sns
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.decomposition import PCA

    sns.set_style('whitegrid')
    return PCA, StandardScaler, pacmap, pl, plt, sns, train_test_split


@app.cell
def _(pl, train_test_split):
    feature_data = pl.read_parquet('data/MNIST/mnist_features.parquet')
    target_data = pl.read_parquet('data/MNIST/mnist_target.parquet')

    X_train, X_test, y_train, y_test = train_test_split(
        feature_data, 
        target_data, 
        test_size=5_000, 
        random_state=42,
    )

    feature_data = X_test.to_numpy()
    target_data = y_test.to_numpy()
    return feature_data, target_data


@app.cell
def _(feature_data):
    feature_data.shape
    return


@app.cell
def _(feature_data, plt):
    _idx = 300

    image_array = feature_data[_idx].reshape(28, 28)

    _fig, _ax = plt.subplots(1, 1, figsize=(2, 2))

    _ax.imshow(image_array)

    plt.show()
    return


@app.cell
def _(PCA, StandardScaler, feature_data, plt, sns, target_data):
    scaler = StandardScaler()
    scaler.fit(feature_data)
    scaled_features = scaler.transform(feature_data)

    PCA_model = PCA(n_components=2)
    PCA_model.fit(scaled_features)
    PCA_feature_data = PCA_model.transform(scaled_features)

    _fig, _ax = plt.subplots(1, 1, figsize=(4, 4))

    sns.scatterplot(
        x=PCA_feature_data[:, 0],
        y=PCA_feature_data[:, 1],
        edgecolor='k',
        hue=target_data.flatten(),
    )
    _ax.legend(
        loc='upper left',
        bbox_to_anchor=(1.01, 1.01),
    )
    plt.show()
    return


@app.cell
def _(feature_data, pacmap, plt, sns, target_data):
    pacmap_model = pacmap.PaCMAP(n_components=2)
    pacmap_feature_data = pacmap_model.fit_transform(feature_data)

    _fig, _ax = plt.subplots(1, 1, figsize=(4, 4))

    sns.scatterplot(
        x=pacmap_feature_data[:, 0],
        y=pacmap_feature_data[:, 1],
        edgecolor='k',
        hue=target_data.flatten(),
        palette='tab10',
    )
    _ax.legend(
        loc='upper left',
        bbox_to_anchor=(1.01, 1.01),
    )
    plt.show()
    return (pacmap_feature_data,)


@app.cell
def _(pacmap_feature_data):
    pacmap_feature_data
    return


@app.cell
def _(feature_data, pacmap, plt, sns, target_data):
    localmap_model = pacmap.LocalMAP(n_components=2)
    localmap_feature_data = localmap_model.fit_transform(feature_data)

    _fig, _ax = plt.subplots(1, 1, figsize=(4, 4))

    sns.scatterplot(
        x=localmap_feature_data[:, 0],
        y=localmap_feature_data[:, 1],
        edgecolor='k',
        hue=target_data.flatten(),
        palette='tab10',
    )
    _ax.legend(
        loc='upper left',
        bbox_to_anchor=(1.01, 1.01),
    )
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
