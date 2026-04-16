import marimo

__generated_with = "0.20.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    import pathlib

    import matplotlib.pyplot as plt
    import numpy as np
    import pacmap
    import polars as pl
    import seaborn as sns
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    sns.set_style("whitegrid")
    return (
        SentenceTransformer,
        TfidfVectorizer,
        cosine_similarity,
        mo,
        np,
        pacmap,
        pathlib,
        pl,
        plt,
        sns,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # TF-IDF on Amazon Reviews

    This notebook demonstrates **TF-IDF** (Term Frequency-Inverse Document Frequency), a
    classic technique for converting text into numeric vectors. We use TF-IDF to (i) find
    reviews that are similar to a chosen review using cosine similarity, (ii) inspect the
    vocabulary the model learned, and (iii) visualize the document space in 2D using
    PaCMAP.

    The dataset is a sample of 10,000 Amazon product reviews.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Loading the Data

    We load a parquet file containing 10,000 Amazon reviews. Each row has a `text` column
    holding the review body.
    """)
    return


@app.cell
def _(pathlib, pl):
    data_filepath = pathlib.Path('data/amazon_reviews/amazon_reviews-10000.parquet')
    raw_data = pl.read_parquet(data_filepath)

    raw_data.head()
    return (raw_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Extracting Review Texts

    `TfidfVectorizer` expects an iterable of raw text strings, so we pull the `text`
    column out of the polars DataFrame as a Python list.
    """)
    return


@app.cell
def _(raw_data):
    all_texts = raw_data['text'].to_list()
    all_texts[:5]
    return (all_texts,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Fitting the TF-IDF Model

    `TfidfVectorizer` converts each document into a sparse vector where each entry
    corresponds to a word in the vocabulary. The value combines how often the word
    appears in the document (term frequency) with how rare the word is across the corpus
    (inverse document frequency), so common words get low weights and distinctive words
    get high weights.

    Two settings shape the vocabulary: (i) `min_df=3` ignores words that appear in fewer
    than 3 reviews, which drops typos and one-off terms, and (ii) `stop_words='english'`
    drops common English words like "the", "and", "is".

    The resulting `tfidf_vectors` is a sparse matrix with one row per review and one
    column per vocabulary term.
    """)
    return


@app.cell
def _(TfidfVectorizer, all_texts):
    tfidf_model = TfidfVectorizer(
        min_df=3,
        stop_words='english',
    )
    tfidf_vectors = tfidf_model.fit_transform(raw_documents=all_texts)
    tfidf_vectors.shape
    return tfidf_model, tfidf_vectors


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Picking a Review to Compare Against

    To demonstrate similarity search, we pick a single review from the dataset and use
    it as our query. Below is the text of the review at index 20.
    """)
    return


@app.cell
def _(all_texts):
    test_idx = 20
    test_text = all_texts[test_idx]
    test_text
    return (test_idx,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Computing Cosine Similarity

    **Cosine similarity** measures the angle between two vectors, ignoring their
    magnitudes. A value of 1 means the documents have identical word distributions, and
    0 means they share no words at all. We compute the cosine similarity between our
    query review and every other review in the corpus.
    """)
    return


@app.cell
def _(cosine_similarity, test_idx, tfidf_vectors):
    test_sims = cosine_similarity(
        X=tfidf_vectors[test_idx],
        Y=tfidf_vectors,
    )
    return (test_sims,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Finding the Most Similar Reviews

    We sort the similarity scores and grab the top 6 matches, reversed so the highest
    scores come first. The first match should be the query review itself (similarity of
    1.0), followed by the 5 most similar reviews.
    """)
    return


@app.cell
def _(np, test_sims):
    similar_indices = np.argsort(test_sims.flatten())[-6:][::-1]
    return (similar_indices,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Inspecting the Matches

    Print the query review and its nearest neighbors so we can read them side by side
    and judge whether TF-IDF found genuinely similar reviews.
    """)
    return


@app.cell
def _(all_texts, similar_indices):
    for _idx in similar_indices:
        print(f' - {all_texts[_idx]}')
        print('-'*75)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Inspecting the Vocabulary

    `get_feature_names_out()` returns the list of vocabulary terms the model learned.
    We peek at the first 20 to get a sense of what words made it through the `min_df`
    and stop-word filters.
    """)
    return


@app.cell
def _(tfidf_model):
    tfidf_model.get_feature_names_out()[:20]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Visualizing the Document Space with PaCMAP

    The TF-IDF vectors live in a high-dimensional space (one dimension per vocabulary
    term), which is impossible to plot directly. **PaCMAP** is a non-linear
    dimensionality reduction method that projects the documents into 2D while trying to
    preserve which documents are close to each other in the original space.

    PaCMAP requires a dense matrix, so we convert the sparse TF-IDF output with
    `.toarray()`.
    """)
    return


@app.cell
def _(pacmap, tfidf_vectors):
    pacmap_reducer = pacmap.PaCMAP()
    pacmap_embeddings = pacmap_reducer.fit_transform(X=tfidf_vectors.toarray())
    return (pacmap_embeddings,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2D Scatter of the Reviews

    Each point is a single review. Clusters of nearby points correspond to reviews that
    use similar vocabulary. Without color coding by topic or rating the structure here
    is just suggestive, but you can already see that the reviews are not uniformly
    spread.
    """)
    return


@app.cell
def _(pacmap_embeddings, plt, sns):
    _fig, _ax = plt.subplots(1, 1, figsize=(5, 5))

    sns.scatterplot(
        x=pacmap_embeddings[:, 0],
        y=pacmap_embeddings[:, 1],
        alpha=0.01,
        color='steelblue',
        edgecolor='k',
    )

    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Sentence Transformers

    TF-IDF treats each document as a bag of words: two reviews that say the same thing
    in different words will look completely different. **Sentence transformers** fix this
    by encoding each piece of text into a dense vector that captures semantic meaning.
    Words like "great" and "excellent" end up near each other in the embedding space, even
    though they share no characters.

    We use the `all-MiniLM-L6-v2` model, a lightweight transformer that maps each review
    to a 384-dimensional vector.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Encoding the Reviews

    We encode all 10,000 reviews with the sentence transformer. Because encoding is
    slow, we cache the embeddings (along with their PaCMAP projections) to a parquet
    file so subsequent runs skip the computation.
    """)
    return


@app.cell
def _(SentenceTransformer, all_texts, pacmap, pathlib, pl):
    embeddings_directory = pathlib.Path('embeddings')
    embeddings_directory.mkdir(exist_ok=True)

    st_model_embeddings_filepath = pathlib.Path(
        embeddings_directory,
        'st_embeddings.parquet'
    )
    st_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    if st_model_embeddings_filepath.exists():
        st_embeddings_df = pl.read_parquet(st_model_embeddings_filepath)
    else:
        st_embedding_array = st_model.encode(
            sentences=all_texts,
            batch_size=64,
            show_progress_bar=True,
        )

        _pacmap_reducer = pacmap.PaCMAP()
        _st_pacmap_embeddings = _pacmap_reducer.fit_transform(X=st_embedding_array)

        st_embeddings_df = pl.DataFrame({
            'text': all_texts,
            'st_embeddings': st_embedding_array,
            'pacmap_embeddings': _st_pacmap_embeddings,
        })
        st_embeddings_df.write_parquet(st_model_embeddings_filepath)
    return st_embeddings_df, st_model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2D Scatter of Sentence Transformer Embeddings

    Just as we did with TF-IDF, we project the sentence transformer embeddings into 2D
    with PaCMAP. Compare this plot with the TF-IDF scatter above: because the transformer
    captures semantic similarity rather than word overlap, the clusters here often
    correspond to topics or sentiments rather than shared vocabulary.
    """)
    return


@app.cell
def _(plt, sns, st_embeddings_df):
    _pacmap_st_embeddings = st_embeddings_df['pacmap_embeddings'].to_numpy()

    _fig, _ax = plt.subplots(1, 1, figsize=(5, 5))

    sns.scatterplot(
        x=_pacmap_st_embeddings[:, 0],
        y=_pacmap_st_embeddings[:, 1],
        alpha=0.01,
        color='steelblue',
        edgecolor='k',
    )

    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Similarity Search with Sentence Transformer Embeddings

    We repeat the same cosine-similarity search we ran with TF-IDF, but now using the
    sentence transformer vectors. Because these embeddings encode meaning, the nearest
    neighbors should be semantically similar, even if they use entirely different words.
    """)
    return


@app.cell
def _(all_texts, cosine_similarity, np, st_embeddings_df):
    _st_embeddings = st_embeddings_df['st_embeddings'].to_numpy()

    _test_idx = 1
    _test_text = all_texts[_test_idx]

    _test_sims = cosine_similarity(
        X=_st_embeddings[_test_idx].reshape(1, -1),
        Y=_st_embeddings,
    )

    _similar_indices = np.argsort(_test_sims.flatten())[-6:][::-1]

    for _idx in _similar_indices:
        print(f' - {all_texts[_idx]}')
        print('-'*75)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Querying with a New Sentence

    A key advantage of sentence transformers is that we can encode text that was never in
    the training corpus. Here we write a brand-new sentence, encode it with the same
    model, and find the most similar reviews. TF-IDF could do this too, but the
    transformer will match on meaning rather than exact word overlap.
    """)
    return


@app.cell
def _(all_texts, cosine_similarity, np, st_embeddings_df, st_model):
    my_sentence = "This GPU is very disappointing. It overheats and constantly shuts off. Very disappointed!!!"
    my_sentence_embedding = st_model.encode(
        sentences=[my_sentence],
    )

    _st_embeddings = st_embeddings_df['st_embeddings'].to_numpy()

    _test_sims = cosine_similarity(
        X=my_sentence_embedding,
        Y=_st_embeddings,
    )

    _similar_indices = np.argsort(_test_sims.flatten())[-6:][::-1]

    for _idx in _similar_indices:
        print(f' - {all_texts[_idx]}')
        print('-'*75)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
