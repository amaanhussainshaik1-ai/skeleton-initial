# Converts text columns into numeric (TF-IDF) feature vectors.
# Kept separate from preprocessing so the vectorisation strategy
# can be swapped without touching data-loading code.

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from Config import Config


def get_tfidf_embd(df: pd.DataFrame) -> np.ndarray:
    """
    Combine Ticket Summary and Interaction Content, then produce
    TF-IDF embeddings.  Returns a dense numpy array (n_samples x n_features).
    """
    # Concatenate the two text columns into one representation
    text = (df[Config.TICKET_SUMMARY].astype(str) + ' ' +
            df[Config.INTERACTION_CONTENT].astype(str))

    vectorizer = TfidfVectorizer(
        max_features=5000,
        sublinear_tf=True,   # apply log(1+tf) scaling
        strip_accents='unicode',
        analyzer='word',
        token_pattern=r'\w{2,}',  # tokens of at least 2 characters
        ngram_range=(1, 2),       # unigrams and bigrams
        min_df=2,                 # ignore very rare terms
    )

    X = vectorizer.fit_transform(text)
    print(f"[embeddings] TF-IDF matrix shape: {X.shape}")
    return X.toarray()
