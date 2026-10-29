#!/usr/bin/env python3

'''
Feature extraction module for catpro.
'''

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def tfidf(X_train_sentences, X_test_sentences, ngram_range=(1, 2), max_features=256, min_df=2, max_df=0.8, model='vector', stem=True):
    if stem:
        try:
            import nltk
            from nltk.stem import PorterStemmer
            stemmer = PorterStemmer()
            analyzer = TfidfVectorizer().build_analyzer()
            def stemmed_words(doc):
                return [stemmer.stem(w) for w in analyzer(doc)]
            vec = TfidfVectorizer(analyzer=stemmed_words, ngram_range=ngram_range, max_features=max_features, min_df=min_df, max_df=max_df)
        except Exception:
            vec = TfidfVectorizer(ngram_range=ngram_range, max_features=max_features, min_df=min_df, max_df=max_df)
    else:
        vec = TfidfVectorizer(ngram_range=ngram_range, max_features=max_features, min_df=min_df, max_df=max_df)

    train_tfidf = vec.fit_transform(X_train_sentences).toarray()
    test_tfidf = vec.transform(X_test_sentences).toarray()
    try:
        feature_names = np.array(vec.get_feature_names_out())
    except AttributeError:
        feature_names = np.array(vec.get_feature_names())
    return train_tfidf, test_tfidf, feature_names
