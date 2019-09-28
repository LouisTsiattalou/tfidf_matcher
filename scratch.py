#!/usr/bin/env python

"""
TITLE: Scratch
AUTHOR: Louis Tsiattalou
DATE STARTED: Fri Sep 20 19:18:37 2019
GITLAB: https://gitlab.com/LouisTsiattalou/tfidf_matcher
DESCRIPTION:
Querying Factiva for an Explain Query, to see whether it's feasible to
use Factiva data for past work, and Webhose data for future time
"""

# IMPORT PACKAGES ##############################################################

# Utils
import pandas as pd
import numpy as np
import re
import json

# IO
from pathlib import Path
import os

# Matching
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# GLOBAL VARIABLES #############################################################

PROJDIR = Path("c:/Users/tsiattaloul/Documents/Projects/BEISBIP/")
DATADIR = PROJDIR / "data/"

# FUNCTION DECLARATIONS ########################################################

# TF-IDF SCORER ----------------------------------------------------------------
def ngrams(string, n=3):
    """Extract characterwise n-grams for a string"""
    string = re.sub(r'[,-./]|\sBD',r'', string)
    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]


def match_companies(original = [], lookup = [], n_matches = 5):
    """Match two string lists together by closest TF-IDF/KNN match"""

    # Generate Sparse TFIDF matrix from Lookup corpus
    vectorizer = TfidfVectorizer(min_df = 1,
                                 analyzer=ngrams)
    tf_idf_lookup = vectorizer.fit_transform(lookup.str.lower())

    # Fit KNN model to sparse TFIDF matrix generated from Lookup
    nbrs = NearestNeighbors(n_neighbors=n_matches,
                            n_jobs=-1, metric='cosine').fit(tf_idf_lookup)

    # Use nbrs model to obtain nearest matches in lookup dataset. Vectorize first.
    tf_idf_original = vectorizer.transform(original.str.lower())
    distances, indices = nbrs.kneighbors(tf_idf_original)

    # Extract Match Score, Original match item, and Lookup matches from distances,indices
    meta_list= []
    lookup_list= []
    for i,idx in enumerate(indices): # i is 0:len(original), j is list of lists of matches
        metadata = [round(distances[i][0],2), original[i]] # Original match and Match Score
        lookups = [lookup.values[x] for x in idx] # Lookup columns
        meta_list.append(metadata)
        lookup_list.append(lookups)

    # Convert to df
    df_metadata = pd.DataFrame(meta_list, columns=['Match Confidence', 'Original Name'])
    df_lookups = pd.DataFrame(lookup_list,
                              columns=['Lookup ' + str(x+1) for x in range(0,n_matches)])

    # bind columns, transform Match Confidence to {0,1} with 1 a guaranteed match.
    matches = pd.concat([df_metadata, df_lookups], axis = 1)
    matches['Match Confidence'] = abs(matches['Match Confidence'] - 1)

    return matches
