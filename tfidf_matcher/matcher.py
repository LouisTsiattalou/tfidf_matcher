#!/usr/bin/env python

# AUTHOR: Louis Tsiattalou
# DATE STARTED: Thu Oct 3 23:05:44 2019
# GITLAB: https://gitlab.com/LouisTsiattalou/tfidf_matcher
# DESCRIPTION: Match list items to closest tf-idf match in second list.

import ngrams
import pandas as pd

def matcher(original = [], lookup = [], k_matches = 5):
    """Takes two lists, returns top `k` matches from `lookup` dataset.

    This function does this by:
    - Splitting the `lookup` list into ngrams.
    - Transforming the resulting ngram list into a TF-IDF Sparse Matrix.
    - Fit a NearestNeighbours Model to the matrix using the lookup data.
    - Transform the `original` list into a TF-IDF Sparse Matrix.
    - Calculates distances to all the `n-matches` nearest neighbours
    - Then extract the `original`, `n-matches` closest lookups, and calculate
    a match score (abs(1 - Distance to Nearest Neighbour))

    :param original: List of strings to generate ngrams from.
    :type original: list (of strings)
    :param lookup: List of strings to match against.
    :type lookup: list (of strings)
    :param k_matches: Number of matches to return.
    :type k_matches: int
    ...
    :raises AssertionError: If you pass in a list that contains
    datatypes other than `string`, you're in trouble!
    ...
    :return: Returns list of ngrams generated from the input string.
    :rtype: list
    """


    # Generate Sparse TFIDF matrix from Lookup corpus
    vectorizer = TfidfVectorizer(min_df = 1,
                                 analyzer = ngrams)
    tf_idf_lookup = vectorizer.fit_transform(lookup.str.lower())

    # Fit KNN model to sparse TFIDF matrix generated from Lookup
    nbrs = NearestNeighbors(n_neighbors=k_matches,
                            n_jobs=-1, metric='cosine').fit(tf_idf_lookup)

    # Use nbrs model to obtain nearest matches in lookup dataset. Vectorize first.
    tf_idf_original = vectorizer.transform(original.str.lower())
    distances, indices = nbrs.kneighbors(tf_idf_original)

    # Extract top Match Score (which is just the distance to the nearest neighbour),
    # Original match item, and Lookup matches.
    meta_list= []
    lookup_list= []
    for i,idx in enumerate(indices): # i is 0:len(original), j is list of lists of matches
        metadata = [round(distances[i][0], 2), original[i]] # Original match and Match Score
        lookups = [lookup.values[x] for x in idx] # Lookup columns
        meta_list.append(metadata)
        lookup_list.append(lookups)

    # Convert to df
    df_metadata = pd.DataFrame(meta_list, columns = ['Match Confidence', 'Original Name'])
    df_lookups = pd.DataFrame(lookup_list,
                              columns=['Lookup ' + str(x+1) for x in range(0,k_matches)])

    # bind columns, transform Match Confidence to {0,1} with 1 a guaranteed match.
    matches = pd.concat([df_metadata, df_lookups], axis = 1)
    matches['Match Confidence'] = abs(matches['Match Confidence'] - 1)

    return matches
