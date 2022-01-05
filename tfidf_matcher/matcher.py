# AUTHOR: Louis Tsiattalou
# DESCRIPTION: Match list items to closest tf-idf match in second list.

import pandas as pd
from tfidf_matcher.ngrams import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


def matcher(original=[], lookup=[], k_matches=5, ngram_length=3):
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
    :type original: list (of strings), or Pandas Series.
    :param lookup: List of strings to match against.
    :type lookup: list (of strings), or Pandas Series.
    :param k_matches: Number of matches to return.
    :type k_matches: int
    :param ngram_length: Length of Ngrams returned by `tfidf_matcher.ngrams` callable
    :type ngram_length: int
    :raises AssertionError: Throws an error if the datatypes in `original` aren't strings.
    :raises AssertionError: Throws an error if the datatypes in `lookup` aren't strings.
    :raises AssertionError: Throws an error if `k_matches` isn't an integer.
    :raises AssertionError: Throws an error if k_matches > len(lookup)
    :raises AssertionError: Throws an error if ngram_length isn't an integer
    :return: Returns a Pandas dataframe with the `original` list,
        `k_matches` columns containing the closest matches from `lookup`,
        as well as a Match Score for the closest of these matches.
    :rtype: Pandas dataframe
    """

    # Assertions
    assert all([type(x) == type("string")
               for x in original]), "Original contains non-str elements!"
    assert all([type(x) == type("string")
               for x in lookup]), "Lookup contains non-str elements!"
    assert type(k_matches) == type(0), "k_matches must be an integer"
    assert k_matches < len(
        lookup), "k_matches must be shorter than the total length of the lookup list"
    assert type(ngram_length) == type(0), "ngram_length must be an integer"

    # Enforce listtype, set to lower
    original = list(original)
    lookup = list(lookup)
    original_lower = [x.lower() for x in original]
    lookup_lower = [x.lower() for x in lookup]

    # Set ngram length for TfidfVectorizer callable
    def ngrams_user(string, n=ngram_length):
        return ngrams(string, n)

    # Generate Sparse TFIDF matrix from Lookup corpus
    vectorizer = TfidfVectorizer(min_df=1,
                                 analyzer=ngrams_user)
    tf_idf_lookup = vectorizer.fit_transform(lookup_lower)

    # Fit KNN model to sparse TFIDF matrix generated from Lookup
    nbrs = NearestNeighbors(n_neighbors=k_matches,
                            n_jobs=-1, metric='cosine').fit(tf_idf_lookup)

    # Use nbrs model to obtain nearest matches in lookup dataset. Vectorize first.
    tf_idf_original = vectorizer.transform(original_lower)
    distances, lookup_indices = nbrs.kneighbors(tf_idf_original)

    # Extract top Match Score (which is just the distance to the nearest neighbour),
    # Original match item, and Lookup matches.
    original_name_list = []
    confidence_list = []
    index_list = []
    lookup_list = []
    # i is 0:len(original), j is list of lists of matches
    for i, lookup_index in enumerate(lookup_indices):
        original_name = original[i]
        # lookup names in lookup list
        lookups = [lookup[index] for index in lookup_index]
        # transform distances to confidences and store
        confidence = [1 - round(dist, 2) for dist in distances[i]]
        original_name_list.append(original_name)
        # store index
        index_list.append(lookup_index)
        confidence_list.append(confidence)
        lookup_list.append(lookups)

    # Convert to df
    df_orig_name = pd.DataFrame(
        original_name_list, columns=['Original Name'])

    df_lookups = pd.DataFrame(lookup_list,
                              columns=['Lookup ' + str(x+1) for x in range(0, k_matches)])
    df_confidence = pd.DataFrame(confidence_list,
                                 columns=['Lookup ' + str(x+1) + ' Confidence' for x in range(0, k_matches)])
    df_index = pd.DataFrame(index_list,
                            columns=['Lookup ' + str(x+1) + ' Index' for x in range(0, k_matches)])

    # bind columns
    matches = pd.concat(
        [df_orig_name, df_lookups, df_confidence, df_index], axis=1)

    # reorder columns | can be skipped
    lookup_cols = list(matches.columns.values)
    lookup_cols_reordered = [lookup_cols[0]]
    for i in range(1, k_matches+1):
        lookup_cols_reordered.append(lookup_cols[i])
        lookup_cols_reordered.append(lookup_cols[i + k_matches])
        lookup_cols_reordered.append(lookup_cols[i + 2 * k_matches])
    matches = matches[lookup_cols_reordered]
    return matches
