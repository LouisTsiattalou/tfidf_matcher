# AUTHOR: Louis Tsiattalou
# DESCRIPTION: Extract ngrams from a string.

import re

def ngrams(string, n=3):
    """Generate a full list of ngrams from a list of strings

    :param string: List of strings to generate ngrams from.
    :type string: list (of strings)
    :param n: Maximum length of the n-gram. Defaults to 3.
    :type n: int
    :raises AssertionError: If you pass in a list that contains datatypes other than `string`, you're in trouble!
    :return: Returns list of ngrams generated from the input string.
    :rtype: list
    """

    # Assert string type
    assert type(string) == type("string"), "String not passed in!"

    # Remove Punctuation from the string
    string = re.sub(r'[,-./]|\sBD',r'', string)

    # Generate zip of ngrams (n defined in function argument)
    ngrams = zip(*[string[i:] for i in range(n)])

    # Return ngram list
    return [''.join(ngram) for ngram in ngrams]
