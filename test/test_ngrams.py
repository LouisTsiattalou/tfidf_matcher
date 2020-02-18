# AUTHOR: Louis Tsiattalou
# DESCRIPTION: Test ngrams module.

import unittest
from tfidf_matcher.ngrams import ngrams
# If import flag fails, try removing tfidf_matcher and just do `from ngrams
# import ngrams` and execute using `python -m unittest discover -s tests -t src`

class TestNgrams(unittest.TestCase):
    def test_ngram_count(self):
        """Test that the number of ngrams returned by the function is correct."""
        data = "Testing Number Of NGrams"
        self.assertEqual(len(ngrams(data)), 22)

    def test_ngram_length(self):
        """Test that the ngram outputs from the function always contain `n` length ngrams."""
        n = 3
        data = "Testing Number Of NGrams"
        self.assertTrue(all([len(x) == n for x in ngrams(data)]))

if __name__ == "__main__":
    unittest.main()
