# AUTHOR: Louis Tsiattalou
# DESCRIPTION: Test ngrams module.

import unittest
from tfidf_matcher.ngrams import ngrams
# If import flag fails, try removing tfidf_matcher and just do `from ngrams
# import ngrams` and execute using `python -m unittest discover -s tests -t src`

class TestNgrams(unittest.TestCase):

    def test_ngram_count(self):
        """Test that the number of ngrams returned by the function is correct."""
        data = "This is a sentence"
        self.assertEqual(len(ngrams(data, n=1)), 18)
        self.assertEqual(len(ngrams(data, n=2)), 17)
        self.assertEqual(len(ngrams(data, n=3)), 16)
        self.assertEqual(len(ngrams(data, n=18)), 1)

    def test_ngram_length(self):
        """Test that the ngram outputs from the function always contain `n` length ngrams."""
        data = "This is a sentence"
        self.assertTrue(all([len(x) == 1 for x in ngrams(data, n=1)]))
        self.assertTrue(all([len(x) == 2 for x in ngrams(data, n=2)]))
        self.assertTrue(all([len(x) == 3 for x in ngrams(data, n=3)]))
        self.assertTrue(all([len(x) == 18 for x in ngrams(data, n=18)]))

if __name__ == "__main__":
    unittest.main()
