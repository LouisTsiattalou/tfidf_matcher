# AUTHOR: Louis Tsiattalou
# DESCRIPTION: Test ngrams module. Execute with `python -m unittest discover -s test/`

import unittest
from tfidf_matcher.ngrams import ngrams

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
