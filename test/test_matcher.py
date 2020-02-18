# AUTHOR: Louis Tsiattalou
# DESCRIPTION: Test matcher module.

import unittest
from tfidf_matcher.ngrams import ngrams
from tfidf_matcher.matcher import matcher
# If import flag fails, try removing tfidf_matcher and just do `from ngrams
# import ngrams` and execute using `python -m unittest discover -s tests -t src`

class TestMatcher(unittest.TestCase):
    def test_matcher_output_shape(self):
        """TODO Test that the output of matcher is as expected."""
        pass

    def test_matcher_outputs_found_in_original(self):
        """TODO"""
        pass

if __name__ == "__main__":
    unittest.main()
