# AUTHOR: Louis Tsiattalou
# DESCRIPTION: Test matcher module.

import unittest
from tfidf_matcher.ngrams import ngrams
from tfidf_matcher.matcher import matcher
# If import flag fails, try removing tfidf_matcher and just do `from ngrams
# import ngrams` and execute using `python -m unittest discover -s tests -t src`

class TestMatcher(unittest.TestCase):

    test_list = ["hello", "theseare", "definitely", "companynames"]
    test_lookup = ["hullo", "company", "finite", "cesare"]

    def test_matcher_output_shape(self):
        """Test that the output of matcher is as expected."""
        for k in range(1, len(self.test_lookup)):
            self.assertEqual(matcher(self.test_list, self.test_lookup, k_matches = k).shape, (len(self.test_list), k+2))

    def test_matcher_outputs_found_in_original(self):
        """TODO Test that all of the matcher outputs are found in the original"""
        pass

if __name__ == "__main__":
    unittest.main()
