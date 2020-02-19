# AUTHOR: Louis Tsiattalou
# DESCRIPTION: Test matcher module.

import unittest
from tfidf_matcher.ngrams import ngrams
from tfidf_matcher.matcher import matcher
# If import flag fails, try removing tfidf_matcher and just do `from ngrams
# import ngrams` and execute using `python -m unittest discover -s test -t src`

class TestMatcher(unittest.TestCase):

    test_list = ["hello", "theseare", "definitely", "companynames"]
    test_lookup = ["hullo", "company", "finite", "cesare"]

    def test_matcher_output_shape(self):
        """Test that the output of matcher is as expected."""
        for k in range(1, len(self.test_lookup)):
            self.assertEqual(matcher(self.test_list, self.test_lookup, k_matches = k).shape, (len(self.test_list), k+2))

    def test_matcher_outputs_found_in_original(self):
        """Test that all of the matched strings are found in the original."""
        res = matcher(self.test_list, self.test_lookup, k_matches = 4)['Original Name']
        self.assertTrue(all([x in self.test_list for x in res]))

    def test_matcher_outputs_found_in_lookup(self):
        """Test that all of the matcher lookup outputs are found in the lookups list."""
        res = matcher(self.test_list, self.test_lookup, k_matches = 4)
        res = res.iloc[:,2:]
        reslist = [list(res[x]) for x in res]
        resset = set([x for sublist in reslist for x in sublist]) # Build unique elems in lookup set
        self.assertTrue(all([x in self.test_lookup for x in resset]))

    def test_matcher_scores_normalized(self):
        """Test that the matcher matchscores are between 0 and 1."""
        res = matcher(self.test_list, self.test_lookup, k_matches = 4)['Match Confidence']

if __name__ == "__main__":
    unittest.main()
