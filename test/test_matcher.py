# AUTHOR: Louis Tsiattalou
# DESCRIPTION: Test matcher module. Execute with `python -m unittest discover -s test/`

import re
import unittest

import pandas as pd

from tfidf_matcher.matcher import matcher
from tfidf_matcher.ngrams import ngrams


class TestMatcher(unittest.TestCase):

    test_list = ["hello", "theseare", "definitely", "companynames"]
    test_lookup = ["hullo", "company", "finite", "cesare"]

    max_k_matches = len(test_lookup) - 1

    def test_matcher_output_shape(self):
        """Test that the output of matcher is as expected."""
        for k in range(1, self.max_k_matches + 1):
            self.assertEqual(
                matcher(self.test_list, self.test_lookup, k_matches=k).shape,
                (len(self.test_list), (k * 3) + 1),
            )

    def test_matcher_outputs_found_in_original(self):
        """Test that all of the matched strings are found in the original."""
        res = matcher(self.test_list, self.test_lookup, k_matches=self.max_k_matches)[
            "Original Name"
        ]
        self.assertTrue(all([x in self.test_list for x in res]))

    def test_matcher_outputs_found_in_lookup(self):
        """Test that all of the matcher lookup outputs are found in the lookups list."""
        res = matcher(self.test_list, self.test_lookup, k_matches=self.max_k_matches)
        lookup_cols = [c for c in res.columns if re.match(r"^Lookup \d+$", c)]
        lookup_unique_values = pd.unique(res[lookup_cols].values.ravel("K"))
        self.assertTrue(all([x in self.test_lookup for x in lookup_unique_values]))

    def test_matcher_scores_normalized(self):
        """Test that the matcher matchscores are between 0 and 1."""
        res = matcher(self.test_list, self.test_lookup, k_matches=self.max_k_matches)
        confidence_cols = [
            c for c in res.columns if re.match(r"^Lookup \d+ Confidence$", c)
        ]
        for confidence_col in confidence_cols:
            self.assertTrue(all([0 <= x <= 1 for x in res[confidence_col]]))


if __name__ == "__main__":
    unittest.main()
