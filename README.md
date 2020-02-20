`tfidf_matcher` is a package for fuzzymatching large datasets together. Most fuzzy
matching libraries like `fuzzywuzzy` get great results, but perform very poorly
due to their O(n^2) complexity.


# How does it work?

This package provides two functions:

-   `ngrams()`: Simple ngram generator.
-   `matcher()`: Matches a list of strings against a reference corpus. Does this by:
    -   Vectorizing the reference corpus using TF-IDF into a term-document matrix.
    -   Fitting a K-NearestNeighbours model to the sparse matrix.
    -   Vectorizing the list of strings to be matched and passing it in to the KNN
        model to calculate the cosine distance (the OOTB `cosine_similarity`
        function in sklearn is very memory-inefficient for our use case).
    -   Some data manipulation to emit `k_matches` closest matches.


# Yeah ok, but how do I use it?

Define two lists; your **original** list (list you want matches for) and your
**lookup** list (list you want to match against). Typically your lookup list will
be much longer than your original list. Pass them into the `matcher` function
along with the number of matches you want to display from the **lookup** list
using the `k_matches` argument. The result will be a pandas DataFrame containing
1 row per item in your **original** list, along with \`k\\\_matches\` columns
containing the closest match from the **lookup** list, and a match score for the
closest match (which is 1 - the cosine distance between the matches normalised
to [0,1])

Simply import with `import tfidf_matcher as tm`, and call the matcher function
with `tm.matcher()`. It takes the following arguments:

-   \`original\`: List of strings you want to match.
-   \`lookup\`: List of strings you want to match against.
-   \`k\_matches\`: Number of the closest results from \`lookup\` to return (1 per column).
-   \`ngram\_length\`: Length of \`ngrams\` used in the algorithm. Anecdotal testing
    shows 2 or 3 to be optimal, but feel free to tinker.


# Strengths and Weaknesses

-   Quick. Very quick.
-   Can emit however many closest matches you want. I found that 3 worked best.
-   Not very well tested so potentially unstable results. Worked well for 640
    company names matched against a lookup corpus of >700,000 company names.
-   It&rsquo;s pretty complicated to get to grips with the method if you wanted to apply
    it in different ways. The underlying algorithms are pretty hard to reason
    about when you jump to the definition of, say, `TfidfVectorizer` from sklearn.
    I *just about* understand the method, which I adapted from [this blog post by
    Josh Taylor](https://towardsdatascience.com/fuzzy-matching-at-scale-84f2bfd0c536), which itself was adapted from another blog post.


# Who do I thank?

As above, credit for the method goes to Josh Taylor and [van den Blog](https://bergvca.github.io/). I wanted
to adapt the methods to work nicely on a company mathcing problem I was having,
and decided to build out my resultant code into a package for two reasons:

1.  Package building experience.
2.  Utility for future projects which may require large-domain fuzzy matching.

