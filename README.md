# Aye-Aye: Semantic Lexicon Induction

A semi-supervised machine learning lexicon inductor! Given just a few examples this algorithm can learn any semantic category

Example: Given the words trout, bass, and salmon, the aye-aye algorithm (based on the [Basilisk](https://www.cs.utah.edu/~riloff/pdfs/emnlp02-thelen.pdf) algorithm) will iteratively expand and find new types of fish such as walleyes or catfish.

Aye-aye leverages the spaCy NLP library to find patterns of words that fit the words that are in the semantic category, filters out false positives, and then securely adds new words to the semantic category until the user judges the category to have been sufficiently covered
