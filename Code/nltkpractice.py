from nltk import *
from nltk.book import text1
import matplotlib.pyplot as plt

# print(text1)

sentence = "This is my sentence"

tokens = word_tokenize(sentence)

"""
word_tokenize is better than split() because it retrieves tokens in the linguistic sense 
(i.e. punctuation separated from the rest of the word)
"""

fdist = FreqDist(text1)
V = sorted(map(lambda s: str(s).lower(), set(text1)))
X = [x for x in V if len(x) > 10 and fdist[x] > 10]     # words longer than 10 that occur more than 10 times

Y = list(bigrams(['more', 'is', 'said', 'than', 'done']))   # extracting word pairs (bigrams)
Z = text1.collocation_list()
print(Z)

print(Y)
print(X)

fdist.plot()