# Examples from course.spacy.io/chapter1
# finished part 6, model packages

import spacy
from spacy.lang.en import English

# created by processing a string of text with the nlp object
nlp = English()


# includes language-specific rules for tokenization
# calling nlp on a string causes spacy to tokenize the text and creates a doc object
doc = nlp("Hello world!")

# iterate over tokens in a doc
for token in doc:
    print(token.text)

# index into the doc (behaves like a list)
token = doc[1]

# get token's text via .text
print(token.text)

# a slice from doc is a span object
span = doc[1:4]

print(span.text)


# ---- LEXICAL ATTRIBUTES ----
doc = nlp("It costs me $5.")
print('Index:   ', [token.i for token in doc])
print('Text:    ', [token.text for token in doc])

print('is_alpha:', [token.is_alpha for token in doc])
print('is_punct:', [token.is_punct for token in doc])
print('like_num:', [token.like_num for token in doc])

# ---- STATISTICAL MODELS ----
# allows spacy to predict attributes in context
# trained on labeled data sets

nlp = spacy.load('en_core_web_sm')
# When we load a model into spacy, the labelled data that the model was trained on is not included

# Process a text
doc = nlp("She ate the pizza")

# Iterate over the tokens
for token in doc:
    # Print the text and the predicted part-of-speech tag
    print(token.text, token.pos_, token.dep_, token.head.text)

# nsubj nominal subject
# dobj direct object
# det determiner (article)
print('\n\n')
doc = nlp("Apple is looking at buying U.K. startup for $1 billion")
for ent in doc.ents:
    # Print the entity text and its label
    print(ent.text, ent.label_)

print('\n\n')

# Check for number followed by percentage sign
# Process the text
doc = nlp(
    "In 1990, more than 60% of people in East Asia were in extreme poverty. "
    "Now less than 4% are."
)

# Iterate over the tokens in the doc
for token in doc:
    # Check if the token resembles a number
    if token.like_num:
        # Get the next token in the document
        next_token = doc[token.i + 1]
        # Check if the next token's text equals '%'
        if next_token.text == "%":
            print("Percentage found:", token.text)

# Get definitions of common tags and labels
print(spacy.explain('NNP'))
print(spacy.explain('dobj'))
print(spacy.explain('GPE'))