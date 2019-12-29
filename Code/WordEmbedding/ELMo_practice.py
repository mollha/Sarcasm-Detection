import tensorflow_hub as hub
import tensorflow as tf
import pandas as pd
import numpy as np
import spacy
from tqdm import tqdm
import re
import time
import pickle
pd.set_option('display.max_colwidth', 200)
nlp = spacy.load('en_core_web_md', disable=['parser', 'ner'])
# Need to disable eager execution



# read data
train = pd.read_csv("Practice_Dataset/train_2kmZucJ.csv")
test = pd.read_csv("Practice_Dataset/test_oJQbWVk.csv")

print(train.shape, test.shape)

# ----------------- CLEANING --------------------
# Normalize the training set
train['label'].value_counts(normalize=True)

# Remove URLs from training and testing set
train['clean_tweet'] = train['tweet'].apply(lambda x: re.sub(r'http\S+', '', x))
test['clean_tweet'] = test['tweet'].apply(lambda x: re.sub(r'http\S+', '', x))

# remove punctuation marks
punctuation = '!"#$%&()*+-/:;<=>?@[\\]^_`{|}~'

train['clean_tweet'] = train['clean_tweet'].apply(lambda x: ''.join(ch for ch in x if ch not in set(punctuation)))
test['clean_tweet'] = test['clean_tweet'].apply(lambda x: ''.join(ch for ch in x if ch not in set(punctuation)))

# convert text to lowercase
train['clean_tweet'] = train['clean_tweet'].str.lower()
test['clean_tweet'] = test['clean_tweet'].str.lower()

# remove numbers
train['clean_tweet'] = train['clean_tweet'].str.replace("[0-9]", " ")
test['clean_tweet'] = test['clean_tweet'].str.replace("[0-9]", " ")

# remove whitespaces
train['clean_tweet'] = train['clean_tweet'].apply(lambda x:' '.join(x.split()))
test['clean_tweet'] = test['clean_tweet'].apply(lambda x: ' '.join(x.split()))

# function to lemmatize text
def lemmatization(texts):
    output = []
    for i in texts:
        s = [token.lemma_ for token in nlp(i)]
        output.append(' '.join(s))
    return output

train['clean_tweet'] = lemmatization(train['clean_tweet'])
test['clean_tweet'] = lemmatization(test['clean_tweet'])
# -----------------------------------------------------------------------------
# Preparing pre-trained ELMo
elmo = hub.Module("https://tfhub.dev/google/elmo/3", trainable=True)

# ELMo practice on a random sentence
# just a random sentence
x = ["Roasted ants are a popular snack in Columbia"]

# Extract ELMo features
embeddings = elmo(x, signature="default", as_dict=True)["elmo"]
# every word in the input sentence has ELMo vector of size 1024


print(embeddings)