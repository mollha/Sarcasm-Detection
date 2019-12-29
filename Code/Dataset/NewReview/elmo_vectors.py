import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import spacy
import re
import pickle

pd.set_option('display.max_colwidth', 200)
nlp = spacy.load('en_core_web_md', disable=['parser', 'ner'])

# ---------------------- READ IN DATASET ---------------------------
ironic_data = pd.read_csv('Ironic.csv', encoding="ISO-8859-1")
ironic_data['ironic'] = 1
ironic_data["title_and_review"] = ironic_data["title"] + '. ' + ironic_data["review"]
ironic_data = ironic_data.drop(['product', 'stars', 'key', 'date', 'author', 'title', 'review'], axis='columns')

regular_data = pd.read_csv('Regular.csv', encoding="ISO-8859-1")
regular_data['ironic'] = 0
regular_data["title_and_review"] = regular_data["title"] + '. ' + regular_data["review"]
regular_data = regular_data.drop(['product', 'stars', 'key', 'date', 'author', 'title', 'review'], axis='columns')
# combined_data = pd.concat([regular_data, ironic_data])


# ------------------------- CLEAN DATASET -----------------------------
def lemmatization(texts):
    output = []
    for i in texts:
        s = [token.lemma_ for token in nlp(i)]
        output.append(' '.join(s))
    return output


def clean_dataframe(data: pd.DataFrame):
    # remove URLs
    data['title_and_review'] = data['title_and_review'].apply(lambda x: re.sub(r'http\S+', '', x))

    # remove punctuation marks
    punctuation = '!"#$%&()*+-/:;<=>?@[\\]^_`{|}~'
    data["title_and_review"] = data["title_and_review"].apply(
        lambda x: ''.join(ch for ch in x if ch not in set(punctuation)))

    # convert to lowercase
    data["title_and_review"] = data["title_and_review"].str.lower()

    # remove numbers
    data["title_and_review"] = data["title_and_review"].str.replace("[0-9]", " ")

    # remove whitespaces
    data["title_and_review"] = data["title_and_review"].apply(lambda x: ' '.join(x.split()))

    # lemmatize text
    data["title_and_review"] = lemmatization(data["title_and_review"])
    return data


regular_data = clean_dataframe(regular_data)
ironic_data = clean_dataframe(ironic_data)

# function for producing batches of data, with the aim of limiting memory usage
def batch_data(data: pd.DataFrame) -> list:
    return [data[i:i + 1] for i in range(0, data.shape[0])]

# -------------------------------------- VECTORIZE DATA ------------------------------------------
# # Preparing pre-trained ELMo
elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)

def elmo_vectors(x):
    embeddings = elmo(x.tolist(), signature="default", as_dict=True)["elmo"]

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(tf.compat.v1.tables_initializer())
        # return average of ELMo features
        return sess.run(tf.reduce_mean(embeddings,1))

# batch data
regular = batch_data(regular_data)
ironic = batch_data(ironic_data)

# vectorize batched data
regular = [elmo_vectors(x['title_and_review']) for x in regular]
ironic = [elmo_vectors(x['title_and_review']) for x in ironic]

# concatenate batched data back into a single array
regular = np.concatenate(regular, axis = 0)
ironic = np.concatenate(ironic, axis = 0)

# save vectors as pickle files
pickle_out = open("elmo_regular.pickle", "wb")
pickle.dump(regular, pickle_out)
pickle_out.close()
pickle_out = open("elmo_ironic.pickle", "wb")
pickle.dump(ironic, pickle_out)
pickle_out.close()