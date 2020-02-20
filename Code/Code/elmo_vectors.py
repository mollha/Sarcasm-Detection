import pandas as pd
import time
import tensorflow as tf
import tensorflow_hub as hub
import spacy
import numpy as np
import pickle
tf.compat.v1.disable_eager_execution()


pd.set_option('display.max_colwidth', 200)
nlp = spacy.load('en_core_web_md', disable=['parser', 'ner'])

# ---------------------- READ IN DATASET ---------------------------
data = pd.read_csv("Datasets/Sarcasm_Amazon_Review_Corpus/processed_data/OriginalData.csv", encoding="ISO-8859-1")
data['clean_data'] = pd.read_csv("Datasets/Sarcasm_Amazon_Review_Corpus/processed_data/CleanData.csv", encoding="ISO-8859-1")

# ------------------------- BATCH DATASET -----------------------------
# function for producing batches of data, with the aim of limiting memory usage
# def batch_data(data: pd.DataFrame) -> list:
#     return [data[i:i + 1] for i in range(0, data.shape[0])]

# -------------------------------------- VECTORIZE DATA ------------------------------------------
# # Preparing pre-trained ELMo
elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)


# Remember - already dealt with CSV header in CleanData.csv

def elmo_vectors(x, sess):
    embeddings = elmo(x, signature="default", as_dict=True)["elmo"]
    #
    # return average of ELMo features
    return sess.run(tf.reduce_mean(embeddings,1))

# batch data
total_amount = len(data['clean_data'])
print('total amount', total_amount)

# vectorize batched data

initial_time = time.time()

# ----- RESET -----
open("Datasets/Sarcasm_Amazon_Review_Corpus/processed_data/Vectors/elmo_vectors.csv", 'w').close()

csv = open("Datasets/Sarcasm_Amazon_Review_Corpus/processed_data/Vectors/elmo_vectors.csv", "a")


csv.write('vector')

start = 1236

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    sess.run(tf.compat.v1.tables_initializer())
    for cnt, x in enumerate(data['clean_data']):
        if cnt < start:
            continue
        count = cnt + 1
        if count % 5 == 0:
            print('\nProgress: ', str(round(count / total_amount, 3) * 100) + '%')
            time_so_far = round(time.time() - initial_time, 2)
            print('Time: ', time_so_far)
            print('Count: ' + str(count) + '/' + str(total_amount))
            print('Estimated time remaining: ' + str(round(((time_so_far/count)*(total_amount - count))/60, 2)) + ' mins')
        elmo_train = elmo_vectors([token.text for token in nlp(x)], sess)
        elmo_train = np.mean(elmo_train, axis=0)
        csv.write('\n"' + str(list(elmo_train)) + '"')

csv.close()