# Import our dependencies
import tensorflow as tf
import pandas as pd
import tensorflow_hub as hub
from Code.DataPreprocessing import *
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import os
import re
from sklearn.metrics import f1_score, accuracy_score
from random import randint
from keras import backend as K
from keras.layers import Dense, Input,  Embedding
from keras.models import Model, load_model, Sequential
from keras.engine import Layer
from sklearn.model_selection import train_test_split
from keras.initializers import Constant

import numpy as np

# Initialize session
sess = tf.Session()
K.set_session(sess)

dataset_paths = ["Datasets/Sarcasm_Amazon_Review_Corpus", "Datasets/news-headlines-dataset-for-sarcasm-detection"]

# Choose a dataset from the list of valid data sets
path_to_dataset_root = dataset_paths[1]
print('Selected dataset: ' + path_to_dataset_root[9:])

# Read in raw data
data = pd.read_csv(path_to_dataset_root + "/processed_data/OriginalData.csv", encoding="ISO-8859-1")[:10000]


def get_clean_data_col(data_frame: pd.DataFrame, path_to_dataset_root: str, re_clean: bool,
                       extend_path='') -> pd.DataFrame:
    """
    Retrieve the column of cleaned data -> either by cleaning the raw data, or by retrieving pre-cleaned data
    :param data_frame: data_frame containing a 'text_data' column -> this is the raw textual data
    :param re_clean: boolean flag -> set to True to have the data cleaned again
    :param extend_path: choose to read the cleaned data at an extended path -> this is not the default clean data
    :return: a pandas DataFrame containing cleaned data
    """
    if re_clean:
        input_data = ''
        while not input_data:
            input_data = input('\nWARNING - This action could overwrite pre-cleaned data: proceed? y / n\n')
            input_data = input_data.strip().lower() if input_data.strip().lower() in {'y', 'n'} else ''

        if input_data == 'y':
            # This could potentially overwrite pre-cleaned text if triggered accidentally
            # The process of cleaning data can take a while, so -> proceed with caution
            print('RE-CLEANING ... PROCEED WITH CAUTION!')
            exit()  # uncomment this line if you would still like to proceed
            data_frame['clean_data'] = data_frame['text_data'].apply(data_cleaning)
            extend_path = '' if not os.path.isfile(path_to_dataset_root + "/processed_data/CleanData.csv") else \
                ''.join([randint(0, 9) for _ in range(0, 8)])
            data_frame['clean_data'].to_csv(
                path_or_buf=path_to_dataset_root + "/processed_data/CleanData" + extend_path + ".csv",
                index=False, header=['clean_data'])
    return pd.read_csv(path_to_dataset_root + "/processed_data/CleanData" + extend_path + ".csv",
                       encoding="ISO-8859-1")[:10000]


# Clean data, or retrieve pre-cleaned data
data['clean_data'] = get_clean_data_col(data, path_to_dataset_root, False)


class MultihotEmbedding(Layer):

    def __init__(self, vocab_size, **kwargs):
        self.trainable = False
        self.vocab_size = vocab_size
        super(MultihotEmbedding, self).__init__(**kwargs)

    def call(self, x, mask=None):
        self.get_embeddings = K.one_hot(x, num_classes=self.vocab_size)
        self.reduce_embeddings = K.sum(self.get_embeddings,axis = -2)
        return self.reduce_embeddings

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.vocab_size


class ElmoEmbeddingLayer(Layer):
    def __init__(self, **kwargs):
        self.dimensions = 1024
        self.trainable = True
        super(ElmoEmbeddingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.elmo = hub.Module('https://tfhub.dev/google/elmo/2', trainable=self.trainable,
                               name="{}_module".format(self.name))

        self.trainable_weights += tf.compat.v1.trainable_variables(scope="^{}_module/.*".format(self.name))
        super(ElmoEmbeddingLayer, self).build(input_shape)

    def call(self, x, mask=None):
        # this is where layer's logic lives, the forward function
        print(x.shape)
        result = self.elmo(K.squeeze(K.cast(x, tf.string), axis=1),
                           as_dict=True,
                           signature='default',
                           )['default']
        print(result.shape)
        return result

    def compute_mask(self, inputs, mask=None):
        return K.not_equal(inputs, '--PAD--')

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.dimensions)

#
#  GloveEmbeddingLayer(input_shape=(1,), input_length= 1000, output_dim=50, trainable=False)
# embedding_layer = Embedding(num_words,
#                             EMBEDDING_DIM,
#                             embeddings_initializer=Constant(embedding_matrix),
#                             input_length=MAX_SEQUENCE_LENGTH,
#                             trainable=False)


class GloveEmbeddingLayer(Embedding):
    def __init__(self, tokeniser, max_num_words):
        self.MAX_SEQUENCE_LENGTH = 1000
        self.MAX_NUM_WORDS = max_num_words
        self.output_dim = 50
        self.embeddings_matrix = self.compute_embedding_matrix(self.MAX_NUM_WORDS, tokeniser)
        print(self.embeddings_matrix)
        super(GloveEmbeddingLayer, self).__init__(input_dim=min(self.MAX_NUM_WORDS, len(tokeniser.word_index) + 1),
                                                  output_dim=50,
                                                  embeddings_initializer=Constant(self.embeddings_matrix),
                                                  input_shape=(self.MAX_SEQUENCE_LENGTH,),
                                                  trainable=False
                                                  )

    def compute_embedding_matrix(self, vocab_size, tokeniser):
        with open('Datasets/GLOVEDATA/glove.twitter.27B.50d.txt', "r", encoding="utf-8") as file:
            embeddings_index = {line.split()[0]: list(map(float, line.split()[1:])) for line in file}
            del embeddings_index['0.45973']  # for some reason, this entry has 49 dimensions instead of 50

        embedding_matrix = np.zeros((vocab_size, self.output_dim))
        for word, i in tokeniser.word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        return embedding_matrix

# class GloveEmbeddingLayer(Layer):
#     def __init__(self, vocab_size, tokeniser):
#         self.trainable = False
#         self.dimensions = 50
#         self.embedding_matrix = self.compute_embedding_matrix(vocab_size, tokeniser)
#         super(GloveEmbeddingLayer, self).__init__(vocab_size=vocab_size)
#
#     def compute_embedding_matrix(self, vocab_size, tokeniser):
#         with open('Datasets/GLOVEDATA/glove.twitter.27B.50d.txt', "r", encoding="utf-8") as file:
#             embeddings_index = {line.split()[0]: list(map(float, line.split()[1:])) for line in file}
#             del embeddings_index['0.45973']  # for some reason, this entry has 49 dimensions instead of 50
#
#         embedding_matrix = np.zeros((vocab_size, self.dimensions))
#         for word, i in tokeniser.word_index.items():
#             embedding_vector = embeddings_index.get(word)
#             if embedding_vector is not None:
#                 embedding_matrix[i] = embedding_vector
#         return embedding_matrix
#
#     def build(self, input_shape):
#         # Create a trainable weight variable for this layer.
#         self.trainable_weights = Constant(self.embedding_matrix)
#         super(GloveEmbeddingLayer, self).build(input_shape)

max_w = 20000
max_s_l = 1000
tokenizer = Tokenizer(num_words=max_w)
tokenizer.fit_on_texts(data['clean_data'])
sequences = tokenizer.texts_to_sequences(data['clean_data'])


data_1 = pad_sequences(sequences, maxlen=max_s_l)

def build_model():
    model = Sequential()
    # model.add(ElmoEmbeddingLayer(input_shape=(1,), input_dtype="string"))
    # model.add(GloveEmbeddingLayer(input_shape=(1,), input_length= 1000, output_dim=50, trainable=False))
    #model.add(MultihotEmbedding(vocab_size=40000, input_shape=(1,)))
    model.add(GloveEmbeddingLayer(tokenizer, max_w))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model



split = data['clean_data'].apply(pd.Series)
train_df, test_df, train_l, test_l = train_test_split(data['clean_data'], data['sarcasm_label'], test_size=0.2)
train_text = [' '.join(t.split()[0:150]) for t in train_df]
print(train_text)
train_text = np.array(train_text, dtype=object)[:, np.newaxis]
train_label = list(train_l)

test_text = [' '.join(t.split()[0:150]) for t in test_df]
test_text = np.array(test_text, dtype=object)[:, np.newaxis]
test_label = list(test_l)

# Build and fit
model = build_model()
model.fit(train_text,
          train_label,
          validation_data=(test_text, test_label),
          epochs=4,
          batch_size=32)
# model.summary()
model.save('ElmoModel.h5')
pre_save_preds = model.predict_classes(test_text)  # predictions before we clear and reload model
score = f1_score(test_l, pre_save_preds)
print(score)

# model = build_model()
# model.load_weights('ElmoModel.h5')