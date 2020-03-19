import os
import numpy as np
import pandas as pd
from keras.engine import Layer
from random import randint
from Code.DataPreprocessing import *
from keras.preprocessing.sequence import pad_sequences
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.layers import LSTM, ReLU, Conv1D, MaxPool1D, Flatten, Dense, Input
from keras.layers.embeddings import Embedding
from keras import backend as K
from keras.initializers import Constant
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import tensorflow as tf
import tensorflow_hub as hub
max_w = 20000
max_batch_size = 32

# ---------------------------- Create Embedding Layer Classes ----------------------------
class ElmoEmbeddingLayer(Layer):

    def __init__(self, mask, **kwargs):
        self.dimensions = 1024
        self.trainable = True
        self.mask = mask
        super(ElmoEmbeddingLayer, self).__init__(**kwargs)


    def build(self, input_shape):
        self.elmo = hub.Module('https://tfhub.dev/google/elmo/2', trainable=self.trainable,
                               name="{}_module".format(self.name))
        self.trainable_weights += tf.compat.v1.trainable_variables(scope="^{}_module/.*".format(self.name))
        super(ElmoEmbeddingLayer, self).build(input_shape)


    def call(self, inputs, mask=None):
        # inputs.shape = [batch_size, seq_len]
        seq_len = [inputs.shape[1]] * inputs.shape[
            0]  # this will give a list of seq_len: [seq_len, seq_len, ..., seq_len] just like the official example.
        result = self.elmo(inputs={"tokens": K.cast(inputs, dtype=tf.string),
                                   "sequence_len": seq_len},
                           as_dict=True,
                           signature='tokens',
                           )['elmo']
        print(result.shape)
        return result


    def compute_mask(self, inputs, mask=None):
        if not self.mask:
            return None

        output_mask = K.not_equal(inputs, '--PAD--')
        return output_mask


    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.dimensions)


class GloveEmbeddingLayer(Embedding):
    def __init__(self, tokeniser, max_seq_len):
        self.MAX_SEQUENCE_LENGTH = max_seq_len
        self.output_dim = 50
        self.embeddings_matrix = self.compute_embedding_matrix(len(tokeniser.word_index) + 1, tokeniser)
        super(GloveEmbeddingLayer, self).__init__(input_dim=len(tokeniser.word_index) + 1,
                                                  output_dim=self.output_dim,
                                                  embeddings_initializer=Constant(self.embeddings_matrix),
                                                  input_length=self.MAX_SEQUENCE_LENGTH,
                                                  trainable=False
                                                  )

    def compute_embedding_matrix(self, vocab_size, tokeniser):
        with open('Datasets/GLOVEDATA/glove.twitter.27B.50d.txt', "r", encoding="utf-8") as file:
            embeddings_index = {line.split()[0]: list(map(float, line.split()[1:])) for line in file}
            del embeddings_index['0.45973']  # for some reason, this entry has 49 dimensions instead of 50

        embedding_matrix = np.zeros((vocab_size, self.output_dim))
        for word, i in tokeniser.word_index.items():
            if i >= vocab_size:
                continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        return embedding_matrix


# -------------------------------- HELPER FUNCTIONS ----------------------------------
def pad_string(tokens: list, limit: int) -> list:
    tokens = tokens[0:limit]
    extend_by = limit - len(tokens)

    if extend_by > 0:
        tokens.extend([""] * extend_by)
    return tokens

def prepare_embedding_layer(sarcasm_data: pd.Series, sarcasm_labels: pd.Series, vector_type: str):
    length_limit = 150
    print('shapes: ', len(sarcasm_data))
    if vector_type == 'elmo':
        # print([t.split()[0:150] for t in sarcasm_data])
        # text = pd.DataFrame([t.split()[0:150] for t in sarcasm_data])
        text = pd.DataFrame([pad_string(t.split(), length_limit) for t in sarcasm_data])

        #print(text)
        print(text.shape)
        text = text.replace({None: ""})
        text = text.to_numpy()
        #print(text)
        # text = [' '.join(t.split()[0:150]) for t in sarcasm_data]
        # text = np.array(text, dtype=object)[:, np.newaxis]
        sequence_length=150
        return text, sarcasm_labels, ElmoEmbeddingLayer(batch_input_shape=(32, length_limit), input_dtype="string", mask=None)

    elif vector_type == 'glove':
        max_len = 1000
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(sarcasm_data)
        sequences = tokenizer.texts_to_sequences(sarcasm_data)
        padded_data = pad_sequences(sequences, maxlen=max_len, padding='post')
        print(padded_data.shape)
        return padded_data, sarcasm_labels, GloveEmbeddingLayer(tokenizer, max_len)
    else:
        raise TypeError('Vector type must be "elmo" or "glove"')

dataset_paths = ["Datasets/Sarcasm_Amazon_Review_Corpus", "Datasets/news-headlines-dataset-for-sarcasm-detection"]

# Choose a dataset from the list of valid data sets
path_to_dataset_root = dataset_paths[1]
print('Selected dataset: ' + path_to_dataset_root[9:])

set_size = 640

# Read in raw data
data = pd.read_csv(path_to_dataset_root + "/processed_data/OriginalData.csv", encoding="ISO-8859-1")[:set_size]


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
                       encoding="ISO-8859-1")[:set_size]


# Clean data, or retrieve pre-cleaned data
data['clean_data'] = get_clean_data_col(data, path_to_dataset_root, False)

print("\nGLOVE MODEL")
# get the data
s_data, l_data, emb_layer = prepare_embedding_layer(data['clean_data'], data['sarcasm_label'], 'elmo')
# Split into training and test data
X_train, X_test, labels_train, labels_test = train_test_split(s_data, l_data, test_size=0.2)


def cnn_network(model):
    model.add(Conv1D(filters=32, kernel_size=4, strides=2, padding='valid', use_bias=False))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(MaxPool1D(pool_size=2, strides=1))
    model.add(Conv1D(filters=32, kernel_size=3, strides=1, padding='valid', use_bias=False))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(MaxPool1D(pool_size=2, strides=1))
    model.add(Conv1D(filters=32, kernel_size=3, strides=1, padding='valid', use_bias=False))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(MaxPool1D(pool_size=2, strides=1))
    model.add(Flatten())
    return model

sequence_length= 150

# Train models from scratch
model = Sequential()
e = emb_layer
e.trainable = False
model.add(e)

model = cnn_network(model)
# model.add(LSTM(units=100, dropout=0.5, recurrent_dropout=0.5, return_sequences=False))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(x=np.array(X_train), y=np.array(labels_train), validation_data=(X_test, labels_test),
                        epochs=10, batch_size=max_batch_size)

# evaluate
y_pred = model.predict_classes(x=X_test)
score = f1_score(labels_test, y_pred)
# model = KerasClassifier(build_fn=new_model)
print(score)

# class_weight = {0: 1.0, 1: 1.0}
# my_adam = optimizers.Adam(lr=0.003, decay=0.001)
# print(model.summary())