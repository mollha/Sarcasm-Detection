import tensorflow as tf
import pandas as pd
import tensorflow_hub as hub
from Code.DataPreprocessing import *
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import os
import re
from keras.layers import LSTM, Bidirectional
from keras.layers import ReLU
from keras.layers import MaxPool1D, MaxPooling1D
from keras.layers import Conv1D
from keras.layers.normalization import BatchNormalization
from sklearn.metrics import f1_score, accuracy_score
from keras.wrappers.scikit_learn import KerasClassifier
from random import randint
from keras import backend as K
from keras.layers import Dense, Input, Embedding, Flatten
from keras.models import Model, load_model, Sequential
from keras.engine import Layer
from sklearn.model_selection import train_test_split
from keras.initializers import Constant
import numpy as np
K.set_session(tf.Session())
max_w = 20000
max_s_l = 1000


# ---------------------------- Create Embedding Layer Classes ----------------------------
class ElmoEmbeddingLayer(Layer):
    def __init__(self, **kwargs):
        self.dimensions = 1024
        self.elmo = None
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
        result = self.elmo(K.squeeze(K.cast(x, tf.string), axis=1),
                           as_dict=True,
                           signature='default',
                           )['default']
        return result

    def compute_mask(self, inputs, mask=None):
        return K.not_equal(inputs, '--PAD--')

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.dimensions


class GloveEmbeddingLayer(Embedding):
    def __init__(self, tokeniser, max_num_words, max_seq_len):
        self.MAX_SEQUENCE_LENGTH = max_seq_len
        self.MAX_NUM_WORDS = max_num_words
        self.num_words = min(max_num_words, len(tokeniser.word_index) + 1)
        self.output_dim = 50
        self.embeddings_matrix = self.compute_embedding_matrix(self.num_words, tokeniser)
        print(self.embeddings_matrix)
        super(GloveEmbeddingLayer, self).__init__(input_dim=min(self.num_words, len(tokeniser.word_index) + 1),
                                                  output_dim=50,
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
            if i >= self.MAX_NUM_WORDS:
                continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        return embedding_matrix


# -------------------------------- HELPER FUNCTIONS ----------------------------------

def prepare_embedding_layer(sarcasm_data: pd.Series, sarcasm_labels: pd.Series, vector_type: str):
    text = [' '.join(t.split()[0:150]) for t in sarcasm_data]
    labels = list(sarcasm_labels)

    if vector_type == 'elmo':
        text = np.array(text, dtype=object)[:, np.newaxis]
        return text, labels, ElmoEmbeddingLayer(input_shape=(1,), input_dtype="string")

    elif vector_type == 'glove':
        tokenizer = Tokenizer(num_words=max_w)
        tokenizer.fit_on_texts(text)
        sequences = tokenizer.texts_to_sequences(text)
        padded_data = pad_sequences(sequences, maxlen=max_s_l)
        return padded_data, labels, GloveEmbeddingLayer(tokenizer, max_w, max_s_l)
    else:
        raise TypeError('Vector type must be "elmo" or "glove"')

def cnn(embedding_layer):
    model = Sequential()
    model.add(embedding_layer)
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
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def build_model(embedding_layer):
    model = Sequential()
    # model.add(ElmoEmbeddingLayer(input_shape=(1,), input_dtype="string"))
    # model.add(MultihotEmbedding(vocab_size=40000, input_shape=(1,)))
    # model.add(GloveEmbeddingLayer(tokenizer, max_w, max_s_l))
    model.add(embedding_layer)
    model.add(Dense(256, activation='relu'))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


if __name__ == "__main__":

    # Initialize session
    dataset_paths = ["Datasets/Sarcasm_Amazon_Review_Corpus", "Datasets/news-headlines-dataset-for-sarcasm-detection"]

    # Choose a dataset from the list of valid data sets
    path_to_dataset_root = dataset_paths[1]
    print('Selected dataset: ' + path_to_dataset_root[9:])

    # Read in raw data
    data = pd.read_csv(path_to_dataset_root + "/processed_data/OriginalData.csv", encoding="ISO-8859-1")


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
                           encoding="ISO-8859-1")


    # Clean data, or retrieve pre-cleaned data
    data['clean_data'] = get_clean_data_col(data, path_to_dataset_root, False)
    s_data, l_data, emb_layer = prepare_embedding_layer(data['clean_data'], data['sarcasm_label'], 'elmo')

    train_d, test_d, train_l, test_l = train_test_split(s_data, l_data, test_size=0.2)

    # model.fit(training_data, training_labels, batch_size=16, epochs=10)
    # preds_valid = model.predict(testing_data)
    # print(preds_valid)
    #
    # # loss, score = model.evaluate(testing_data, testing_labels, batch_size=16)
    # # print(testing_labels)
    # score = f1_score(testing_labels, preds_valid)
    #
    # print('Score: %.2f' % score)


    # Build and fit
    #model = build_model(emb_layer)
    model = cnn(emb_layer)
    model.summary()
    model.fit(train_d,
              train_l,
              validation_data=(test_d, test_l),
              epochs=15,
              batch_size=32)
    pre_save_preds = model.predict_classes(test_d)  # predictions before we clear and reload model
    score = f1_score(test_l, pre_save_preds)
    # model = KerasClassifier(build_fn=new_model)
    print(score)


# model.summary()
#model.save('ElmoModel.h5')


# model = build_model()
# model.load_weights('ElmoModel.h5')
