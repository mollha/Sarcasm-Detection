import os
from keras.utils import CustomObjectScope
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.engine import Layer
from random import randint
from Code.DataPreprocessing import *
from keras.preprocessing.sequence import pad_sequences
from keras.layers.normalization import BatchNormalization
from keras.models import load_model
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import LSTM, ReLU, Conv1D, MaxPool1D, Flatten, Dense, Dropout, Activation, GlobalMaxPooling1D
from keras.layers.embeddings import Embedding
from keras import backend as K
from keras.initializers import Constant
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import tensorflow as tf
import tensorflow_hub as hub
max_w = 20000
max_batch_size = 1


# ---------------------------- Create Embedding Layer Classes ----------------------------
class ElmoEmbeddingLayer(Layer):

    def __init__(self, **kwargs):
        self.dimensions = 1024
        self.trainable = True
        self.mask = None
        self.elmo = None
        super(ElmoEmbeddingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.elmo = hub.Module('https://tfhub.dev/google/elmo/2', trainable=self.trainable,
                               name="{}_module".format(self.name))
        self.trainable_weights += tf.compat.v1.trainable_variables(scope="^{}_module/.*".format(self.name))
        super(ElmoEmbeddingLayer, self).build(input_shape)

    def call(self, inputs, mask=None):
        # inputs.shape = [batch_size, seq_len]
        seq_len = [inputs.shape[1]] * inputs.shape[0]
        # this will give a list of seq_len: [seq_len, seq_len, ..., seq_len] just like the official example.
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
        return K.not_equal(inputs, '--PAD--')

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], self.dimensions


class GloveEmbeddingLayer(Embedding):
    def __init__(self, word_index, max_seq_len, **kwargs):
        self.word_index = word_index
        self.MAX_SEQUENCE_LENGTH = max_seq_len
        self.output_dim = 50
        self.embeddings_matrix = self.compute_embedding_matrix(len(self.word_index) + 1)
        super(GloveEmbeddingLayer, self).__init__(input_dim=len(self.word_index) + 1,
                                                  output_dim=self.output_dim,
                                                  embeddings_initializer=Constant(self.embeddings_matrix),
                                                  input_length=self.MAX_SEQUENCE_LENGTH,
                                                  trainable=False,
                                                  **kwargs
                                                  )

    def get_config(self):
        return {'word_index': self.word_index,
                'max_seq_len': self.MAX_SEQUENCE_LENGTH}

    def compute_embedding_matrix(self, vocab_size):
        with open('Datasets/GLOVEDATA/glove.twitter.27B.50d.txt', "r", encoding="utf-8") as file:
            embeddings_index = {line.split()[0]: list(map(float, line.split()[1:])) for line in file}
            del embeddings_index['0.45973']  # for some reason, this entry has 49 dimensions instead of 50

        embedding_matrix = np.zeros((vocab_size, self.output_dim))
        for word, i in self.word_index.items():
            if i >= vocab_size:
                continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        return embedding_matrix


# ----------------------------------------------- HELPER FUNCTIONS -----------------------------------------------------
def pad_string(tokens: list, limit: int) -> list:
    tokens = tokens[0:limit]
    extend_by = limit - len(tokens)

    if extend_by > 0:
        tokens.extend([""] * extend_by)
    return tokens


def prepare_embedding_layer(sarcasm_data: pd.Series, sarcasm_labels: pd.Series, vector_type: str, cv: int):
    length_limit = 150
    if vector_type == 'elmo':
        number_of_batches = len(sarcasm_data) // (max_batch_size*cv)
        print('Amount used', number_of_batches*max_batch_size)
        sarcasm_data = sarcasm_data[:number_of_batches*max_batch_size]
        sarcasm_labels = sarcasm_labels[:number_of_batches*max_batch_size]
        text = pd.DataFrame([pad_string(t.split(), length_limit) for t in sarcasm_data])
        text = text.replace({None: ""})
        text = text.to_numpy()
        return text, sarcasm_labels, ElmoEmbeddingLayer(batch_input_shape=(32, length_limit), input_dtype="string")

    elif vector_type == 'glove':
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(sarcasm_data)
        sequences = tokenizer.texts_to_sequences(sarcasm_data)
        padded_data = pad_sequences(sequences, maxlen=length_limit, padding='post')
        return padded_data, sarcasm_labels, GloveEmbeddingLayer(tokenizer.word_index, length_limit)
    else:
        raise TypeError('Vector type must be "elmo" or "glove"')


# -------------------------------------------- DEEP LEARNING ARCHITECTURES ---------------------------------------------
def lstm_network(model):
    model.add(LSTM(units=128, dropout=0.2, kernel_initializer='he_normal', activation='tanh', return_sequences=True))
    model.add(LSTM(units=128, dropout=0.2, kernel_initializer='he_normal', activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def new_lstm_network(model):
    model.add(LSTM(60, return_sequences=True))
    model.add(GlobalMaxPooling1D())
    model.add(Dropout(0.1))
    model.add(Dense(50, activation="relu"))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def cnn_network(model):
    model.add(Dropout(0.2))
    model.add(Conv1D(filters=32, kernel_size=4, padding='valid', activation='relu', strides=1))
    model.add(GlobalMaxPooling1D())
    # vanilla hidden layer:
    model.add(Dense(250))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def cnn_network_batch_norm(model):
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
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ---------------------------------------------------------------------------------------------------------------------

dataset_paths = ["Datasets/Sarcasm_Amazon_Review_Corpus", "Datasets/news-headlines-dataset-for-sarcasm-detection"]

# Choose a dataset from the list of valid data sets
path_to_dataset_root = dataset_paths[1]
print('Selected dataset: ' + path_to_dataset_root[9:])

set_size = 5000  # 22895

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
s_data, l_data, emb_layer = prepare_embedding_layer(data['clean_data'], data['sarcasm_label'], 'glove', 5)
# Split into training and test data
X_train, X_test, labels_train, labels_test = train_test_split(s_data, l_data, test_size=0.2)



sequence_length= 150

# Train models from scratch
model = Sequential()
e = emb_layer
e.trainable = False
model.add(e)

# model = cnn_network_batch_norm(model)
model = new_lstm_network(model)
#model = cnn_network(model)
model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='auto', save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')
#model.fit(train_X, train_y, validation_split=0.3)
history = model.fit(x=np.array(X_train), y=np.array(labels_train), validation_data=(X_test, labels_test),
                        epochs=300, batch_size=max_batch_size, callbacks=[early_stopping, model_checkpoint])

with CustomObjectScope({'GloveEmbeddingLayer': GloveEmbeddingLayer}):
    model = load_model('best_model.h5')


# evaluate
y_pred = model.predict_classes(x=X_test)
score = f1_score(labels_test, y_pred)
# model = KerasClassifier(build_fn=new_model)
print(score)

# class_weight = {0: 1.0, 1: 1.0}
# my_adam = optimizers.Adam(lr=0.003, decay=0.001)
# print(model.summary())
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(loss)+1)
plt.plot(epochs, loss, label="Training loss")
plt.plot(epochs, val_loss, label="Validation loss")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

accuracy = history.history["acc"]
val_accuracy = history.history["val_acc"]
plt.plot(epochs, accuracy, label="Training accuracy")
plt.plot(epochs, val_accuracy, label="Validation accuracy")
plt.title("Training and validation accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epochs")
plt.legend()
plt.show()