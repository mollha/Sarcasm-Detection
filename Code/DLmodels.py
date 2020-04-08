import matplotlib.pyplot as plt
import numpy as np
from keras import utils
import os
from random import randint
from Code.DataPreprocessing import data_cleaning
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.engine import Layer
from keras.initializers import Constant
from keras.layers import Dropout, Activation, GlobalMaxPooling1D, Bidirectional, TimeDistributed
from keras.layers import LSTM, Conv1D, Dense
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import CustomObjectScope
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


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
        return result

    def compute_mask(self, inputs, mask=None):
        if not self.mask:
            return None
        return K.not_equal(inputs, '--PAD--')

    def get_config(self):
        return {'batch_input_shape': self.batch_input_shape,
                'input_dtype': "string"}

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


class BagOfWordsEmbeddingLayer(Embedding):
    def __init__(self, word_index, model_weights, **kwargs):
        self.word_index = word_index
        self.model_weights = model_weights
        super(BagOfWordsEmbeddingLayer, self).__init__(input_dim=len(self.word_index) + 1,
                                                       output_dim=self.word_index,
                                                       embeddings_initializer=Constant(self.model_weights),
                                                       input_length=self.word_index,
                                                       trainable=False,
                                                       **kwargs
                                                       )

    def get_config(self):
        return {'word_index': self.word_index,
                'model_weights': self.model_weights}


# ----------------------------------------------- HELPER FUNCTIONS -----------------------------------------------------
def get_length_limit(dataset_name: str) -> int:
    # TODO define reasoned integers for this
    datasets = {'news-headlines-dataset-for-sarcasm-detection': 150, 'Sarcasm_Amazon_Review_Corpus': 500}
    try:
        return datasets[dataset_name]
    except KeyError:
        raise ValueError('Dataset name "' + dataset_name + '" does not exist')


def pad_string(tokens: list, limit: int) -> list:
    tokens = tokens[0:limit]
    extend_by = limit - len(tokens)

    if extend_by > 0:
        tokens.extend([""] * extend_by)
    return tokens


def visualise_results(history):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(loss) + 1)
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


def load_model_from_file(filename: str, custom_layers: dict):
    with CustomObjectScope(custom_layers):
        model = load_model(filename)
    return model

# def load_model_from_file_name(json_name, custom_layers: dict):
#     loaded_model_json = open(json_name, 'r', encoding="ISO-8859-1").read()
#
#     with CustomObjectScope(custom_layers):
#         model = model_from_json(loaded_model_json)
#         model.load_weights(json_name + '.h5')
#     return model

# -------------------------------------------- DEEP LEARNING ARCHITECTURES ---------------------------------------------
def lstm_network(model):
    model.add(LSTM(60, activation='tanh', return_sequences=True))
    model.add(GlobalMaxPooling1D())
    model.add(Dropout(0.1))
    model.add(Dense(50, activation="relu"))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

def bidirectional_lstm_network(model):
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
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


# ---------------------------------------------- MAIN FUNCTIONS ------------------------------------------------------
def prepare_embedding_layer(sarcasm_data: pd.Series, sarcasm_labels: pd.Series, vector_type: str, split: float, max_batch_size: int):
    length_limit = 150
    if vector_type == 'elmo':
        number_of_batches = len(sarcasm_data) // (max_batch_size * (1/split))
        sarcasm_data = sarcasm_data[:int((number_of_batches * max_batch_size) / split)]
        sarcasm_labels = sarcasm_labels[:int((number_of_batches * max_batch_size) / split)]
        text = pd.DataFrame([pad_string(t.split(), length_limit) for t in sarcasm_data])
        text = text.replace({None: ""})
        text = text.to_numpy()
        return text, sarcasm_labels, ElmoEmbeddingLayer(batch_input_shape=(max_batch_size, length_limit),
                                                        input_dtype="string"), {'ElmoEmbeddingLayer': ElmoEmbeddingLayer}
    elif vector_type == 'glove':
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(sarcasm_data)
        sequences = tokenizer.texts_to_sequences(sarcasm_data)
        padded_data = pad_sequences(sequences, maxlen=length_limit, padding='post')
        return padded_data, sarcasm_labels, GloveEmbeddingLayer(tokenizer.word_index, length_limit), \
               {'GloveEmbeddingLayer': GloveEmbeddingLayer}
    elif vector_type == 'bag_of_words':
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(sarcasm_data)
        sequences = tokenizer.texts_to_sequences(sarcasm_data)
        matrix = tokenizer.texts_to_matrix(sarcasm_data, mode='freq')
        print(tokenizer.word_index)
        print(matrix)
        return sequences, sarcasm_labels, BagOfWordsEmbeddingLayer(tokenizer.word_index, model_weights=matrix), \
               {'BagOfWordsEmbeddingLayer': BagOfWordsEmbeddingLayer}
    else:
        raise TypeError('Vector type must be "elmo", "glove" or "bag_of_words"')


def get_model(model_name: str, sarcasm_data: pd.Series, sarcasm_labels: pd.Series, vector_type: str, split: float):
    max_batch_size = 1 if model_name == 'lstm' or model_name == 'bi-lstm' else 32

    model = Sequential()
    sarcasm_data, sarcasm_labels, embedding_layer, cus_layers = prepare_embedding_layer(sarcasm_data=sarcasm_data,
                                                                                        sarcasm_labels=sarcasm_labels,
                                                                                        vector_type=vector_type,
                                                                                        split=split,
                                                                                        max_batch_size=max_batch_size)
    model.add(embedding_layer)
    if model_name == 'lstm':
        model = lstm_network(model)
    elif model_name == 'bi-lstm':
        model = bidirectional_lstm_network(model)
    elif model_name == 'cnn':
        model = cnn_network(model)
    return sarcasm_data, sarcasm_labels, model, cus_layers, max_batch_size


def get_results(model_name: str, dataset_name: str, sarcasm_data: pd.Series, sarcasm_labels: pd.Series, vector_type: str, split: float):
    s_data, l_data, dl_model, custom_layers, max_batch_size = get_model(model_name, sarcasm_data, sarcasm_labels, vector_type, split)
    X_train, X_test, labels_train, labels_test = train_test_split(s_data, l_data, test_size=split)
    file_name = 'TrainedModels/' + model_name + '_with_' + vector_type + '_on_' + dataset_name + '.h5'
    model_checkpoint = ModelCheckpoint(file_name, monitor='val_loss', mode='auto', save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=1, verbose=1, mode='auto')
    model_history = dl_model.fit(x=np.array(X_train), y=np.array(labels_train), validation_data=(X_test, labels_test),
                                 epochs=1, batch_size=max_batch_size, callbacks=[early_stopping, model_checkpoint])

    dl_model = load_model_from_file(file_name, custom_layers)
    y_pred = dl_model.predict_classes(x=X_test, batch_size=max_batch_size)
    print(y_pred)
    score = f1_score(labels_test, y_pred)
    print('Score: ', score)

    # class_weight = {0: 1.0, 1: 1.0}
    # my_adam = optimizers.Adam(lr=0.003, decay=0.001)
    # print(model.summary())
    visualise_results(model_history)



if __name__ == '__main__':
    dataset_paths = ["Datasets/Sarcasm_Amazon_Review_Corpus", "Datasets/news-headlines-dataset-for-sarcasm-detection"]

    # Choose a dataset from the list of valid data sets
    path_to_dataset_root = dataset_paths[1]
    print('Selected dataset: ' + path_to_dataset_root[9:])

    # set_size = 20000  # 22895

    # Read in raw data
    data = pd.read_csv(path_to_dataset_root + "/processed_data/OriginalData.csv", encoding="ISO-8859-1")#[:set_size]
    print(data.shape)


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
                           encoding="ISO-8859-1")# [:set_size]

    data['clean_data'] = get_clean_data_col(data, path_to_dataset_root, False)

    model_n = 'cnn'
    vector = 'elmo'
    d_name = path_to_dataset_root[9:]
    get_results(model_n, d_name, data['clean_data'], data['sarcasm_label'], vector, 0.2)