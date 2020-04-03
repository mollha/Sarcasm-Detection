import os
from keras.utils import CustomObjectScope
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.engine import Layer
from random import randint
from DataPreprocessing import data_cleaning
from keras.preprocessing.sequence import pad_sequences
from keras.layers.normalization import BatchNormalization
from keras.models import load_model
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import LSTM, Conv1D, MaxPool1D, Flatten, Dense, Dropout, Activation, GlobalMaxPooling1D, \
    Bidirectional, LeakyReLU, MaxPooling1D
from keras.layers import SimpleRNN, GRU
from keras.layers.embeddings import Embedding
from keras import backend as K
from keras.initializers import Constant
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import tensorflow as tf
import tensorflow_hub as hub


# ---------------------------- Create Embedding Layer Classes ----------------------------
class ElmoEmbeddingLayer(Layer):

    def __init__(self, **kwargs):
        self.dimensions = 1024
        self.trainable = True
        self.mask = None
        self.elmo = None
        super(ElmoEmbeddingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        #self.elmo = hub.Module('https://tfhub.dev/google/elmo/2', trainable=self.trainable,
                               #name="{}_module".format(self.name))
        self.elmo = hub.Module(elmo_path, trainable=self.trainable,
                               name="{}_module".format(self.name))
        self.trainable_weights += tf.compat.v1.trainable_variables(scope="^{}_module/.*".format(self.name))
        super(ElmoEmbeddingLayer, self).build(input_shape)

    def call(self, inputs, mask=None):
        seq_len = [inputs.shape[1]] * inputs.shape[0]
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


# def repeat_positive_samples(sarcasm_data: np.ndarray, sarcasm_labels: np.ndarray, sample_pro: float) -> tuple:
#     # merged = pd.DataFrame({"clean_data": sarcasm_data, "sarcasm_label": sarcasm_labels}, columns=["clean_data", "sarcasm_label"])
#     positive_samples = merged[merged['sarcasm_label'] == 1]
#
#     number_of_positive_samples = len(positive_samples) // (1 / sample_pro)
#     my_samples = positive_samples[:number_of_positive_samples]
#     return my_samples['clean_data'].toarray(), my_samples['sarcasm_label'].toarray()


def get_length_limit(dataset_name: str) -> int:
    # based approximately on the average number of tokens in dataset
    names = {'news-headlines-dataset-for-sarcasm-detection': 150, 'Sarcasm_Amazon_Review_Corpus': 1000}
    try:
        return names[dataset_name]
    except KeyError:
        raise ValueError('Dataset name "' + dataset_name + '" does not exist')


def augment_data(sarcasm_data: np.ndarray, labels: np.ndarray, flag=False) -> tuple:
    if flag:
        sarcasm_data = np.concatenate((sarcasm_data, sarcasm_data.copy()))
        labels = np.concatenate((labels, labels.copy()))
    return sarcasm_data, labels


def get_batch_size(model_name: str) -> int:
    # online learning for lstms sets batch size to 1
    batch_sizes = {'lstm': 1, 'bi-lstm': 1, 'cnn': 32, 'vanilla-rnn': 1, 'vanilla-gru': 1}
    return batch_sizes[model_name]


def load_model_from_file(filename: str, custom_layers: dict):
    with CustomObjectScope(custom_layers):
        model = load_model(filename)
    return model


# -------------------------------------------- DEEP LEARNING ARCHITECTURES ---------------------------------------------
def lstm_network(model):
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


def bidirectional_lstm_network(model):
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def deep_cnn_network(model):
    model.add(Conv1D(128, 7, activation='relu', padding='same'))
    model.add(MaxPooling1D())
    model.add(Conv1D(256, 5, activation='relu', padding='same'))
    model.add(MaxPooling1D())
    model.add(Conv1D(512, 3, activation='relu', padding='same'))
    model.add(MaxPooling1D())
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def vanilla_rnn(model, shape):
    model.add(SimpleRNN(50, batch_input_shape=shape, return_sequences=True))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def vanilla_gru(model, shape):
    model.add(GRU(50, batch_input_shape=shape, return_sequences=True))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def cnn(model):
    model.add(Dropout(0.2))
    model.add(Conv1D(filters=32, kernel_size=4, padding='valid', activation='relu', strides=1))
    #model.add(LeakyReLU(alpha=0.1))
    model.add(GlobalMaxPooling1D())
    # vanilla hidden layer:
    model.add(Dense(250))
    model.add(Dropout(0.2))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def cnn_network(model):
    model.add(Conv1D(512, 3, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


# -------------------------------------------- MAIN FUNCTIONS -----------------------------------------------
def prepare_embedding_layer(s_data: pd.Series, s_labels: pd.Series, vector_type: str, split: float, max_batch_size: int, limit: int):
    if vector_type == 'elmo':
        number_of_batches = len(s_data) // (max_batch_size * (1/split))
        sarcasm_data = s_data[:int((number_of_batches * max_batch_size) / split)]
        sarcasm_labels = s_labels[:int((number_of_batches * max_batch_size) / split)]
        text = pd.DataFrame([pad_string(t.split(), limit) for t in sarcasm_data])
        text = text.replace({None: ""})
        text = text.to_numpy()
        return text, sarcasm_labels, ElmoEmbeddingLayer(batch_input_shape=(max_batch_size, limit), input_dtype="string"), {'ElmoEmbeddingLayer': ElmoEmbeddingLayer}

    elif vector_type == 'glove':
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(s_data)
        sequences = tokenizer.texts_to_sequences(s_data)
        padded_data = pad_sequences(sequences, maxlen=limit, padding='post')
        return padded_data, s_labels, GloveEmbeddingLayer(tokenizer.word_index, limit), {'GloveEmbeddingLayer': GloveEmbeddingLayer}
    else:
        raise TypeError('Vector type must be "elmo" or "glove"')



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

    accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]
    plt.plot(epochs, accuracy, label="Training accuracy")
    plt.plot(epochs, val_accuracy, label="Validation accuracy")
    plt.title("Training and validation accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()

def get_model(model_name: str, dataset_name: str, sarcasm_data: pd.Series, sarcasm_labels: pd.Series, vector_type: str, split: float):
    max_batch_size = get_batch_size(model_name)
    length_limit = get_length_limit(dataset_name)
    s_data, l_data, e_layer, c_layer = prepare_embedding_layer(sarcasm_data, sarcasm_labels, vector_type, split, max_batch_size, length_limit)

    model = Sequential()
    e_layer.trainable = False
    model.add(e_layer)
    if model_name == 'lstm':
        model = lstm_network(model)
    elif model_name == 'bi-lstm':
        model = bidirectional_lstm_network(model)
    elif model_name == 'cnn':
        model = cnn_network(model)
    elif model_name == 'vanilla-rnn':
        model = vanilla_rnn(model, (max_batch_size, length_limit))
    elif model_name == 'vanilla-gru':
        model = vanilla_gru(model, (len(s_data), length_limit))
    return s_data, l_data, model, c_layer


def get_results(model_name: str, sarcasm_data: pd.Series, sarcasm_labels: pd.Series, dataset_name: str, vector_type: str):
    print('Training ' + model_name.upper() + ' using ' + vector + ' vectors.')
    print('Dataset size: ' + str(len(sarcasm_data)) + '\n')
    split = 0.2
    epochs = 1
    patience = 10
    max_batch_size = get_batch_size(model_name)

    s_data, l_data, dl_model, custom_layer = get_model(model_name, dataset_name, sarcasm_data, sarcasm_labels, vector_type, split)
    training_data, testing_data, training_labels, testing_labels = train_test_split(s_data, l_data, test_size=split)
    print(training_data.shape)

    # training_data, training_labels = repeat_positive_samples(training_data, training_labels, 0.5)

    training_data, training_labels = augment_data(training_data, training_labels, flag=False)
    testing_data, testing_labels = augment_data(testing_data, testing_labels, flag=False)

    file_name = 'TrainedModels/' + model_name + '_with_' + vector_type + '_on_' + dataset_name + '.h5'
    model_checkpoint = ModelCheckpoint(file_name, monitor='val_loss', mode='auto', save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, verbose=1, mode='auto')

    model_history = dl_model.fit(x=np.array(training_data), y=np.array(training_labels), validation_data=(testing_data, testing_labels),
                                 epochs=epochs, batch_size=max_batch_size, callbacks=[early_stopping, model_checkpoint])

    dl_model = load_model_from_file(file_name, custom_layer)
    y_pred = dl_model.predict_classes(x=np.array(testing_data), batch_size=max_batch_size)
    score = f1_score(np.array(testing_labels), np.array(y_pred))
    print('F1 Score: ', score)

    # evaluate

    # class_weight = {0: 1.0, 1: 1.0}
    # my_adam = optimizers.Adam(lr=0.003, decay=0.001)
    # print(model.summary())
    visualise_results(model_history)


# ---------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    dataset_paths = ["Datasets/Sarcasm_Amazon_Review_Corpus", "Datasets/news-headlines-dataset-for-sarcasm-detection"]
    elmo_path = 'Datasets/elmo'

    # Choose a dataset from the list of valid data sets
    path_to_dataset_root = dataset_paths[1]
    print('Selected dataset: ' + path_to_dataset_root[9:])

    set_size = 2000  # 22895

    # Read in raw data
    data = pd.read_csv(path_to_dataset_root + "/processed_data/OriginalData.csv", encoding="ISO-8859-1")[:set_size]
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
                           encoding="ISO-8859-1")[:set_size]

    data['clean_data'] = get_clean_data_col(data, path_to_dataset_root, False)

    model_n = 'lstm'
    vector = 'elmo'
    d_name = path_to_dataset_root[9:]

    get_results(model_n, data['clean_data'], data['sarcasm_label'], d_name, vector)