from keras.utils import CustomObjectScope
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from keras.engine import Layer
from Code.pkg.data_processing.helper import prepare_data
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import LSTM, Conv1D, Flatten, Dense, Dropout, GlobalMaxPooling1D, \
    Bidirectional, LeakyReLU, MaxPooling1D
from keras.layers import SimpleRNN, GRU
from keras.layers.embeddings import Embedding
import keras.backend as K
from keras.initializers import Constant
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import tensorflow as tf
import tensorflow_hub as hub
import spacy


nlp = spacy.load('en_core_web_md')


# ncc terminal
# ssh -D 8080 -q -C -N kgxj22@mira.dur.ac.uk


# ---------------------------- Create Embedding Layer Classes ----------------------------
class ElmoEmbeddingLayer(Layer):

    def __init__(self, **kwargs):
        self.dimensions = 1024
        self.trainable = True
        self.mask = None
        self.elmo = None
        super(ElmoEmbeddingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # base_path = Path(__file__).parent

        self.elmo = hub.Module('https://tfhub.dev/google/elmo/2', trainable=self.trainable,
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
        base_path = Path(__file__).parent
        file_path = (base_path / "../language_models/glove/glove.twitter.27B.50d.txt").resolve()

        with open(file_path, "r", encoding="utf-8") as file:
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
    names = {'news_headlines': 150, 'amazon_reviews': 1000, 'ptacek': 150}
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
    batch_sizes = {'lstm': 32, 'bi-lstm': 32, 'cnn': 32, 'vanilla-rnn': 32, 'vanilla-gru': 32}
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
def prepare_vector_embedding_layer(s_data: pd.Series, s_labels: pd.Series, vector_type: str, split: float, max_batch_size: int, limit: int):
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


def visualise_results(history, file_name):

    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, label="Training loss")
    plt.plot(epochs, val_loss, label="Validation loss")
    plt.title("Training and validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(file_name + '_loss.png')
    plt.show()

    accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]
    plt.plot(epochs, accuracy, label="Training accuracy")
    plt.plot(epochs, val_accuracy, label="Validation accuracy")
    plt.title("Training and validation accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()
    plt.savefig(file_name + '_accuracy.png')
    plt.show()



def get_model(model_name: str, dataset_name: str, sarcasm_data: pd.Series, sarcasm_labels: pd.Series, vector_type: str, split: float):
    max_batch_size = get_batch_size(model_name)
    length_limit = get_length_limit(dataset_name)

    # need to deal with concatenating multiple embedding layers - difficult for time series data, as our current annotators work for sentences only
    # e.g. sentiment annotator produces a 6-dimensional vector for each sentence - not for individual words
    # sentiment may be equally valid if we decide to include sentiment per word. However, these dimensions will not match
    # if i remove the overall sentiment from the 6-dimensional vector, we could mimic the behaviour
    # e.g. [0 0 0 0 1] for each word, when averaging embeddings will be very similar

    s_data, l_data, e_layer, c_layer = prepare_vector_embedding_layer(sarcasm_data, sarcasm_labels, vector_type,
                                                                      split, max_batch_size, length_limit)



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


def get_dl_results(model_name: str, dataset_number: int, vector_type: str, set_size=None):
    base_path = Path(__file__).parent
    dataset_name, sarcasm_labels, _, clean_data, _ = prepare_data(dataset_number, vector_type, [], set_size)

    print('Training ' + model_name.upper() + ' using ' + vector_type + ' vectors.')
    print('Dataset size: ' + str(len(clean_data)) + '\n')
    split = 0.1
    epochs = 300
    patience = 10
    max_batch_size = get_batch_size(model_name)

    s_data, l_data, dl_model, custom_layer = get_model(model_name, dataset_name, clean_data, sarcasm_labels, vector_type, split)
    training_data, testing_data, training_labels, testing_labels = train_test_split(s_data, l_data, test_size=split)
    print(training_data.shape)

    # training_data, training_labels = repeat_positive_samples(training_data, training_labels, 0.5)

    training_data, training_labels = augment_data(training_data, training_labels, flag=False)
    testing_data, testing_labels = augment_data(testing_data, testing_labels, flag=False)

    file_name = str(base_path / ('../trained_models/' + model_name + '_with_' + vector_type + '_on_' + str(dataset_number) + '.h5'))

    model_checkpoint = ModelCheckpoint(file_name, monitor='val_loss', mode='auto', save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, verbose=1, mode='auto')

    with tf.compat.v1.Session().as_default():
        # sess.run(tf.compat.v1.global_variables_initializer())
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
        visualise_results(model_history, str(base_path / ('../training_images/' + model_name + '_with_' + vector_type + '_on_' + str(dataset_number))))


# ---------------------------------------------------------------------------------------------------------------------