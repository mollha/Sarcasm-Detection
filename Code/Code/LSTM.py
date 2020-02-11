from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import LSTM
from keras.datasets import imdb
import pandas as pd
from sklearn.model_selection import KFold

max_features = 50
# cut texts after this number of words (among top max_features most common words)
maxlen = 50
batch_size = 32


def build_LSTM_network():
    model = Sequential()
    # first layer in the LSTM is the embedding layer - 300dim word embedding
    model.add(Embedding(input_dim=-10, output_dim=128, input_length=50))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def get_LSTM():
    return KerasClassifier(build_fn=build_LSTM_network,
                           epochs=10,
                           batch_size=batch_size,
                           verbose=0)
