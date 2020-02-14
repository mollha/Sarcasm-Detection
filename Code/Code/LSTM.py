# from keras.preprocessing import sequence
# from keras.models import Sequential
# from keras.layers import Dense, Embedding
# from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D
# from keras.layers import Dense, Dropout, Flatten, LSTM, GRU, Bidirectional, Input, Multiply
# from keras.wrappers.scikit_learn import KerasClassifier
# from keras.layers import LSTM
# from keras.datasets import imdb
# import pandas as pd
# from sklearn.model_selection import KFold
#
# max_features = 50
# # cut texts after this number of words (among top max_features most common words)
# maxlen = 50
# batch_size = 32
# trainable = True
# plot = True
# shuffle = False
# epochs = 50
# # batch_size = 256
# embedding_dim = 300
# hidden_units = 256
# dropout_val = 0.3
#
# # A model using just convolutional neural networks
# def cnn_model(**kwargs):
#     X = Conv1D(filters=kwargs['hidden_units'], kernel_size=3, kernel_initializer='he_normal', padding='valid',
#                activation='relu')(kwargs['embeddings'])
#     X = Conv1D(filters=kwargs['hidden_units'], kernel_size=3, kernel_initializer='he_normal', padding='valid',
#                activation='relu')(X)
#     X = GlobalMaxPooling1D()(X)
#     # X = MaxPooling1D(pool_size=3)(X)      # an alternative to global max pooling
#     # X = Flatten()(X)
#     return X
#
# # A model using Long Short Term Memory (LSTM) Units
# def lstm_model(**kwargs):
#     emb_layer = Embedding(vocab_size, embedding_dim, input_length=max_len, trainable=trainable)
#     X = LSTM(hidden_units, kernel_initializer='he_normal', activation='tanh',
#              dropout=dropout_val, return_sequences=True)(emb_layer)
#     X = LSTM(hidden_units, kernel_initializer='he_normal', activation='tanh',
#              dropout=dropout_val, return_sequences=True)(X)
#     X = Flatten()(X)
#     return X
#
# #
# # def build_LSTM_network():
# #     model = Sequential()
# #     # first layer in the LSTM is the embedding layer - 300dim word embedding
# #     model.add(Embedding(input_dim=-10, output_dim=128, input_length=50))
# #     model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
# #     model.add(Dense(1, activation='sigmoid'))
# #     model.compile(loss='binary_crossentropy',
# #                   optimizer='adam',
# #                   metrics=['accuracy'])
# #     return model
#
# # This is the precise architecture as Ghosh has proposed in his paper "Fracking Sarcasm using Neural Network" (2016)
# def cnn_lstm_model(**kwargs):
#     X = Conv1D(hidden_units, 3, kernel_initializer='he_normal', padding='valid', activation='relu')(kwargs['embeddings'])
#     X = Conv1D(hidden_units, 3, kernel_initializer='he_normal', padding='valid', activation='relu')(X)
#     X = LSTM(hidden_units, kernel_initializer='he_normal', activation='tanh',
#              dropout=kwargs['dropout'], return_sequences=True)(X)
#     X = LSTM(kwargs['hidden_units'], kernel_initializer='he_normal', activation='tanh',
#              dropout=kwargs['dropout'])(X)
#     X = Dense(kwargs['hidden_units'], kernel_initializer='he_normal', activation='sigmoid')(X)
#     return X
#
# def get_LSTM():
#     return KerasClassifier(build_fn=lstm_model,
#                            epochs=10,
#                            batch_size=batch_size,
#                            verbose=0)

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
plt.style.use('dark_background')
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, GlobalMaxPooling1D, SpatialDropout1D