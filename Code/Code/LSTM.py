from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import ast
from sklearn.model_selection import cross_val_score
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten
from keras.layers import *
from keras.layers.embeddings import Embedding
np.random.seed(7)


def create_embedding_layer(training_data: pd.Series):
    return Embedding(input_dim=len(training_data), output_dim=len(training_data.columns), weights=[training_data], input_length=len(training_data.columns), trainable=False)

# Function to create model, required for KerasClassifier
def create_model():
    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=50, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def conv(data: pd.Series):
    model = Sequential()
    embedding_layer = create_embedding_layer(data)
    model.add(embedding_layer)
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    return model


def lstm_2(data):
    trainable = True
    plot = True
    shuffle = False
    epochs = 50
    batch_size = 256
    embedding_dim = 300
    hidden_units = 256
    dropout = 0.3
    embedding_layer = create_embedding_layer(data)
    X = LSTM(hidden_units, kernel_initializer='he_normal', activation='tanh',
             dropout=dropout, return_sequences=True)
    return X


def lstm_model(data):
    # create a sequential model
    model = Sequential()
    embedding_layer = create_embedding_layer(data)
    model.add(embedding_layer)
    model.add(LSTM(128, activation='tanh', dropout=0.3, kernel_initializer='he_normal'))
    model.add(Dense(1, activation='sigmoid'))
    # loss function binary cross entropy
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def construct_model(dataset: pd.Series):
    return lstm_model(dataset)


if __name__ == '__main__':
    # Data cleaning has already been applied
    data = pd.read_csv("Datasets/Sarcasm_Amazon_Review_Corpus/processed_data/OriginalData.csv", encoding="ISO-8859-1")
    data['clean_data'] = pd.read_csv("Datasets/Sarcasm_Amazon_Review_Corpus/processed_data/CleanData.csv",
                                         encoding="ISO-8859-1")
    vector = pd.read_csv("Datasets/Sarcasm_Amazon_Review_Corpus/processed_data/Vectors/glove_vectors.csv",
                              encoding="ISO-8859-1")['vector']
    vector = vector.apply(lambda x: ast.literal_eval(x))
    labels = data['sarcasm_label']

    training_data, testing_data, training_labels, testing_labels = train_test_split(vector.apply(pd.Series), labels, test_size=0.3)

    def make_conv():
        return construct_model(training_data)

    scaler = MinMaxScaler(feature_range=(0, len(training_data) - 1))
    array = scaler.fit_transform(training_data)
    training_data = pd.DataFrame(list(array))
    model = KerasClassifier(build_fn=make_conv, epochs=6, batch_size=10, verbose=0)
    # scores = cross_val_score(model, vector.apply(pd.Series), labels, cv=5, scoring='f1_macro')
    # five_fold_cross_validation = np.mean(scores)
    # print('Score: ', five_fold_cross_validation)

    model.fit(training_data, training_labels)
    new_array = scaler.fit_transform(testing_data)
    testing_data = pd.DataFrame(list(new_array))
    predictions = model.predict(testing_data)
    print(predictions)
    print(testing_labels)
    score = f1_score(testing_labels, predictions)
    print(score)
