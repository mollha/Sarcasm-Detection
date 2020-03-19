import time
import pandas as pd
from sklearn.metrics import f1_score
from Code.Main import get_clean_data_col, get_vector_col
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, SpatialDropout1D, Dropout, Flatten
from keras.layers.embeddings import Embedding
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM, Bidirectional
from keras.layers import ReLU
from keras.layers import Conv1D
from keras.layers import MaxPool1D, MaxPooling1D
from keras.layers import Dropout
from keras.layers import Input
from keras.layers.embeddings import Embedding


def get_embedding_layer(embed_dim):
    scale_by = 2500
    return Embedding(input_dim=scale_by + 1, output_dim=embed_dim, input_length=50)

def old_model():
    embed_dim = 128
    lstm_out = 128
    model = Sequential()
    model.add(Embedding(input_dim=scale_by + 1, output_dim=embed_dim, input_length=50))
    # model.add(SpatialDropout1D(0.2))
    # model.add(Dropout(0.2))
    # model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2, activation='tanh'))
    model.add(LSTM(lstm_out, return_sequences=True))
    model.add(LSTM(lstm_out, return_sequences=True))
    model.add(LSTM(lstm_out))
    # model.add(Dense(units=lstm_out))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    # model.add(Dense(units=1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    print(model.summary())
    return model




def cnn_lstm_network():
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=3, strides=2, padding='valid', use_bias=False))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(MaxPool1D(pool_size=2, strides=1))

    model.add(LSTM(units=64, dropout=0.5, recurrent_dropout=0.5))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model


def new_model():
    model = Sequential()
    model.add(get_embedding_layer(50))
    model.add(Dense(units=50, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def modely():
    model = Sequential()
    model.add(get_embedding_layer())
    model.add(LSTM(100, return_sequences=True))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

def create_model():
    embed_dim = 128
    lstm_out = 128
    # create model
    model = Sequential()
    model.add(Embedding(input_dim=2500 + 1, output_dim=embed_dim, input_length=50))
    # model.add(SpatialDropout1D(0.2))
    # model.add(Dropout(0.2))
    model.add(LSTM(embed_dim, dropout=0.2, recurrent_dropout=0.2, activation='tanh', kernel_initializer='he_normal'))
    model.add(LSTM(lstm_out, return_sequences=True))
    model.add(LSTM(lstm_out, return_sequences=True))
    # model.add(Dense(units=lstm_out))
    model.add(Dropout(0.5))

    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model


def construct_model(scalar: int):
    embedding_layer = Embedding(input_dim=scalar + 1, output_dim=embed_dim, input_length=50)


if __name__ == '__main__':
    start = time.time()
    dataset_paths = ["Datasets/Sarcasm_Amazon_Review_Corpus", "Datasets/news-headlines-dataset-for-sarcasm-detection"]

    # Choose a dataset from the list of valid data sets
    path_to_dataset_root = dataset_paths[0]
    print('Selected dataset: ' + path_to_dataset_root[9:])

    # Read in raw data
    data = pd.read_csv(path_to_dataset_root + "/processed_data/OriginalData.csv", encoding="ISO-8859-1")

    # Clean data, or retrieve pre-cleaned data
    data['clean_data'] = get_clean_data_col(data,  path_to_dataset_root, False)

    # Vectorise data, or retrieve pre-computed vectors
    vector = 'glove'
    print('Vector Type: ' + vector)
    data['vector'] = get_vector_col(data, path_to_dataset_root, vector)
    vectorised_data = data['vector'].apply(pd.Series)
    print(vectorised_data.head())

    scale_by = 2500

    scaler = MinMaxScaler(feature_range=(0, scale_by))
    vectorised_data = scaler.fit_transform(vectorised_data)
    print(vectorised_data.shape)

    labels = data['sarcasm_label']
    batch_size = 16

    training_data, testing_data, training_labels, testing_labels = train_test_split(vectorised_data, labels, test_size=0.20)
    model = KerasClassifier(build_fn=new_model)
    model.fit(training_data, training_labels, batch_size=16, epochs=10)
    preds_valid = model.predict(testing_data)
    print(preds_valid)

    # loss, score = model.evaluate(testing_data, testing_labels, batch_size=16)
    # print(testing_labels)
    score = f1_score(testing_labels, preds_valid)

    print('Score: %.2f' % score)

    # Create features, or retrieve pre-generated features
    # feature = 'sentiment'
    # print('Feature Type: ' + feature)
    # data['feature'] = get_feature_col(data, path_to_dataset_root, "sentiment")

#    sentiment_evaluate(data)

    # # Use feature INSTEAD of vector
    # data['vector'] = data['feature']

    # ---------------------------------------------------------------------------------------------------------------

    # print('Training ML models')
    # labels = data['sarcasm_label']
    # classifier_name, classifier = get_model(4)
    # print('Classifier: ' + classifier_name)
    #
    # scores = cross_val_score(classifier, data['vector'].apply(pd.Series), labels, cv=2, scoring='f1_macro')
    # five_fold_cross_validation = np.mean(scores)
    # print('Score: ', five_fold_cross_validation)
    # print('Time taken: ' + str(round((time.time() - start)/60, 2)) + ' minutes')