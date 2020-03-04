# Import our dependencies
import tensorflow as tf
import pandas as pd
import tensorflow_hub as hub
from Code.DataPreprocessing import *
import os
import re
from sklearn.metrics import f1_score, accuracy_score
from random import randint
from keras import backend as K
from keras.layers import Dense, Input
from keras.models import Model, load_model, Sequential
from keras.engine import Layer
from sklearn.model_selection import train_test_split
import numpy as np

# Initialize session
sess = tf.Session()
K.set_session(sess)

dataset_paths = ["Datasets/Sarcasm_Amazon_Review_Corpus", "Datasets/news-headlines-dataset-for-sarcasm-detection"]

# Choose a dataset from the list of valid data sets
path_to_dataset_root = dataset_paths[1]
print('Selected dataset: ' + path_to_dataset_root[9:])

# Read in raw data
data = pd.read_csv(path_to_dataset_root + "/processed_data/OriginalData.csv", encoding="ISO-8859-1")[:10000]


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
                       encoding="ISO-8859-1")[:10000]


# Clean data, or retrieve pre-cleaned data
data['clean_data'] = get_clean_data_col(data, path_to_dataset_root, False)


class ElmoEmbeddingLayer(Layer):
    def __init__(self, **kwargs):
        self.dimensions = 1024
        self.trainable = True
        super(ElmoEmbeddingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.elmo = hub.Module('https://tfhub.dev/google/elmo/2', trainable=self.trainable,
                               name="{}_module".format(self.name))

        self.trainable_weights += tf.compat.v1.trainable_variables(scope="^{}_module/.*".format(self.name))
        super(ElmoEmbeddingLayer, self).build(input_shape)

    def call(self, x, mask=None):
        result = self.elmo(K.squeeze(K.cast(x, tf.string), axis=1),
                           as_dict=True,
                           signature='default',
                           )['default']
        return result

    def compute_mask(self, inputs, mask=None):
        return K.not_equal(inputs, '--PAD--')

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.dimensions)

def build_model():
    model = Sequential()
    model.add(ElmoEmbeddingLayer(input_shape=(1,), input_dtype="string"))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model



split = data['clean_data'].apply(pd.Series)
train_df, test_df, train_l, test_l = train_test_split(data['clean_data'], data['sarcasm_label'], test_size=0.2)
train_text = [' '.join(t.split()[0:150]) for t in train_df]
print(train_text)
train_text = np.array(train_text, dtype=object)[:, np.newaxis]
train_label = list(train_l)

test_text = [' '.join(t.split()[0:150]) for t in test_df]
test_text = np.array(test_text, dtype=object)[:, np.newaxis]
test_label = list(test_l)

# Build and fit
model = build_model()
model.fit(train_text,
          train_label,
          validation_data=(test_text, test_label),
          epochs=8,
          batch_size=32)
# model.summary()
model.save('ElmoModel.h5')
pre_save_preds = model.predict_classes(test_text)  # predictions before we clear and reload model
score = f1_score(test_l, pre_save_preds)
print(score)

# model = build_model()
# model.load_weights('ElmoModel.h5')