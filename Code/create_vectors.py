import pandas as pd
import spacy
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import tensorflow as tf
import tensorflow_hub as hub
import time
import numpy as np
tf.compat.v1.disable_eager_execution()

nlp = spacy.load('en_core_web_md')


class ElMoVectorizer:
    def __init__(self):
        self.elmo_module = hub.Module("https://tfhub.dev/google/elmo/2", trainable=False)
        self.dataset = None
        self.vector_list = []
        self.step = 5

    def elmo_vectors(self, tokens, session):
        embeddings = self.elmo_module(tokens, signature="default", as_dict=True)["elmo"]
        return session.run(tf.reduce_mean(embeddings, 1))

    def fit_transform(self, dataset: pd.Series):
        self.dataset = dataset
        dataset_size = len(self.dataset)
        initial_time = time.time()

        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            sess.run(tf.compat.v1.tables_initializer())

            for index in range(0, dataset_size, self.step):
                print('\nProgress: ', str(round((index / dataset_size)*100, 3)) + '%')

                time_taken = round(time.time() - initial_time, 2)
                print('Time: ', time_taken)

                if index + self.step >= dataset_size:
                    elmo_train = self.elmo_vectors(data['clean_data'].iloc[index:].tolist(), sess)
                else:
                    elmo_train = self.elmo_vectors(data['clean_data'].iloc[index:index + self.step].tolist(), sess)

                for vector in elmo_train:
                    self.vector_list.append(vector)
        return self.vector_list


class GloVeVectorizer:
    def __init__(self):
        # ----------------- CONFIGURE DATASET -------------------
        self.dataset = None
        self.glove_dict = None

    def fit_transform(self, dataset: pd.Series):
        self.dataset = dataset
        if self.glove_dict is None:
            self.glove_dict = self.refresh_dict()
        return self.vectorize()

    @staticmethod
    def refresh_dict() -> dict:
        print('Building Glove Dictionary....')
        with open('Datasets/GLOVEDATA/glove.twitter.27B.50d.txt', "r", encoding="utf-8") as file:
            gloveDict = {line.split()[0]: list(map(float, line.split()[1:])) for line in file}
            del gloveDict['0.45973']    # for some reason, this entry has 49 dimensions instead of 50
        return gloveDict

    def check_proportion(self):
        all_in, total_in, total_not = set(), 0, 0
        for line in self.dataset:
            for token in line:
                if token in self.glove_dict:
                    all_in.add(token)
                    total_in += 1
                else:
                    total_not += 1
        return total_in, total_not

    def print_stats(self):
        print('\n---------------- Results ------------------')
        total_found, total_not_found = self.check_proportion()
        print('Total tokens found: ', total_found)
        print('Total tokens not found: ', total_not_found)
        print('Percentage found: ', 100 * round((total_found / (total_found + total_not_found)), 4))
        print('Percentage not found: ', 100 * round((total_not_found / (total_found + total_not_found)), 4))

    def get_glove_embedding(self, token: str) -> list:
        return [] if token not in self.glove_dict else self.glove_dict[token]

    def vectorize(self):
        print('Vectorizing textual data')

        def get_mean_embedding(row: list) -> list:
            tokenized_row = [self.get_glove_embedding(token) for token in row]
            valid_row = [list_value for list_value in tokenized_row if list_value]
            if len(valid_row) == 0:
                raise Exception('No words from ' + str(row) + ' could be found in the glove dictionary')
            zipped_values = list(zip(*valid_row))
            return [sum(value) / len(zipped_values) for value in zipped_values]

        return self.dataset.apply(lambda x: get_mean_embedding(x))


vectorisers = {'bag_of_words': CountVectorizer(), 'tf_idf': TfidfVectorizer(), 'glove': GloVeVectorizer(),
               'elmo': ElMoVectorizer()}


def sparse_vectors(path_to_root: str, data: pd.DataFrame, vector: str):
    # vectors can be bag_of_words or tf_idf
    open(path_to_root + "/processed_data/Vectors/" + vector + ".pckl", 'wb').close()
    store_in = open(path_to_root + "/processed_data/Vectors/" + vector + ".pckl", 'ab')
    vectoriser = vectorisers[vector]

    print('4')
    array = vectoriser.fit_transform(data['clean_data'])
    # array = vectoriser.fit_transform(data['token_data'])

    if type(array) != list:
        if type(array) != pd.Series:
            array = array.toarray()
        array = array.tolist()
    pickle.dump(array, store_in)
    store_in.close()


if __name__ == '__main__':
    path_to_dataset_root = "Datasets/Sarcasm_Amazon_Review_Corpus"
    # path_to_dataset_root = "Datasets/news-headlines-dataset-for-sarcasm-detection"
    data = pd.read_csv(path_to_dataset_root + "/processed_data/CleanData.csv", encoding="ISO-8859-1")

    # data['token_data'] = data['clean_data'].apply(lambda x: [token.text for token in nlp(x)])  # tokenizing sentences
    sparse_vectors(path_to_dataset_root, data, 'elmo')
