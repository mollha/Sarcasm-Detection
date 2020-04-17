import numpy as np
import pandas as pd
from pathlib import Path
from Code.pkg.model_training.DLmodels import prepare_pre_vectors, load_model_from_file, get_custom_layers
from Code.pkg.vectors.create_vectors import ElMoVectorizer, GloVeVectorizer
from Code.pkg.vectors.create_features import SentimentAnnotator, PunctuationAnnotator
from keras.utils import CustomObjectScope
from keras.models import load_model
from keras.layers import Activation
import keras.backend as K


class Vectoriser:
    def __init__(self, trained_vectorisers: list):
        """
        Create a vectoriser that can concatenate multiple vector and feature types
        :param trained_vectorisers: a list of trained vectorisers, where all vectorisers contain the transform method
        """
        self.trained_vectorisers = trained_vectorisers

    def transform(self, list_of_strings: list):
        list_of_vectors = [[] for _ in range(len(list_of_strings))]

        for vectoriser in self.trained_vectorisers:
            feature_list = vectoriser.transform(list_of_strings).tolist()

            try:
                feature_list = feature_list.toarray()
            except AttributeError:
                pass

            try:
                feature_list = feature_list.tolist()
            except AttributeError:
                pass


            for idx, feature in enumerate(feature_list):
                list_of_vectors[idx] += feature

        return list_of_vectors


def get_fit_vectoriser(path_to_root: str, vector_type: str, features: list):
    vectorisers = [{'glove': GloVeVectorizer(), 'elmo': ElMoVectorizer()}[vector_type]] \
        if vector_type in {'glove', 'elmo'} else [pd.read_pickle(path_to_root + "/processed_data/vectorisers/" +
                                                                 vector_type + "_vectoriser.pckl")]

    vectorisers += [{'sentiment': SentimentAnnotator(), 'punctuation': PunctuationAnnotator()}[ft]
                    if ft in {'sentiment', 'punctuation'} else
                    pd.read_pickle(path_to_root + "/processed_data/vectorisers/" + ft + "_annotator.pckl")
                    for ft in features]

    return Vectoriser(vectorisers)



def get_attention(text: str, trained_model):
    tokens, sequence = prepare_pre_vectors(text, 'glove', 2, 'attention-lstm')

    get_embedding_layer_output = K.function([model.layers[0].input],
                                      [model.layers[0].output])
    get_attention_layer_output = K.function([model.layers[1].input],
                                            [model.layers[2].output])

    get_full_attention = K.function([model.layers[0].input], [model.layers[2].output])

    array = get_full_attention(sequence)  # 1 x 150
    list_array = array[0][0].tolist()
    tuple_list = []
    for val in range(len(tokens)):
        attention_val = list_array[val]
        token = tokens[val]
        tuple_list.append((attention_val, token))
    print(tuple_list)


    # print(get_full_attention([sequence]))

    # sentence_word_embeddings = get_embedding_layer_output([sequence])
    # get_attention_layer_output(array)


    return
    # print(sentence_word_embeddings[0][0])
    # print(len(sentence_word_embeddings))
    # print(len(sentence_word_embeddings[0]))
    # print(len(sentence_word_embeddings[0][0]))

    # -----------------------------------------------
    print(get_embedding_layer_output([sequence])[0])



    # for vec in sentence_word_embeddings[0][0]:
    #     if np.count_nonzero(vec) > 0:
    #         array = np.array([[vec] + [np.zeros(50) for _ in range(149)]])
    #         attention_output = get_attention_layer_output(array)
    #         print(attention_output)

        # print(sentence_word_embeddings[0][0][i].shape)
        # print(model.layers[1].input.shape)
        # attention = get_attention_layer_output([sentence_word_embeddings[0][0][i]])
        # print(attention)
        # print(tokens[i])
    return


def get_prediction(text: str, trained_model, d_num, m_name):
    tokens, sequence = prepare_pre_vectors(text, 'glove', d_num, m_name)
    prediction = trained_model.predict(sequence)
    return prediction


def get_trained_model(path_to_root: str, vector_type: str, feature_types: list, model_name: str, dataset_number: int):
    base_path = Path(__file__).parent
    ml_models = {"Support Vector Machine",
                 "Logistic Regression",
                 "Random Forest Classifier",
                 "Gaussian Naive Bayes",
                 "K-Means"}
    dl_models = {'cnn', 'attention-lstm'}

    if model_name in ml_models:
        vectoriser = get_fit_vectoriser(path_to_root, vector_type, feature_types)
        # TODO get ml_model with correct feature types
        pass    # it is a machine learning model
    elif model_name in dl_models:
        file_name = str(base_path / (
                    'pkg/trained_models/' + model_name + '_with_' + vector_type + '_on_' + str(dataset_number) + '.h5'))
        custom_layers = get_custom_layers(model_name, vector_type)
        trained_model = load_model_from_file(file_name, custom_layers)
        return trained_model
    else:
        raise ValueError('The model_name argument should be the name of a machine learning or deep learning classifier')


if __name__ == "__main__":
    dataset_paths = ["Datasets/Sarcasm_Amazon_Review_Corpus", "Datasets/news-headlines-dataset-for-sarcasm-detection",
                     "Datasets/ptacek"]

    # Choose a dataset from the list of valid data sets
    dataset_number = 2
    path_to_dataset_root = dataset_paths[dataset_number]
    print('Selected dataset: ' + path_to_dataset_root[9:])

    vector = 'glove'
    feature_list = []
    model_name = 'attention-lstm'


    model = get_trained_model(path_to_dataset_root, vector, feature_list, model_name, dataset_number)
    # prediction = get_prediction('example string of text', model, dataset_number, model_name)
    while True:
        sentence = input('Type sentence:\n')
        get_attention(sentence, model)

        c = input('Continue? y/n\n')
        if c == 'n':
            break
