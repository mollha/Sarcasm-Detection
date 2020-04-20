import numpy as np
import pandas as pd
from pathlib import Path
from Code.pkg.model_training.DLmodels import prepare_pre_vectors, load_model_from_file, get_custom_layers
from Code.pkg.vectors.create_vectors import ElMoVectorizer, GloVeVectorizer
from Code.pkg.vectors.create_features import SentimentAnnotator, PunctuationAnnotator
import matplotlib
from matplotlib.colors import rgb2hex
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
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


def colorise(token_list: list, color_array: list):
    """
    Given attention weights and tokens, visualise colour map
    :param token_list: list of tokens (strings)
    :param color_array: array of numbers between 0 and 1
    :return:
    """
    cmap = get_cmap('Reds')
    template = '<span class="barcode"; style="color: black; background-color: {}">{}</span>'
    colored_string = ''
    for t, color in zip(token_list, color_array):
        # if negative, set to white
        color_val = rgb2hex((1, 1, 1)) if color < 0 else rgb2hex(cmap(color)[:3])
        print(color_val)
        colored_string += template.format(color_val, '&nbsp' + t + '&nbsp')
    return colored_string


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

    get_full_attention = K.function([trained_model.layers[0].input], [trained_model.layers[3].output])
    print('Getting output from', trained_model.layers[3].name)



    attention_output = get_full_attention(sequence)  # 1 x 150
    attention_weights, context_vectors = attention_output.pop(0)
    print(attention_weights[0])

    attention_weights = attention_weights[0]
    #attention_weights = (attention_weights - attention_weights.min())/(attention_weights.max()-attention_weights.min())

    attention_weights = np.interp(attention_weights, (attention_weights.min(), attention_weights.max()), (-0.8, 0.8))

    s = colorise(tokens, attention_weights[:len(tokens)])



    attention_weights = attention_weights.clip(min=0)

    # or simply save in an html file and open in browser
    with open('colorise.html', 'w') as f:
        f.write(s)

    list_array = attention_weights.tolist()
    tuple_list = []
    for val in range(len(tokens)):
        attention_val = list_array[val]
        token = tokens[val]
        tuple_list.append((attention_val, token))

    return s


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
