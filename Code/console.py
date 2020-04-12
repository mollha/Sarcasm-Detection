import numpy as np
import pandas as pd
from pathlib import Path
from Code.pkg.model_training.DLmodels import prepare_pre_vectors, ElmoEmbeddingLayer, GloveEmbeddingLayer, AttentionLayer
from Code.pkg.vectors.create_vectors import ElMoVectorizer, GloVeVectorizer
from Code.pkg.vectors.create_features import SentimentAnnotator, PunctuationAnnotator
from keras.utils import CustomObjectScope
from keras.models import load_model
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


def load_model_from_file(filename: str, vector_type: str, model_name: str):
    custom_layers = {}
    if 'attention' in model_name:
        custom_layers['AttentionLayer'] = AttentionLayer

    if vector_type == 'glove':
        custom_layers['GloveEmbeddingLayer'] = GloveEmbeddingLayer
    elif vector_type == 'elmo':
        custom_layers['ElmoEmbeddingLayer'] = ElmoEmbeddingLayer

    with CustomObjectScope(custom_layers):
        return load_model(filename)


def get_attention(text: str, trained_model, d_num, m_name):
    tokens, sequence = prepare_pre_vectors(text, 'glove', 2, 'attention-lstm')

    get_embedding_layer_output = K.function([model.layers[0].input],
                                      [model.layers[0].output])
    get_attention_layer_output = K.function([model.layers[1].input],
                                            [model.layers[2].output])

    sentence_word_embeddings = get_embedding_layer_output(sequence)

    print(len(sentence_word_embeddings[0]))
    print(len(sentence_word_embeddings[0][0]))
    print(len(sentence_word_embeddings))

    for i in range(len(sentence_word_embeddings[0][0])):
        print(sentence_word_embeddings[0][0][i].shape)
        print(model.layers[1].input.shape)
        attention = get_attention_layer_output([sentence_word_embeddings[0][0][i]])
        print(attention)
        print(tokens[i])

    get_attention_layer_output()

    attention = []
    for i in range(len(sentence_word_embeddings[0])):
        single_embedding = np.ndarray(np.ndarray(sentence_word_embeddings[0][i]))

        print([[sentence_word_embeddings[0][i]]])
        attention.append(get_attention_layer_output([single_embedding])[0])
    print(attention)



    print(type(get_embedding_layer_output([sequence])[0]))
    print(get_embedding_layer_output([sequence])[0])

    print(trained_model.layers[2].output)

    get_attention_layer_output = K.function([model.layers[0].input],
                                      [model.layers[2].output])
    layer_output = get_attention_layer_output([sequence])[0]
    print(layer_output)

    #prediction = trained_model.predict(sequence)
    #return prediction


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
        trained_model = load_model_from_file(file_name, vector_type, model_name)
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
    prediction = get_prediction('example string of text', model, dataset_number, model_name)
    print(prediction)
    get_attention('example string of text', model, dataset_number, model_name)