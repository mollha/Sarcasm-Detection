import pandas as pd
from Code.pkg.vectors.create_vectors import ElMoVectorizer, GloVeVectorizer
from Code.pkg.vectors.create_features import SentimentAnnotator, PunctuationAnnotator


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

            # try:
            #     feature_list = feature_list.toarray()
            # except AttributeError:
            #     pass
            #
            # try:
            #     feature_list = feature_list.tolist()
            # except AttributeError:
            #     pass


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


def get_trained_model(path_to_root: str, vector_type: str, feature_types: list, model_name: str):
    ml_models = {"Support Vector Machine",
                 "Logistic Regression",
                 "Random Forest Classifier",
                 "Gaussian Naive Bayes",
                 "K-Means"}
    dl_models = {'cnn'}

    if model_name in ml_models:
        vectoriser = get_fit_vectoriser(path_to_root, vector_type, feature_types)
        # TODO get ml_model with correct feature types
        pass    # it is a machine learning model
    elif model_name in dl_models:
        pass
    else:
        raise ValueError('The model_name argument should be the name of a machine learning or deep learning classifier')


if __name__ == "__main__":
    dataset_paths = ["Datasets/Sarcasm_Amazon_Review_Corpus", "Datasets/news-headlines-dataset-for-sarcasm-detection",
                     "Datasets/ptacek"]

    # Choose a dataset from the list of valid data sets
    path_to_dataset_root = dataset_paths[1]
    print('Selected dataset: ' + path_to_dataset_root[9:])

    vector = 'bag_of_words'
    feature_list = ['sentiment']
    model = 'K-Means'


    model = get_trained_model(path_to_dataset_root, vector, feature_list, model)