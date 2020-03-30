import spacy
# from warnings import filterwarnings; filterwarnings('ignore')
from Code.MLmodels import *
from Code.create_vectors import compute_vectors
from Code.create_features import extract_features
from Code.evaluate_data import feature_evaluate, count_distinct_tokens
from Code.DataPreprocessing import *
from sklearn.model_selection import cross_val_score
import numpy as np
import os.path
from random import randint
from Code.DLmodels import get_results
import time

nlp = spacy.load('en_core_web_md')


def get_clean_data_col(data_frame: pd.DataFrame, path_to_dataset_root: str, re_clean: bool, extend_path='') -> pd.DataFrame:
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
            # exit()  # comment this line if you would still like to proceed
            data_frame['clean_data'] = data_frame['text_data'].apply(data_cleaning)
            extend_path = '' if not os.path.isfile(path_to_dataset_root + "/processed_data/CleanData.csv") else \
                ''.join([randint(0, 9) for _ in range(0, 8)])
            data_frame['clean_data'].to_csv(
                path_or_buf=path_to_dataset_root + "/processed_data/CleanData" + extend_path + ".csv",
                index=False, header=['clean_data'])
    return pd.read_csv(path_to_dataset_root + "/processed_data/CleanData" + extend_path + ".csv", encoding="ISO-8859-1")


def get_vector_col(data_frame: pd.DataFrame, path_to_root, vector_type: str) -> list:
    """
    Given a vector type, retrieve pre-computed vectors for data, or compute them and return them as a list
    :param data_frame: DataFrame containing 'clean_data' column
    :param path_to_root: path to dataset root -> used for file naming
    :param vector_type: a string representing the type of vectors to produce
    :return: a list of vectors
    """
    valid_vector_types = {'elmo', 'bag_of_words', 'tf_idf', 'glove'}
    vector_type = vector_type.lower().strip()

    if vector_type not in valid_vector_types:
        raise TypeError('Invalid vector type "' + vector_type + '"')

    if not os.path.isfile(path_to_root + "/processed_data/Vectors/" + vector_type + ".pckl"):
        input_data = ''
        while not input_data:
            input_data = input('\nWARNING - This action could overwrite pre-vectorised data: proceed? y / n\n')
            input_data = input_data.strip().lower() if input_data.strip().lower() in {'y', 'n'} else ''

        if input_data == 'y':
            print('RE-VECTORIZING ... PROCEED WITH CAUTION!')
            # exit()  # comment this line if you would still like to proceed
            if vector_type in {'bag_of_words', 'tf_idf', 'glove'}:
                data_frame['token_data'] = data_frame['clean_data'].apply(
                    lambda x: " ".join([token.text for token in nlp(x)]))
                compute_vectors(path_to_root, data_frame,  vector_type)
            elif vector_type == 'elmo':
                compute_vectors(path_to_root, data_frame, vector_type)
    return pd.read_pickle(path_to_root + "/processed_data/Vectors/" + vector_type + ".pckl")


def get_feature_col(data_frame: pd.DataFrame, path_to_root: str, feature_type: str):
    valid_feature_types = {'sentiment', 'punctuation', 'topic_model'}
    feature_type = feature_type.lower().strip()

    if feature_type not in valid_feature_types:
        raise TypeError('Invalid feature type "' + feature_type + '"')

    if not os.path.isfile(path_to_root + "/processed_data/Features/" + feature_type + ".pckl"):
        input_data = ''
        while not input_data:
            input_data = input('\nWARNING - This action could overwrite pre-extracted data: proceed? y / n\n')
            input_data = input_data.strip().lower() if input_data.strip().lower() in {'y', 'n'} else ''

        if input_data == 'y':
            print('RE-VECTORIZING ... PROCEED WITH CAUTION!')
            # exit()  # comment this line if you would still like to proceed
            extract_features(path_to_root, data_frame,  feature_type)
    return pd.read_pickle(path_to_root + "/processed_data/Features/" + feature_type + ".pckl")


def combine_features(feature_columns: list) -> pd.Series:
    if len(feature_columns) <= 1:
        return feature_columns.pop()

    def add(l1, l2) -> list:
        if type(l1) != list:
            l1 = l1.tolist()
        elif type(l2) != list:
            l2 = l2.tolist()
        return l1 + l2

    feature_col = feature_columns[0]
    for index in range(1, len(feature_columns)):
        feature_col = feature_col.combine(feature_columns[index], add)

    return feature_col


if __name__ == '__main__':
    start = time.time()
    dataset_paths = ["Datasets/Sarcasm_Amazon_Review_Corpus", "Datasets/news-headlines-dataset-for-sarcasm-detection", "Datasets/ptacek"]

    # Choose a dataset from the list of valid data sets
    path_to_dataset_root = dataset_paths[0]
    print('Selected dataset: ' + path_to_dataset_root[9:])

    # set_size = 200

    # Read in raw data
    data = pd.read_csv(path_to_dataset_root + "/processed_data/OriginalData.csv", encoding="ISO-8859-1")#[:set_size]

    # Clean data, or retrieve pre-cleaned data
    data['clean_data'] = get_clean_data_col(data, path_to_dataset_root, False)#[:set_size]

    # machine learning models
    # Vectorise data, or retrieve pre-computed vectors
    vector = 'elmo'
    print('Vector Type: ' + vector)
    data['vector'] = get_vector_col(data, path_to_dataset_root, vector)

    # Create features, or retrieve pre-generated features
    features = ['sentiment', 'topic_model']
    print('Feature Types: ' + ','.join(features))
    feature_series = [get_feature_col(data, path_to_dataset_root, feature_type) for feature_type in features]
    data['feature'] = combine_features(feature_series)

    print(data['feature'])
    feature_evaluate(data)

    print(type(data['vector'][0]))

    # Use feature INSTEAD of vector
    data['vector'] = combine_features([data['vector'], data['feature']])



    # ---------------------------------------------------------------------------------------------------------------
    # exit()
    # model_name = 'cnn'
    # dataset_name = path_to_dataset_root[9:]
    # vector_type = 'glove'
    # sarc_data, sarc_labels = data['clean_data'], data['sarcasm_label']
    #
    # get_results(model_name, dataset_name, sarc_data, sarc_labels, vector_type, 0.2)
    # # get_results(model_name, sarc_data, sarc_labels, vector_type, 0.2)
    # print('\nTraining ML models')
    labels = data['sarcasm_label']
    classifier_name, classifier = get_model(1)
    print('Classifier: ' + classifier_name)

    scores = cross_val_score(classifier, data['vector'].apply(pd.Series), labels, cv=2, scoring='f1_macro')
    five_fold_cross_validation = np.mean(scores)
    print('Score: ', five_fold_cross_validation)
    print('Time taken: ' + str(round((time.time() - start)/60, 2)) + ' minutes')