from pycorenlp import StanfordCoreNLP
import spacy
import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from Code.DataPreprocessing import *
from random import randint
import os

nlp = spacy.load('en_core_web_md')


# ---------------------------------------------------------------------
# Start CoreNLP server before using sentiment annotator
# cd stanford-corenlp-full-2018-10-05/
# java -mx1g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer
# note: change the 2g to 1g if space requirements too high
# ---------------------------------------------------------------------


class SentimentAnnotator:
    def __init__(self):
        self.nlp_wrapper = None
        self.settings = {'annotators': 'sentiment',
                         'outputFormat': 'json',
                         'timeout': 1000000,
                         }

    def sentence_level(self, sentence_tokens: list) -> list:
        """
        Given a sentence as a list of tokens, return the breakdown of sentiment values in the sentence
        :param sentence_tokens: A list of tokens
        :return: list of sentiment counts e.g. [0.1, 0.3, 0.2, 0.3, 0.1]
        counts[0:4] --> frequency of tokens with sentiment values 0 - 4
        """
        counts = [0] * 5

        for token in sentence_tokens:
            sentiment_val = int(self.nlp_wrapper.annotate(token,
                                                          properties=self.settings)["sentences"][0]["sentimentValue"])
            counts[sentiment_val] += 1
        return [count / len(sentence_tokens) for count in counts]

    def transform(self, string: str) -> list:
        """
        Given a string, decompose it into sentences and annotate the sentiments of each sentence
        :param string: string of data
        :return: list of sentiment count averages across all sentences
        """
        if self.nlp_wrapper is None:
            self.nlp_wrapper = StanfordCoreNLP('http://localhost:9000')

        all_sentences = self.nlp_wrapper.annotate(string, properties=self.settings)["sentences"]
        sentiment_values = []

        for sentence in all_sentences:
            token_list = [token['originalText'] for token in sentence['tokens']]
            sentiment_values.append(self.sentence_level(token_list) + [int(sentence["sentimentValue"])])
            # print(self.sentence_level(token_list) + [int(sentence["sentimentValue"])])

        return list(np.mean(sentiment_values, axis=0))

    def fit_transform(self, data: pd.Series):
        return data.apply(lambda x: self.transform(x)).to_numpy()


class PunctuationAnnotator:
    @staticmethod
    def transform(string: str) -> list:
        """
        Given a string, decompose it into sentences and annotate each sentence
        :param string: string of data
        :return: frequency of ! and ? characters in sentence
        """
        tokens = [token.text for token in nlp(string)]
        counts = [0] * 2

        # TODO check percentage capitalisation

        for token in tokens:
            counts[0] += token.count('!')
            counts[1] = token.count('?')

            # if not token.islower():
            #     counts[2] += 1

        return [count / len(tokens) for count in counts]

    def fit_transform(self, data: pd.Series):
        return data.apply(lambda x: self.transform(x)).to_numpy()


class TopicModelAnnotator:
    def __init__(self):
        self.num_topics = 20
        self.max_iterations = 1000
        self.bow_vectoriser = None
        self.lda = None

    @staticmethod
    def lemmatisation(text: str) -> str:
        return " ".join([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in nlp(text) if
                         token.pos_ in ('NOUN', 'ADJ', 'VERB', 'ADV')])

    def transform(self, string: str) -> list:
        vector = self.bow_vectoriser.transform(string)
        topic_distribution = self.lda.transform(vector)
        return topic_distribution.tolist()

    def fit_transform(self, data: pd.Series):
        # apply lemmatisation
        lemmatised_data = data.apply(lambda x: self.lemmatisation(x))
        self.bow_vectoriser = CountVectorizer(stop_words='english', lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')
        vectorised_data = self.bow_vectoriser.fit_transform(lemmatised_data)

        self.lda = LatentDirichletAllocation(n_components=self.num_topics, max_iter=self.max_iterations, learning_method='online',
                                             verbose=True)
        topic_distribution = self.lda.fit_transform(vectorised_data)

        print("\n----- LDA Model ------")
        top_n = 10

        words = self.bow_vectoriser.get_feature_names()
        for topic_idx, topic in enumerate(self.lda.components_):
            print("\nTopic #%d:" % topic_idx)
            print(" ".join([words[i]
                            for i in topic.argsort()[:-top_n - 1:-1]]))
        return topic_distribution


feature_type = {'sentiment': SentimentAnnotator(), 'punctuation': PunctuationAnnotator(),
                'topic_model': TopicModelAnnotator()}


def extract_features(path_to_root: str, data: pd.DataFrame, feature: str):
    """
    Given a particular dataset and feature type, extract these features from the data and store them in a pickle file
    :param path_to_root: insert the path to the correct location to store newly generated features
    :param data: DataFrame containing column called CleanData (strings of data to extract features from)
    :param feature: feature type is a string representing the feature type to extract - valid strings include 'sentiment'
    :return: NoneType
    """
    open(path_to_root + "/processed_data/Features/" + feature + ".pckl", 'wb').close()
    store_in = open(path_to_root + "/processed_data/Features/" + feature + ".pckl", 'ab')
    annotator = feature_type[feature]

    data['feature_col'] = annotator.fit_transform(data['clean_data'])
    pickle.dump(data['feature_col'], store_in)
    store_in.close()

    if feature == 'topic_model':
        annotator_file = open(path_to_root + "/processed_data/Vectorisers/" + feature + "_annotator.pckl", 'ab')
        pickle.dump(annotator, annotator_file)
        annotator_file.close()
#
#
#
# def get_clean_data_col(data_frame: pd.DataFrame, path_to_dataset_root: str, re_clean: bool, extend_path='') -> pd.DataFrame:
#     """
#     Retrieve the column of cleaned data -> either by cleaning the raw data, or by retrieving pre-cleaned data
#     :param data_frame: data_frame containing a 'text_data' column -> this is the raw textual data
#     :param re_clean: boolean flag -> set to True to have the data cleaned again
#     :param extend_path: choose to read the cleaned data at an extended path -> this is not the default clean data
#     :return: a pandas DataFrame containing cleaned data
#     """
#     if re_clean:
#         input_data = ''
#         while not input_data:
#             input_data = input('\nWARNING - This action could overwrite pre-cleaned data: proceed? y / n\n')
#             input_data = input_data.strip().lower() if input_data.strip().lower() in {'y', 'n'} else ''
#
#         if input_data == 'y':
#             # This could potentially overwrite pre-cleaned text if triggered accidentally
#             # The process of cleaning data can take a while, so -> proceed with caution
#             print('RE-CLEANING ... PROCEED WITH CAUTION!')
#             # exit()  # comment this line if you would still like to proceed
#             data_frame['clean_data'] = data_frame['text_data'].apply(data_cleaning)
#             extend_path = '' if not os.path.isfile(path_to_dataset_root + "/processed_data/CleanData.csv") else \
#                 ''.join([randint(0, 9) for _ in range(0, 8)])
#             data_frame['clean_data'].to_csv(
#                 path_or_buf=path_to_dataset_root + "/processed_data/CleanData" + extend_path + ".csv",
#                 index=False, header=['clean_data'])
#     return pd.read_csv(path_to_dataset_root + "/processed_data/CleanData" + extend_path + ".csv", encoding="ISO-8859-1")
#
#
# if __name__ == "__main__":
#     dataset_paths = ["Datasets/Sarcasm_Amazon_Review_Corpus", "Datasets/news-headlines-dataset-for-sarcasm-detection",
#                      "Datasets/ptacek"]
#
#     # Choose a dataset from the list of valid data sets
#     path_to_dataset_root = dataset_paths[1]
#     print('Selected dataset: ' + path_to_dataset_root[9:])
#
#     # Read in raw data
#     data = pd.read_csv(path_to_dataset_root + "/processed_data/OriginalData.csv", encoding="ISO-8859-1")  # [:set_size]
#
#     # Clean data, or retrieve pre-cleaned data
#     data['clean_data'] = get_clean_data_col(data, path_to_dataset_root, False)  # [:set_size]
#
#     extract_features(path_to_dataset_root, data, 'topic_model')
#
