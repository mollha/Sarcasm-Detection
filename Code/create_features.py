from pycorenlp import StanfordCoreNLP
import spacy
import numpy as np
import pandas as pd
import pickle

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

    def annotate_data(self, string: str) -> list:
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


feature_type = {'sentiment': SentimentAnnotator()}


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

    data['feature_col'] = data['clean_data'].apply(lambda x: annotator.annotate_data(x))
    pickle.dump(data['feature_col'], store_in)
    store_in.close()


sa = SentimentAnnotator()
print(sa.annotate_data("i"))
print(sa.annotate_data("love"))
print(sa.annotate_data("when"))
print(sa.annotate_data("my"))
print(sa.annotate_data("train"))
print(sa.annotate_data("is"))
print(sa.annotate_data("late"))
print(sa.annotate_data("!"))
print(sa.annotate_data("i love when my train is late!"))