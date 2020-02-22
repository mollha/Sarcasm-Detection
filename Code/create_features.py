from pycorenlp import StanfordCoreNLP
import spacy
import numpy as np
import pandas as pd

nlp = spacy.load('en_core_web_md')


# ---------------------------------------------------------------------
# Start CoreNLP server before using sentiment annotator
# cd stanford-corenlp-full-2018-10-05/
# java -mx2g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer
# ---------------------------------------------------------------------

class SentimentAnnotator:
    def __init__(self):
        self.nlp_wrapper = StanfordCoreNLP('http://localhost:9000')
        self.settings = {'annotators': 'sentiment',
                         'outputFormat': 'json',
                         'timeout': 1000,
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
        all_sentences = self.nlp_wrapper.annotate(string, properties=self.settings)["sentences"]
        sentiment_values = []

        for sentence in all_sentences:
            token_list = [token['originalText'] for token in sentence['tokens']]
            sentiment_values.append(self.sentence_level(token_list) + [int(sentence["sentimentValue"])])
            # print(self.sentence_level(token_list) + [int(sentence["sentimentValue"])])

        return list(np.mean(sentiment_values, axis=0))


if __name__ == '__main__':
    dataset_paths = ["Datasets/Sarcasm_Amazon_Review_Corpus", "Datasets/news-headlines-dataset-for-sarcasm-detection"]
    path_to_dataset_root = dataset_paths[1]
    data = pd.read_csv(path_to_dataset_root + "/processed_data/CleanData.csv", encoding="ISO-8859-1")

    annotator = SentimentAnnotator()
    print(annotator.annotate_data("This is a string"))

    # data['sentiment_features'] = data.apply(lambda x: annotator.annotate_data(x))
    # print(data['sentiment_features'])
