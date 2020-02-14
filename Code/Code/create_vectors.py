import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


nlp = spacy.load('en_core_web_md', disable=['parser', 'ner'])

def bag_of_words(dataset: pd.Series):
    vectoriser = CountVectorizer()
    x = vectoriser.fit_transform(dataset)
    open("Datasets/Sarcasm_Amazon_Review_Corpus/processed_data/Vectors/bag_of_words.csv", 'w').close()
    csv = open("Datasets/Sarcasm_Amazon_Review_Corpus/processed_data/Vectors/bag_of_words.csv", "a")
    # write feature names as the first line
    csv.write('vector')

    array = x.toarray()
    list_of_vectors = array.tolist()
    for vector in list_of_vectors:
        csv.write('\n')
        csv.write('"' + str(vector) + '"')
    return list_of_vectors

def tf_idf(dataset: pd.Series):
    vectoriser = TfidfVectorizer()
    x = vectoriser.fit_transform(dataset)
    open("Datasets/Sarcasm_Amazon_Review_Corpus/processed_data/Vectors/tf_idf.csv", 'w').close()
    csv = open("Datasets/Sarcasm_Amazon_Review_Corpus/processed_data/Vectors/tf_idf.csv", "a")
    # write feature names as the first line
    csv.write('vector')

    array = x.toarray()
    list_of_vectors = array.tolist()
    for vector in list_of_vectors:
        csv.write('\n')
        csv.write('"' + str(vector) + '"')
    return list_of_vectors


# def tf_idf(dataset: pd.Series):
#     dictionary = all_words(dataset)
#     for _, data_point in dataset.iteritems():
#         for x in nlp(data_point):
#             val = x.text
#
#     return dictionary

def term_frequency():
    pass


def vectorize(dataset: pd.Series) -> list:
    """
    Given a dataset, return a list of lists
    where the nested list contains vectors
    :return:
    """
    for _, data in dataset.iteritems():
        pass