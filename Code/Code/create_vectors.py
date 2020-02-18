import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


nlp = spacy.load('en_core_web_md', disable=['parser', 'ner'])


def bag_of_words(path_to_root: str, dataset: pd.Series):
    vectoriser = CountVectorizer()
    x = vectoriser.fit_transform(dataset)
    open(path_to_root + "/processed_data/Vectors/bag_of_words.csv", 'w').close()
    csv = open(path_to_root + "/processed_data/Vectors/bag_of_words.csv", "a")
    # write feature names as the first line
    csv.write('vector')

    array = x.toarray()
    list_of_vectors = array.tolist()
    for vector in list_of_vectors:
        csv.write('\n')
        csv.write('"' + str(vector) + '"')
    return pd.Series(list_of_vectors)


def tf_idf(path_to_root: str, data_chunk_list: list):
    open(path_to_root + "/processed_data/Vectors/tf_idf.csv", 'w').close()
    csv = open(path_to_root + "/processed_data/Vectors/tf_idf.csv", "a")
    csv.write('vector')
    for chunk in data_chunk_list:
        tokens = chunk['token_data']
        vectoriser = TfidfVectorizer()
        x = vectoriser.fit_transform(tokens)

        array = x.toarray()
        list_of_vectors = array.tolist()
        for vector in list_of_vectors:
            csv.write('\n')
            csv.write('"' + str(vector) + '"')
        chunk['vector'] = pd.Series(list_of_vectors)
    return data_chunk_list
