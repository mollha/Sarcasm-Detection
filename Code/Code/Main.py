import spacy
import pandas as pd
import time
from Code.Code.MLmodels import *
from sklearn.model_selection import train_test_split
from Code.Code.DataPreprocessing import data_cleaning
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from Code.Code.glove_vectors import GloVeConfig
from sklearn.svm import SVC
from spacy.tokenizer import Tokenizer
from sklearn.model_selection import cross_val_score
import numpy as np

nlp = spacy.load('en_core_web_md')
gb_token = Tokenizer(nlp.vocab)


def tokenize(x):
    #return gb_token(x)
    return x.split()

if __name__ == '__main__':
    start = time.time()
    # --------- READING DATA ----------
    data = pd.read_csv("Datasets/Sarcasm_Amazon_Review_Corpus/Data.csv", encoding="ISO-8859-1")

    print('Starting Data Cleaning...')
    data['clean_data'] = data['text_data'].apply(data_cleaning)
    print('Finished Data Cleaning.')

    print('Tokenizing...')
    token_data = data['clean_data'].apply(lambda x: tokenize(x))  # tokenizing sentences
    print('Finished Tokenizing.')

    print('Vectorizing...')
    glove_embeddings = GloVeConfig(token_data)
    vector = glove_embeddings.get_vectorized_data()  # my glove embeddings
    # vector = data['text_data'].apply(lambda x: nlp(x).vector)   # spaCy glove embeddings
    # data['vector'] = glove_embeddings.get_vectorized_data()
    # TODO need to make this cope with the scenario that no words in a sentence belong to glove dictionary
    print('Finished Vectorizing.')

    print('Total time: ', time.time() - start)

    # try and use our SVM
    print('Training ML models')
    labels = data['sarcasm_label']
    clf = MultinomialNB()
    # scores = cross_val_score(clf, data['vector'], labels, cv=5)
    scores = cross_val_score(clf, vector.apply(pd.Series), labels, cv=5, scoring='f1_macro')
    five_fold_cross_validation = np.mean(scores)
    print('Score: ', five_fold_cross_validation)

    #
    # training_data, testing_data, training_labels, testing_labels = train_test_split(vector.apply(pd.Series), labels, test_size=0.3)
    #
    # # lda_with_svm((training_data, testing_data, training_labels, testing_labels))
    #
    #
    # untrained_model = get_model(0)
    # trained_model, accuracy, f1_score_val = train_and_evaluate(untrained_model, (training_data, testing_data, training_labels, testing_labels))
    # print('Accuracy: ', accuracy)
    # print('F1 score: ', f1_score_val)