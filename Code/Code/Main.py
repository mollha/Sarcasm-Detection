import spacy
import pandas as pd
import time
from Code.Code.MLmodels import SupportVectorMachine
from sklearn.model_selection import train_test_split
from Code.Code.DataPreprocessing import data_cleaning
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from Code.Code.glove_vectors import GloVeConfig
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from spacy.tokenizer import Tokenizer
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
    data['vector'] = glove_embeddings.get_vectorized_data()
    # TODO need to make this cope with the scenario that no words in a sentence belong to glove dictionary
    print('Finished Vectorizing.')

    print('Total time: ', time.time() - start)

    # try and use our SVM
    print('Training ML models')
    labels = data['sarcasm_label']
    training_data, testing_data, training_labels, testing_labels = train_test_split(vector.apply(pd.Series), labels, test_size=0.3)


    svm = SupportVectorMachine()
    svm.train(training_data, training_labels)
    accuracy = svm.accuracy(testing_data, testing_labels)
    f1_sc = svm.f1(testing_data, testing_labels)
    print('Accuracy: ', accuracy)
    print('F1 score: ', f1_sc)


    exit()

    # data['clean_data'] = data['clean_data'].apply(data_cleaning)
    # vectorizer = CountVectorizer(min_df=5, max_df=0.9, stop_words='english', lowercase=True,
    #                              token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')
    # data_vectorized = vectorizer.fit_transform(data['clean_data'])
    # print(data_vectorized)
    # print('Finished Data Cleaning')


    # try and use our SVM
    print('Training ML models')
    labels = data['sarcasm_label']
    data = data['vector'].apply(pd.Series)
    training_data, testing_data, training_labels, testing_labels = train_test_split(data, labels, test_size=0.3)

    # svm = SupportVectorMachine()
    # svm.train(training_data, training_labels)
    # score = svm.score(testing_data, testing_labels)

    # svm = SVC(gamma='auto', C=10, kernel='linear')
    # svm.fit(training_data, training_labels)
    # svm.score(testing_data, testing_labels)
    # preds_valid = svm.predict(testing_data)
    # fl_s = f1_score(testing_data, preds_valid)
    # print(fl_s)

    NUM_TOPICS = 10
    lda = LatentDirichletAllocation(n_components=NUM_TOPICS, max_iter=10, learning_method='online', verbose=True)
    lda.fit(training_data, training_labels)
    s = lda.score(testing_data, testing_labels)
    print('s', s)
    # data_lda = lda.fit_transform(data_vectorized)
    # print(data_lda.score((testing_data, testing_labels)))

