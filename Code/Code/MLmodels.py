from sklearn.svm import SVC
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.model_selection import train_test_split

class SupportVectorMachine:
    def __init__(self):
        # self.svm = SVC(gamma='auto', C=10, kernel='linear')
        self.svm = SVC(gamma='auto', C=10, kernel='rbf')

    def train(self, training_data, training_labels):
        self.svm.fit(training_data, training_labels)
        return self.svm

    def accuracy(self, testing_data, testing_labels):
        return self.svm.score(testing_data, testing_labels)

    def f1(self, testing_data, testing_labels):
        predictions = self.svm.predict(testing_data)
        return f1_score(testing_labels, predictions)


def my_SVM(data, data_labels):
    # ------------------------------ SPLIT DATA INTO TRAIN AND TEST -----------------------------------
    # shuffles data and splits
    combined_train, combined_test, combined_train_labels, combined_test_labels = train_test_split(data, data_labels,
                                                                    test_size=0.3)
    model = SVC(gamma='auto', C=10, kernel='linear')
    model = LogisticRegression(C = 10)
    model = RandomForestClassifier(n_estimators = 100, max_depth = None, max_features = 'sqrt')
    model = GaussianNB()
    model.fit(combined_train, combined_train_labels)
    a = model.score(combined_test, combined_test_labels)
    print('score: ', a)


    # regular_train, regular_test, x_train, x_test = train_test_split(regular_data, regular_data['label'],
    #                                                                 test_size=0.3)
    # sarcastic_train, sarcastic_test, sarcastic_train_labels, sarcastic_test_labels = train_test_split(sarcastic_data, sarcastic_data['label'],
    #                                                               test_size=0.3)


    # model.fit(sarcastic_train['vector'], sarcastic_train['label'])
    # a = model.score(sarcastic_test['vector'], sarcastic_train['label'])
    # print('score: ', a)


# MACHINE LEARNING MODELS


# Create a dataFrame with feature names as columns
#
# lreg = LogisticRegression()
# lreg.fit(regular_train, x_train)
# preds_valid = lreg.predict(regular_test)
# print(f1_score(x_test, preds_valid))