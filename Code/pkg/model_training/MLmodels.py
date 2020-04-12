from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from ..data_processing.helper import prepare_data
import pandas as pd
import numpy as np
import pickle
import time
from pathlib import Path

models = {"svm": SVC(gamma='auto', C=10, kernel='linear'),
          "log_reg": LogisticRegression(C=10, max_iter=300),
          "rfc": RandomForestClassifier(n_estimators=100, max_depth=None, max_features='sqrt'),
          "n_bayes": GaussianNB(),
          "knn": KNeighborsClassifier(n_neighbors=5)
          }


full_names = {"svm": "Support Vector Machine",
              "log_reg": "Logistic Regression",
              "rfc": "Random Forest Classifier",
              "n_bayes": "Gaussian Naive Bayes",
              "knn": "K-Nearest Neighbour"
              }


def get_model(model_name) -> tuple:
    try:
        return full_names[model_name], models[model_name]
    except KeyError:
        print('Invalid model name')



# def get_model(model_name) -> tuple:
#     if model_name == 'svm':
#         return full_names[model_name], SVC(gamma='auto', C=10, kernel='linear')
#     elif model_name == 'log_reg':
#         return full_names[model_name], LogisticRegression(C=10, max_iter=300)
#     elif model_name == 'rfc':
#         return full_names[model_name], RandomForestClassifier(n_estimators=100, max_depth=None, max_features='sqrt')
#     elif model_name == 'n_bayes':
#         return full_names[model_name], GaussianNB()
#     elif model_name == 'knn':
#         return full_names[model_name], KNeighborsClassifier()
#     else:
#         raise KeyError('Not a valid model name')


def calculate_f1_score(trained_model, testing_data, testing_labels):
    predictions = trained_model.predict(testing_data)
    return f1_score(testing_labels, predictions)


def train_and_evaluate(classifier, data, labels):
    scores = cross_val_score(classifier, data.apply(pd.Series), labels, cv=5, scoring='f1_macro')
    return np.mean(scores)


def get_ml_results(model_name: str, vector_type: str, feature_list: list, dataset_number: int):
    base_path = Path(__file__).parent
    start = time.time()
    _, sarcasm_labels, sarcasm_data, _, _ = prepare_data(dataset_number, vector_type, feature_list)

    print('\nTraining ML models')
    classifier_name, classifier = get_model(model_name)
    print('Classifier: ' + classifier_name)

    scores = cross_val_score(classifier, sarcasm_data, sarcasm_labels, cv=2, scoring='f1_macro')
    five_fold_cross_validation = np.mean(scores)
    print('Score: ', five_fold_cross_validation)
    print('Time taken: ' + str(round((time.time() - start)/60, 2)) + ' minutes')

    classifier.fit(sarcasm_data, sarcasm_labels)

    with open(str(base_path / ('../trained_models/' + model_name + '_with_' + '_'.join([vector_type] + feature_list) +
                               '_on_' + str(dataset_number) + '.pckl')), 'wb') as f:
        pickle.dump(classifier, f)
