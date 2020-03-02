from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np

models = {0: SVC(gamma='auto', C=10, kernel='linear'),
          1: LogisticRegression(C=10, max_iter=300),
          2: RandomForestClassifier(n_estimators=100, max_depth=None, max_features='sqrt'),
          3: GaussianNB(),
          4: KMeans(n_clusters=2, init='k-means++', max_iter=300, n_init=10, random_state=0)
          }

model_names = {0: "Support Vector Machine",
               1: "Logistic Regression",
               2: "Random Forest Classifier",
               3: "Gaussian Naive Bayes",
               4: "K-Means"
               }


def get_model(model_number) -> tuple:
    try:
        return model_names[model_number], models[model_number]
    except KeyError:
        print('Invalid model number, please enter a number between 0 and ', len(models))


def calculate_f1_score(trained_model, testing_data, testing_labels):
    predictions = trained_model.predict(testing_data)
    return f1_score(testing_labels, predictions)


def train_and_evaluate(classifier, data, labels):
    scores = cross_val_score(classifier, data.apply(pd.Series), labels, cv=5, scoring='f1_macro')
    return np.mean(scores)


def lda_with_svm(data: tuple):
    training_data, testing_data, training_labels, testing_labels = data
    # data['clean_data'] = data['clean_data'].apply(data_cleaning)
    # vectorizer = CountVectorizer(min_df=5, max_df=0.9, stop_words='english', lowercase=True,
    #                              token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')
    # data_vectorized = vectorizer.fit_transform(data['clean_data'])
    # print(data_vectorized)
    # print('Finished Data Cleaning')

    NUM_TOPICS = 100
    # lda = LatentDirichletAllocation(n_components=NUM_TOPICS, max_iter=10, learning_method='online', verbose=True)
    lda = LatentDirichletAllocation(n_components=NUM_TOPICS, max_iter=10, learning_method='batch', verbose=True)

    lda.fit(training_data, training_labels)
    s = lda.score(testing_data, testing_labels)
    print('s', s)
    # data_lda = lda.fit_transform(data_vectorized)
    # print(data_lda.score((testing_data, testing_labels)))
