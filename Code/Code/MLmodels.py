from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB


models = {0: SVC(gamma='auto', C=10, kernel='linear'),
          1: LogisticRegression(C=10),
          2: RandomForestClassifier(n_estimators=100, max_depth=None, max_features='sqrt'),
          3: GaussianNB(),
          4: MultinomialNB()}


def calculate_f1_score(trained_model, testing_data, testing_labels):
    predictions = trained_model.predict(testing_data)
    return f1_score(testing_labels, predictions)


def train_and_evaluate(untrained_model, data: tuple):
    # data is a tuple containing (training_data, testing_data, training_labels, testing_labels)
    training_data, testing_data, training_labels, testing_labels = data

    trained_model = untrained_model.fit(training_data, training_labels)
    accuracy = trained_model.score(testing_data, testing_labels)
    f1 = calculate_f1_score(trained_model, testing_data, testing_labels)
    return trained_model, accuracy, f1
