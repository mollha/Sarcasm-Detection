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


def get_model(model_number):
    try:
        return models[model_number]
    except KeyError:
        print('Invalid model number, please enter a number between 0 and ', len(models))
        return None


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