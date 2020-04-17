import sys
import pathlib; base_path = pathlib.Path(__file__).parent.parent.resolve(); sys.path.insert(1, str(base_path))
import spacy
from Code.pkg.model_training.DLmodels import get_dl_results
from Code.pkg.model_training.MLmodels import get_ml_results
from itertools import combinations

nlp = spacy.load('en_core_web_md')


# ------------------------- HELPER FUNCTIONS --------------------------
def run_ml(models: list, vectors: list, data_sets: list, features: list) -> None:
    """
        Given lists of machine learning models, vectors, features and dataset numbers, train, or collect the
        pre-trained model, then return results
        :param models: list of abbreviated model names
        :param vectors: list of vector types
        :param data_sets: list of dataset numbers (ints)
        :param features: list of feature types
        :return: NoneType
    """
    feature_combos = []
    for r in range(0, len(features) + 1):
        combo = combinations(features, r)
        feature_combos += [list(c) for c in combo]

    for dataset in data_sets:  # 0 - AMAZON, 1 - NEWS, 2 - TWEETS
        for model in models:
            for vector in vectors:
                for feature in feature_combos:
                    if vector or len(feature) != 0:
                        get_ml_results(model, vector, feature, dataset)


def run_dl(models: list, vectors: list, data_sets: list) -> None:
    """
    Given lists of deep learning models, vectors and dataset numbers, train, or collect the pre-trained model,
    and return results
    :param models: list of abbreviated model names
    :param vectors: list of vector types
    :param data_sets: list of dataset numbers (ints)
    :return: NoneType
    """
    for dataset in data_sets:  # 0 - AMAZON, 1 - NEWS, 2 - TWEETS
        for model in models:
            for vector in vectors:
                get_dl_results(model, dataset, vector, set_size=10000)
# ---------------------------------------------------------------------


if __name__ == '__main__':
    # --------------------------
    dataset_list = [1]
    # --------------------------

    # ml_model_list = ['svm', 'log_reg', 'rfc', 'n_bayes', 'k_means']
    # ml_model_list = ['log_reg']
    # # ml_vector_list = ['bag_of_words', 'tf_idf', 'glove', 'elmo']

    # ml_feature_list = ['sentiment', 'punctuation', 'topic_model']
    # run_ml(ml_model_list, ml_vector_list, dataset_list, ml_feature_list)

    ml_model_list = ['log_reg']
    ml_vector_list = ['elmo']

    ml_feature_list = ['sentiment', 'topic_model']
    run_ml(ml_model_list, ml_vector_list, dataset_list, ml_feature_list)
    #
    # dl_model_list = ['cnn', 'lstm', 'bi-lstm', 'vanilla-rnn', 'vanilla-gru']
    # dl_vector_list = ['glove', 'elmo']
    # run_dl(dl_model_list, dl_vector_list, dataset_list)

# ---------------------------------------
#     dl_model_list = ['lstm']
#     dl_vector_list = ['glove']
#     run_dl(dl_model_list, dl_vector_list, [1])

    # dl_model_list = ['cnn']
    # dl_vector_list = ['elmo']
    # run_dl(dl_model_list, dl_vector_list, [2])

    # dl_model_list = ['lstm']
    # dl_vector_list = ['glove']
    # run_dl(dl_model_list, dl_vector_list, [0])
    #
    # dl_model_list = ['bi-lstm']
    # dl_vector_list = ['elmo']
    # run_dl(dl_model_list, dl_vector_list, [1])
    #
    # dl_model_list = ['vanilla-rnn']
    # dl_vector_list = ['elmo']
    # run_dl(dl_model_list, dl_vector_list, [1])
    #
    # dl_model_list = ['vanilla-gru']
    # dl_vector_list = ['elmo']
    # run_dl(dl_model_list, dl_vector_list, [1])
    # ------------------------------------
    #
    # dl_model_list = ['vanilla-gru']
    # dl_vector_list = ['glove']
    # run_dl(dl_model_list, dl_vector_list, [1])
