# TODO make k-fold cross validation
import ast
from Code.MLmodels import *
import spacy

nlp = spacy.load('en_core_web_md')

def get_dataset_size(data_chunk_list: list) -> int:
    """
    Given a list of data chunks, return the total number of data points in the dataset
    :param data_chunk_list: list of TextFileReader objects
    :return: int representing total length of dataset
    """
    return sum([chunk.shape[0] for chunk in data_chunk_list])


def get_predictions(classifier, data_chunk: pd.DataFrame) -> pd.DataFrame:
    classifier.fit(data_chunk)
    return classifier.predict()

    pass


def k_fold_cross_validation(k: int, data_chunk_list: list):
    dataset_size = get_dataset_size(data_chunk_list)
    # e.g. we have 3 chunks of data, 500, 500, 383 = 1383
    prelim_fold_size = dataset_size // k
    # if we have 5 folds, we need to split 1383 into 5 = 276
    remainder = dataset_size % k
    # this is the number of chunks we have to add one to = 3
    classifier = get_model(0)
    for index, chunk in enumerate(data_chunk_list):
        print(index)
        print(type(chunk))
        classifier = classifier.partial_fit(chunk['vector'])
        # iterate through each chunk
        # repeatedly take 276 from chunk
        chunk1 = chunk.iloc[:, 100:]
        print(type(chunk1))
        chunk2 = chunk.iloc[:, :100]


if __name__ == "__main__":
    path_to_dataset_root = "Datasets/news-headlines-dataset-for-sarcasm-detection"
    chunk_size = 1000
    ma_list = pd.read_pickle(path_to_dataset_root + "/processed_data/Vectors/glove.pckl")
    print(len(ma_list))
    print(type(ma_list[0]))
    print(ma_list[0].tolist())

    exit()

    original_data_chunks = pd.read_csv(path_to_dataset_root + "/processed_data/Vectors/tf_idf.csv",
                                       encoding="ISO-8859-1", chunksize=chunk_size)
    original_data_chunk_list = [chunk for chunk in original_data_chunks]
    print('Length of list: ', len(original_data_chunk_list))

    for index in range(len(original_data_chunk_list)):
        print(index)
        chunk = original_data_chunk_list[index]
        chunk['vector'] = chunk['vector'].apply(lambda x: ast.literal_eval(x))

    k_fold_cross_validation(3, original_data_chunk_list)
