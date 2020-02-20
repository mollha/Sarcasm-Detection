import time
from Code1.LSTM import *
from Code1.MLmodels import *
from Code1.create_vectors import *
import ast
from Code1.DataPreprocessing import data_cleaning
from sklearn.model_selection import cross_val_score
import numpy as np

nlp = spacy.load('en_core_web_md')

if __name__ == '__main__':
    start = time.time()
    # path_to_dataset_root = "Datasets/Sarcasm_Amazon_Review_Corpus"
    path_to_dataset_root = "Datasets/news-headlines-dataset-for-sarcasm-detection"
    # chunk_size = 500

    # -------------------------- READING AND CLEANING DATA ------------------------------
    re_run_cleaning = False  # Set to False if data cleaning has already been applied
    # original_data_chunks = pd.read_csv(path_to_dataset_root + "/processed_data/OriginalData.csv",
    #                                    encoding="ISO-8859-1", chunksize=chunk_size)
    # original_data_chunk_list = [chunk for chunk in original_data_chunks]

    data = pd.read_csv(path_to_dataset_root + "/processed_data/OriginalData.csv", encoding="ISO-8859-1")

    if re_run_cleaning:
        print('Starting Data Cleaning...')
        # for data in original_data_chunks:
        #     data['clean_data'] = data['text_data'].apply(data_cleaning)
        data['clean_data'] = data['text_data'].apply(data_cleaning)
        print('Data Cleaning complete.\n')
    else:
        # clean_data_chunks = pd.read_csv(path_to_dataset_root + "/processed_data/CleanData.csv",
        #                                 encoding="ISO-8859-1", chunksize=chunk_size)
        data['clean_data'] = pd.read_csv(path_to_dataset_root + "/processed_data/CleanData.csv",
                                        encoding="ISO-8859-1")
        # for index, data in enumerate(clean_data_chunks):
        #     og_chunk = original_data_chunk_list[index]
        #     og_chunk['clean_data'] = data
    # --------------------------- TOKENIZE AND VECTORIZE -------------------------------
    re_run_token_vector = True

    if re_run_token_vector:
        # print('GloVe Vectorizing...')
        # token_data = data['clean_data'].apply(lambda x: [token.text for token in nlp(x)])  # tokenizing sentences
        # glove_embeddings = GloVeConfig(token_data)
        # vector = glove_embeddings.get_vectorized_data()  # my glove embeddings
        # vector.to_csv(path_or_buf=path_to_dataset_root + "/processed_data/Vectors/glove_vectors.csv",
        #               index=False, header=['vector'])

        print('BOW Vectorizing...')
        data['token_data'] = data['clean_data'].apply(lambda x: " ".join([token.text for token in nlp(x)]))
        vector = sparse_vectors(path_to_dataset_root, data, 'bag_of_words')
         # TODO AttributeError: 'list' object has no attribute 'apply' (for cross eval pandas series)

        # print('TFIDF Vectorizing...')
        # data['token_data'] = data['clean_data'].apply(lambda x: " ".join([token.text for token in nlp(x)]))
        # vector = tf_idf(path_to_dataset_root, data)


        # for chunk in original_data_chunk_list:
        #     chunk['token_data'] = chunk['clean_data'].apply(lambda x: " ".join([token.text for token in nlp(x)]))  # tokenizing sentences
        # original_data_chunk_list = tf_idf(path_to_dataset_root, original_data_chunk_list)
    # TODO AttributeError: 'list' object has no attribute 'apply' (for cross eval pandas series)

    else:
        # glove vectors
        # vector_chunk_list = pd.read_csv(path_to_dataset_root + "/processed_data/Vectors/glove_vectors.csv",
        #                      encoding="ISO-8859-1", chunksize=chunk_size)

        # bag of words vectors
        # vector_chunk_list = pd.read_csv(path_to_dataset_root + "/processed_data/Vectors/bag_of_words.csv",
        #                      encoding="ISO-8859-1", chunksize=chunk_size)

        vector_chunk_list = pd.read_csv(path_to_dataset_root + "/processed_data/Vectors/tf_idf.csv",
                                        encoding="ISO-8859-1", chunksize=chunk_size)
        # tf-idf
        for index, data in enumerate(vector_chunk_list):
            og_chunk = original_data_chunk_list[index]
            og_chunk['vector'] = data
            og_chunk['vector'] = og_chunk['vector'].apply(lambda x: ast.literal_eval(x))

    # ---------------------------------------------------------------------------------------------------------------

    print('Configuration time: ', time.time() - start)

    # try and use our SVM
    print('Training ML models')
    labels = data['sarcasm_label']
    classifier = get_model(3)

    scores = cross_val_score(classifier, vector.apply(pd.Series), labels, cv=5, scoring='f1_macro')
    five_fold_cross_validation = np.mean(scores)
    print('Score: ', five_fold_cross_validation)
