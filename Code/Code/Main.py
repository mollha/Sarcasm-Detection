import spacy
import pandas as pd
import time
from Code.Code.LSTM import get_LSTM
from Code.Code.MLmodels import *
from Code.Code.create_vectors import *
import ast
from sklearn.model_selection import train_test_split
from Code.Code.DataPreprocessing import data_cleaning
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from Code.Code.glove_vectors import GloVeConfig
from sklearn.model_selection import cross_val_score
import numpy as np

nlp = spacy.load('en_core_web_md')

if __name__ == '__main__':
    start = time.time()

    # -------------------------- READING AND CLEANING DATA ------------------------------
    re_run_cleaning = False
    # Data cleaning has already been applied
    data = pd.read_csv("Datasets/Sarcasm_Amazon_Review_Corpus/processed_data/OriginalData.csv", encoding="ISO-8859-1")

    if re_run_cleaning:
        print('Starting Data Cleaning...')
        data['clean_data'] = data['text_data'].apply(data_cleaning)
    else:
        data['clean_data'] = pd.read_csv("Datasets/Sarcasm_Amazon_Review_Corpus/processed_data/CleanData.csv",
                                         encoding="ISO-8859-1")

    # --------------------------- TOKENIZE AND VECTORIZE -------------------------------
    re_run_token_vector = False


    if re_run_token_vector:
        print('GloVe Vectorizing...')
        token_data = data['clean_data'].apply(lambda x: [token.text for token in nlp(x)])  # tokenizing sentences
        glove_embeddings = GloVeConfig(token_data)
        vector = glove_embeddings.get_vectorized_data()  # my glove embeddings
        # TODO need to make this cope with the scenario that no words in a sentence belong to glove dictionary
        vector.to_csv(path_or_buf="Datasets/Sarcasm_Amazon_Review_Corpus/processed_data/Vectors/glove_vectors.csv",
                      index=False, header=['vector'])

        # print('BOW Vectorizing...')
        # token_data = data['clean_data'].apply(lambda x: " ".join([token.text for token in nlp(x)]))  # tokenizing sentences
        # vector = bag_of_words(token_data)
    else:
        # glove vectors
        # vector = pd.read_csv("Datasets/Sarcasm_Amazon_Review_Corpus/processed_data/Vectors/glove_vectors.csv",
        #                      encoding="ISO-8859-1")['vector']
        vector = pd.read_csv("Datasets/Sarcasm_Amazon_Review_Corpus/processed_data/Vectors/bag_of_words.csv",
                             encoding="ISO-8859-1")['vector']
        vector = vector.apply(lambda x: ast.literal_eval(x))

    # ---------------------------------------------------------------------------------------------------------------

    print('Configuration time: ', time.time() - start)

    # try and use our SVM
    print('Training ML models')
    labels = data['sarcasm_label']
    classifier = get_model(0)
    scores = cross_val_score(classifier, vector.apply(pd.Series), labels, cv=5, scoring='f1_macro')
    five_fold_cross_validation = np.mean(scores)
    print('Score: ', five_fold_cross_validation)