import spacy
import pandas as pd
import time
from Code.Code.MLmodels import my_SVM
from Code.Code.DataPreprocessing import data_cleaning

nlp = spacy.load('en_core_web_md')




if __name__ == '__main__':
    # produce Spacy glove embeddings
    start = time.time()

    data = pd.read_csv("Datasets/Sarcasm_Amazon_Review_Corpus/Data.csv", encoding="ISO-8859-1")

    print('Starting Data Cleaning...')
    data['clean_data'] = data['clean_data'].apply(data_cleaning)
    print('Finished Data Cleaning')

    print('Vectorizing...')
    def vector_func(x):
        return nlp(x).vector

    data['vector'] = data['clean_data'].apply(vector_func)
    print('Finished Vectorizing...')
    print('Total time: ', time.time() - start)

    # try and use our SVM
    print('Training ML models')
    labels = data['sarcasm_label']
    data = data['vector'].apply(pd.Series)
    my_SVM(data, labels)




""" MANUAL GLOVE CONFIG
s_data = pd.DataFrame()
sarcastic_data = pd.read_csv("Ironic.csv", encoding="ISO-8859-1")
sarcastic_data['title_and_review'] = sarcastic_data["title"] + '. ' + sarcastic_data["review"]

print('Starting Data Cleaning...')
s_data['data'] = sarcastic_data['title_and_review'].apply(tokenize)
s_data['label'] = 1

r_data = pd.DataFrame()
regular_data = pd.read_csv("Regular.csv", encoding="ISO-8859-1")
regular_data['title_and_review'] = regular_data["title"] + '. ' + regular_data["review"]
r_data['data'] = regular_data['title_and_review'].apply(tokenize)
print('Finished Data Cleaning')
r_data['label'] = 0
combined_data = pd.concat([r_data, s_data])
glove_embeddings = GloVeConfig(combined_data)
"""