import spacy
from Code.Dataset.NewReview.glove_vectors import GloVeConfig
import pandas as pd
import re
import time
from Code.Dataset.NewReview.MLmodels import my_SVM

nlp = spacy.load('en_core_web_md')
def data_cleaning(data):
    # data = re.sub(r'http\S+', '', data)  # remove URLs
    punctuation = '!"#$%&()*+-/:;<=>?@[\\]^_`{|}~'
    data = ''.join(ch for ch in data if ch not in set(punctuation))  # remove punctuation marks
    data = data.lower()  # convert to lowercase
    data = data.replace("[0-9]", " ")  # remove numbers
    data = ' '.join(data.split())  # remove whitespaces
    return data


def tokenize(sentence):
    value = data_cleaning(sentence)
    nlp_sentence = nlp(value)
    return [token.norm_ for token in nlp_sentence]


if __name__ == '__main__':
    # produce Spacy glove embeddings
    start = time.time()
    s_data = pd.DataFrame()
    sarcastic_data = pd.read_csv("Ironic.csv", encoding="ISO-8859-1")
    sarcastic_data['title_and_review'] = sarcastic_data["title"] + '. ' + sarcastic_data["review"]

    print('Starting Data Cleaning...')
    s_data['data'] = sarcastic_data['title_and_review'].apply(data_cleaning)
    s_data['label'] = 1

    r_data = pd.DataFrame()
    regular_data = pd.read_csv("Regular.csv", encoding="ISO-8859-1")
    regular_data['title_and_review'] = regular_data["title"] + '. ' + regular_data["review"]
    r_data['data'] = regular_data['title_and_review'].apply(data_cleaning)
    r_data['label'] = 0
    print('Finished Data Cleaning')

    combined_data = pd.concat([r_data, s_data])

    def vector_func(x):
        return nlp(x).vector
    # combined_data['vector'] = combined_data['data'].apply(vector_func)
    sarcastic_data['vector'] = sarcastic_data['data'].apply(vector_func)
    regular_data['vector'] = regular_data['data'].apply(vector_func)

    print('Total time: ', time.time() - start)

    # try and use our SVM
    my_SVM(sarcastic_data, regular_data)




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