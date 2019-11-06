import pandas as pd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
from spacy.lang.en import English

# nlp = spacy.load('en_core_web_md')

dataset_path = 'Sarcasm_Headlines_Dataset.json'
comment_column = 'headline'         # which column contains the main data e.g. comment
drop_columns = ['article_link']     # columns to remove
indicator_column = 'is_sarcastic'   # column which indicates sarcastic or non-sarcastic

data_frame = pd.read_json(dataset_path, lines=True)
for column in drop_columns:
    data_frame = data_frame.drop([column], axis=1)
# Add the word count for each column
data_frame['len'] = data_frame[comment_column].apply(lambda x: len(x.split(" ")))


print(data_frame.head())

print('\n---------------------- Dataset Summary ----------------------')
print('Average length: ', round(data_frame.loc[:, "len"].mean(), 2))
sarcastic_comments = len(data_frame[data_frame[indicator_column] == 1].index)
non_sarcastic_comments = len(data_frame[data_frame[indicator_column] == 0].index)
total_comments = sarcastic_comments + non_sarcastic_comments
print('Number of sarcastic comments: ' + str(sarcastic_comments) + ' - ' +
      str(round((sarcastic_comments*100)/total_comments, 2)) + '%')
print('Number of non-sarcastic comments: ' + str(non_sarcastic_comments) + ' - ' +
      str(round((non_sarcastic_comments*100)/total_comments, 2)) + '%')
