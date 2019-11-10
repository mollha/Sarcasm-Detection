import pandas as pd
from wordcloud import WordCloud, STOPWORDS
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

from spacy import displacy
import string

from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
from spacy.lang.en import English

dataset_path = 'Sarcasm_Headlines_Dataset.json'
comment_column = 'headline'         # which column contains the main data e.g. comment
drop_columns = ['article_link']     # columns to remove
indicator_column = 'is_sarcastic'   # column which indicates sarcastic or non-sarcastic

data_frame = pd.read_json(dataset_path, lines=True)
for column in drop_columns:
    data_frame = data_frame.drop([column], axis=1)
# Add the word count for each column
data_frame['len'] = data_frame[comment_column].apply(lambda x: len(x.split(" ")))

# nlp = spacy.load('en_core_web_md')
# doc = nlp(data_frame[comment_column][3])
# spacy.displacy.render(doc, style='ent', jupyter=True)


# vectorizer = CountVectorizer(min_df=5, max_df=0.9, stop_words='english', lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')
# data_vectorized = vectorizer.fit_transform(data_frame[comment_column])
# NUM_TOPICS = 10
# lda = LatentDirichletAllocation(n_components=NUM_TOPICS, max_iter=10, learning_method='online',verbose=True)
# data_lda = lda.fit_transform(data_vectorized)
#
# # Functions for printing keywords for each topic
# def selected_topics(model, vectorizer, top_n=10):
#     for idx, topic in enumerate(model.components_):
#         print("Topic %d:" % idx)
#         print([(vectorizer.get_feature_names()[i], topic[i])
#                         for i in topic.argsort()[:-top_n - 1:-1]])
#
#
# print("LDA Model:")
# selected_topics(lda, vectorizer)

# print(data_frame.head())
text = ' '.join(data_frame[comment_column])
wordcloud = WordCloud(stopwords=STOPWORDS).generate(text)
dataset_name = dataset_path[:dataset_path.rfind('.')]
wordcloud.to_file(dataset_name + ".png")

print('\n---------------------- Dataset Summary ----------------------')
print('Average length: ', round(data_frame.loc[:, "len"].mean(), 2))
sarcastic_comments = len(data_frame[data_frame[indicator_column] == 1].index)
non_sarcastic_comments = len(data_frame[data_frame[indicator_column] == 0].index)
total_comments = sarcastic_comments + non_sarcastic_comments
print('Number of comments: ', total_comments)
print('Number of sarcastic comments: ' + str(sarcastic_comments) + ' - ' +
      str(round((sarcastic_comments*100)/total_comments, 2)) + '%')
print('Number of non-sarcastic comments: ' + str(non_sarcastic_comments) + ' - ' +
      str(round((non_sarcastic_comments*100)/total_comments, 2)) + '%')
