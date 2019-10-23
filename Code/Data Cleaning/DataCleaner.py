from re import sub
import numpy as np
from Code.TwitterCrawler import TwitterCrawler  # Use to pull in a few tweets
import nltk

wpt = nltk.WordPunctTokenizer()
stop_words = nltk.corpus.stopwords.words('english')
print(stop_words)

# def normalize_document(doc):
#     # lower case and remove special characters\whitespaces
#     doc = re.sub(r'[^a-zA-Z\s]', '', doc, re.I|re.A)
#     doc = doc.lower()
#     doc = doc.strip()
#     # tokenize document
#     tokens = wpt.tokenize(doc)
#     # filter stopwords out of document
#     filtered_tokens = [token for token in tokens if token not in stop_words]
#     # re-create document from filtered tokens
#     doc = ' '.join(filtered_tokens)
#     return doc

# normalize_corpus = np.vectorize(normalize_document)



# Cleans data by removing URLs
# Using re (regular expression library)
# Dealing with human data, therefore it may be worthwhile replacing acronyms with their actual meaning
# e.g. BTW -> By the way
# Some form of spellchecking would be good - incorrectly spelled words could damage the capability of the model
# Possibly remove, or convert, hashtags (would be good to split them into actual words)
# May have the issue of the susan boyle album party incident (susanalbumparty)

tweet_list = TwitterCrawler.get_some_tweets()


def clean_data(text):
    text.strip()    # remove whitespaces on the ends
    # Replaces any http links with an empty string
    text = sub(r'http\S+', '', text)

    # Replaces any @ signs with 'at '
    #text = sub(r'@', 'at ', text)

    return text.lower()

cleaned_tweets = [clean_data(tweet) for tweet in tweet_list]
print(tweet_list)
print(cleaned_tweets)