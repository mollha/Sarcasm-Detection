from re import sub
from Code.TwitterCrawler import TwitterCrawler  # Use to pull in a few tweets

# Cleans data by removing URLs
# Using re (regular expression library)


tweet_list = TwitterCrawler.get_some_tweets()


def clean_data(text):
    # Replaces any http links with an empty string
    text = sub(r'http\S+', '', text)

    # Replaces any @ signs with 'at '
    #text = sub(r'@', 'at ', text)

    return text.lower()

cleaned_tweets = [clean_data(tweet) for tweet in tweet_list]
print(tweet_list)
print(cleaned_tweets)