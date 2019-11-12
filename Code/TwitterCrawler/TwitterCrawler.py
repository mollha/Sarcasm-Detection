from twitter import *
from Code.TwitterCrawler.config import consumer_key, consumer_secret, access_key, access_secret
api = Api(consumer_key, consumer_secret, access_token_key=access_key, access_token_secret=access_secret)


def get_tweets_by_id(id_string):
    try:
        tweet = api.GetStatus(id_string)
        return tweet.AsDict()['text']
    except TwitterError as e:
        return None