from twitter import *
from Code.Datasets.ptacek.processing_scripts.TwitterCrawler.config import settings
api = Api(settings['consumer_key'], settings["consumer_secret"],
          access_token_key=settings["access_key"], access_token_secret=settings["access_secret"],
          sleep_on_rate_limit=True)


def get_tweets_by_id(id_string):
    try:
        tweet = api.GetStatus(id_string)
        return tweet.AsDict()['text']
    except TwitterError as e:
        return None