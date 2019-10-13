from twitter import *
from Code.TwitterCrawler.config import consumer_key, consumer_secret, access_key, access_secret
api = Api(consumer_key, consumer_secret, access_token_key=access_key, access_token_secret=access_secret)


statuses = api.GetUserTimeline(screen_name="realDonaldTrump",  count=5)
print([s.text for s in statuses])

