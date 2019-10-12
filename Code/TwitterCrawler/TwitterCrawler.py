from twitter import *
from config import consumer_key, consumer_secret, access_key, access_secret
import sys
import config
sys.path.append(".")
api = Api(consumer_key, consumer_secret, access_token_key=access_key, access_token_secret=access_secret)


statuses = api.GetUserTimeline(screen_name="realDonaldTrump",  count=5)
print([s.text for s in statuses])

