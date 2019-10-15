from re import sub
from Code.TwitterCrawler import TwitterCrawler  # Use to pull in a few tweets

# Cleans data by removing URLs
# Using re (regular expression library)
# Dealing with human data, therefore it may be worthwhile replacing acronyms with their actual meaning
# e.g. BTW -> By the way
# Some form of spellchecking would be good - incorrectly spelled words could damage the capability of the model
# Possibly remove, or convert, hashtags (would be good to split them into actual words)
# May have the issue of the susan boyle album party incident (susanalbumparty)

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