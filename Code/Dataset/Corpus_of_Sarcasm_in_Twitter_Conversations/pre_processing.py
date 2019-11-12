from Code.TwitterCrawler.TwitterCrawler import get_tweets_by_id
import pandas as pd

def pre_process(dataset_path):
    data_frame = pd.read_csv(dataset_path)
    author_list = data_frame["author"].tolist()
    audience_list = data_frame["audience"].tolist()
    label_list = data_frame["label"].tolist()
    author_tweets = [get_tweets_by_id(str(author)) for author in author_list]
    audience_tweets = [get_tweets_by_id(str(audience)) for audience in audience_list]
    print(author_tweets)
    new_list = []
    for i in range(len(author_tweets)):
        if not author_tweets[i] or not audience_tweets[i]:
            continue
        else:
            new_list.append([author_tweets[i] + '. ' + audience_tweets[i], label_list[i]])
    df = pd.DataFrame(new_list, columns =['tweet', 'label'])
    return df