from Code.Datasets.ptacek.processing_scripts.TwitterCrawler.TwitterCrawler import get_tweets_by_id
import re
import time


def quick_clean_string(string: str) -> str:
    string = string.replace('"', "'").strip('\n')
    return string


def create_csv(id_file_name: str, label: str, removed_tweets, start = 0):
    # Open files in read mode
    csv = open("../processed_data/OriginalData.csv", "a", encoding="ISO-8859-1")
    id_file = open("../raw_data/" + id_file_name, "r", encoding="ISO-8859-1").readlines()

    for index in range(start, len(id_file)):
        print('Index: ', index)
        if index % 250 == 0 and index > 0:
            print("\nNumber of tweets downloaded: ", index)
            progress = index / 100000
            print("Progress: " + str(round(progress, 2)))
            time_so_far = (time.time() - start_time) / 60
            print("Time taken so far (minutes): ",  round(time_so_far, 2))
            print('\n')
        id_val = id_file[index]
        id_val = quick_clean_string(id_val)

        token = get_tweets_by_id(id_val)

        # tweet has been removed :(
        if token is None:
            removed_tweets += 1
            continue

        token = re.sub("@.*?(?=\s)", '@user', token)
        token = re.sub("\n", '', token)
        token = quick_clean_string(token)

        line = '\n"' + label + '","' + token + '"'

        try:
            csv.write(line)
        except UnicodeEncodeError:
            removed_tweets += 1
            continue

    csv.close()
    return removed_tweets


if __name__ == "__main__":
    start_time = time.time()
    open("../processed_data/OriginalData.csv", 'w', encoding="ISO-8859-1").close()
    csv = open("../processed_data/OriginalData.csv", "a", encoding="ISO-8859-1")
    csv.write('sarcasm_label,text_data')
    csv.close()

    rm_tweets = create_csv("normal.txt", "0", 0)
    rm_tweets = create_csv("sarcastic.txt", "1", 0, start=rm_tweets)
    print(str(rm_tweets) + ' tweets have been removed :(')
