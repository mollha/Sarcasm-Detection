import re
import spacy

nlp = spacy.load('en_core_web_md')


def data_cleaning(data_string: str, rm_urls=True, rm_punc=True, lower=True, rm_numbers=True, rm_dp_wspc=True, rm_stop=True):
    """
    Given data as a string and a set of flags, clean data accordingly
    :param data_string:
    :param rm_urls: remove urls
    :param rm_punc: remove punctuation, optional parameter list can be provided of the punctuation to remove
    :param lower: convert text to lowercase
    :param rm_numbers: remove numbers
    :param rm_dp_wspc: remove duplicate whitespaces -> converting to a single whitespace
    :return: cleaned string
    """
    def remove_urls(text: str) -> str:
        return re.sub(r'http\S+', '', text)  # remove URLs

    def remove_punctuation(text: str) -> str:
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        text = re.sub(r'([@][\w_-]+)', '<user>', text)
        text = text.replace("#sarcasm", ' ')
        banned_punctuation = set([char for char in '#$%&()*+-/:;<>[]^_`{|}~'])
        return ''.join(ch for ch in text if ch not in banned_punctuation)  # remove punctuation marks

    def remove_numbers(text: str) -> str:
        return re.sub("[0-9]", "", text)

    def remove_duplicate_whitespaces(text: str) -> str:
        return ' '.join(text.split())  # remove duplicate whitespaces

    def remove_stopwords(text: str) -> str:
        words = [token.text for token in nlp(text) if not token.is_stop]
        return ' '.join(words)

    if rm_urls:
        data_string = remove_urls(data_string)  # remove URLs

    data_string += ' '  # add space so that user mentions are detected

    if lower:
        data_string = data_string.lower()  # convert to lowercase

    if rm_punc:
        data_string = remove_punctuation(data_string)  # remove punctuation



    if rm_numbers:
        data_string = remove_numbers(data_string)  # convert to lowercase

    if rm_dp_wspc:
        data_string = remove_duplicate_whitespaces(data_string)  # remove duplicate whitespaces

    data_string.strip()

    # if rm_stop:
    #     data_string = remove_stopwords(data_string)  # remove stop words
    return data_string


def apply_params(x: str):
    settings = {
        "remove_urls": True,
        "remove_punctuation": True,
        "lowercase": True,
        "remove_numbers": True,
        "remove_duplicate_whitespaces": True,
        "remove_stopwords": True}

    return data_cleaning(x, settings["remove_urls"], settings["remove_punctuation"],
                         settings["lowercase"], settings["remove_numbers"], settings["remove_duplicate_whitespaces"],
                         settings['remove_stopwords'])
