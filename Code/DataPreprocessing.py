import re
import pandas as pd


def split_positive_and_negative_samples(dataset: pd.DataFrame) -> tuple:
    """
    Given a dataset, separate the samples into positive (sarcastic) and negative (non-sarcastic) samples
    :param dataset: dataset containing a column called sarcasm_label, this column must contain integers
    :return:
    """
    return dataset[dataset['sarcasm_label'] == 1], dataset[dataset['sarcasm_label'] == 0]


def data_cleaning(data_string: str, rm_urls=True, rm_punc=None, lower=True, rm_numbers=True, rm_dp_wspc=True):
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

    def remove_punctuation(text: str, characters: list) -> str:
        banned_punctuation = set([char for char in '!"#$%&()*+-/:;<=>?@[\\]^_`{|}~'])
        banned_punctuation = banned_punctuation if not characters else set(characters)
        return ''.join(ch for ch in text if ch not in banned_punctuation)  # remove punctuation marks

    def remove_numbers(text: str) -> str:
        return text.replace("[0-9]", "")

    def remove_duplicate_whitespaces(text: str) -> str:
        return ' '.join(text.split())  # remove duplicate whitespaces

    if rm_urls:
        data_string = remove_urls(data_string)  # remove URLs

    if rm_punc:
        data_string = remove_punctuation(data_string, rm_punc)  # remove punctuation

    if lower:
        data_string = data_string.lower()  # convert to lowercase

    if rm_numbers:
        data_string = remove_numbers(data_string)  # convert to lowercase

    if rm_dp_wspc:
        data_string = remove_duplicate_whitespaces(data_string)  # remove duplicate whitespaces
    return data_string


def apply_params(x: str):
    settings = {
        "remove_urls": True,
        "remove_punctuation": True,
        "lowercase": True,
        "remove_numbers": True,
        "remove_duplicate_whitespaces": True}

    return data_cleaning(x, settings["remove_urls"], settings["remove_punctuation"],
                         settings["lowercase"], settings["remove_numbers"], settings["remove_duplicate_whitespaces"])
