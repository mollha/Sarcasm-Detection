import re

def data_cleaning(data, rm_urls=True, rm_punc=None, lower=True, rm_numbers=True, rm_dp_wspc=True):

    def remove_urls(text: str) -> str:
        return re.sub(r'http\S+', '', text)  # remove URLs

    def remove_punctuation(text: str, characters: list) -> str:
        banned_punctuation = set([char for char in '!"#$%&()*+-/:;<=>?@[\\]^_`{|}~'])
        banned_punctuation = banned_punctuation if not characters else set(characters)
        return ''.join(ch for ch in text if ch not in banned_punctuation)  # remove punctuation marks

    def remove_numbers(text: str) -> str:
        return text.replace("[0-9]", "")

    def remove_duplicate_whitespaces(text: str) -> str:
        return ' '.join(data.split())  # remove duplicate whitespaces

    if rm_urls:
        data = remove_urls(data)  # remove URLs

    if rm_punc:
        data = remove_punctuation(data, rm_punc)  # remove punctuation

    if lower:
        data = data.lower()  # convert to lowercase

    if rm_numbers:
        data = remove_numbers(data)  # convert to lowercase

    if rm_dp_wspc:
        data = remove_duplicate_whitespaces(data)  # remove duplicate whitespaces
    return data