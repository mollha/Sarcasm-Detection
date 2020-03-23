import pandas as pd
from spacy.lang import en


def split_positive_and_negative_samples(dataset: pd.DataFrame) -> tuple:
    """
    Given a dataset, separate the samples into positive (sarcastic) and negative (non-sarcastic) samples
    :param dataset: dataset containing a column called sarcasm_label, this column must contain integers
    :return:
    """
    return dataset[dataset['sarcasm_label'] == 1], dataset[dataset['sarcasm_label'] == 0]


def count_distinct_tokens(dataset: pd.DataFrame, tokenizer: en.English) -> int:
    """
    Given a data frame and a tokenizer, count the number of distinc
    :param tokenizer:
    :param dataset:
    :return:
    """
    token_set = set()
    total = len(dataset['clean_data'])

    def extract_tokens(x: str) -> None:
        token_set.update({token.text for token in tokenizer(x)})

    for index, item in enumerate(dataset['clean_data']):
        extract_tokens(item)

        # if not index % 100:
        #     progress = index / total
        #     print(100 * round(progress, 2), '%')

    return len(token_set)


def sentiment_evaluate(data: pd.DataFrame):
    """
    Given a DataFrame containing the columns data['feature'] and data['sarcasm_label'], deduce the difference
    in sentiment of the sarcastic and non-sarcastic data
    :param data: pandas DataFrame
    :return:
    """
    positive_samples, negative_samples = split_positive_and_negative_samples(data)

    positive_feature_df = positive_samples['feature'].apply(pd.Series)
    negative_feature_df = negative_samples['feature'].apply(pd.Series)

    print('Positive features: ', positive_feature_df.mean(axis=0).tolist())
    print('Negative features: ', negative_feature_df.mean(axis=0).tolist())


