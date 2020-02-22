from pycorenlp import StanfordCoreNLP
import spacy

nlp = spacy.load('en_core_web_md')

nlp_wrapper = StanfordCoreNLP('http://localhost:9000')


def sentiment_annotator(sentence: str) -> list:
    """
    Given a string, return the breakdown of sentiment values in the sentence, as well as the overall sentiment
    :param sentence: A string of tokens
    :return: list of sentiment counts e.g. [0.1, 0.3, 0.2, 0.3, 0.1, 3]
    counts[0:4] --> frequency of tokens with sentiment values 0 - 4
    counts[5] --> sentiment value of sentence
    """
    tokens = [token.text for token in nlp(sentence)]
    counts = [0] * 5

    settings = {'annotators': 'sentiment',
                'outputFormat': 'json',
                'timeout': 1000,
                }

    for doc in tokens:
        sentiment_val = int(nlp_wrapper.annotate(doc, properties=settings)["sentences"][0]["sentimentValue"])
        counts[sentiment_val] += 1

    counts = [count / len(tokens) for count in counts]
    overall_sentiment = int(nlp_wrapper.annotate(sentence, properties=settings)["sentences"][0]["sentimentValue"])
    counts.append(overall_sentiment)
    return counts
