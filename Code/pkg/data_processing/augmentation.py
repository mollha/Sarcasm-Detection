import spacy
import warnings
from warnings import filterwarnings
filterwarnings('ignore')

nlp = spacy.load('en_core_web_md')


def most_similar(word: str, n: int):
    print(nlp.vocab[word].cluster)
    token = nlp.vocab[word]
    queries = [w for w in token.vocab if w.is_lower == token.is_lower and w.prob >= -15]
    by_similarity = sorted(queries, key=lambda w: token.similarity(w), reverse=True)[1:n]
    return [w.lower_ for w in by_similarity]


def synonym_replacement(text: str) -> list:
    tokens = nlp(text)
    # TODO this will not work if there are no adjectives or nouns in the sentence
    token_tuples = [(tokens[i].text, i) for i in range(len(tokens)) if tokens[i].pos_ in {'ADJ', 'NOUN'}]
    texts_only = [t.text for t in tokens]

    if len(token_tuples) > 0:
        replace, position = token_tuples.pop(0)
        synonyms = most_similar(replace, 7)

        modified_sentences = []
        for ind in range(len(synonyms)):
            new_sentence = ' '.join(texts_only[0:position] + [synonyms[ind]])

            if position < len(tokens) - 1:
                new_sentence += ' '
                new_sentence += ' '.join(texts_only[position+1:])

            modified_sentences.append(new_sentence)
        return modified_sentences

print(synonym_replacement('matt is one super cool guy'))