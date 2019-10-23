from gensim.models import Word2Vec, KeyedVectors

model = KeyedVectors.load_word2vec_format('data/GoogleGoogleNews-vectors-negative300.bin', binary=True)

sentences = [
    'It was the best of times',
    'It was the worst of times',
    'It was the age of wisdom',
    'It was the age of foolishness'
]


for i, sentence in enumerate(sentences):
	print(sentence)
	tokenized= []
	for word in sentence.split(' '):
		word = word.split('.')[0]
		word = word.lower()
		tokenized.append(word)
	sentences[i] = tokenized

print(sentences)

chosen_word = 'best'

model = Word2Vec.Word2Vec(sentences, workers = 1, size = 4, min_count = 1, window = 5, sg = 0)
similar_word = model.wv.most_similar(chosen_word)
print("Most common word to " + chosen_word + " is: " + str(similar_word[0]))

