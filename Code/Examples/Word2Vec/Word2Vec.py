from gensim.models import word2vec

sentences = [
    'It was the best of times',
    'It was the worst of times',
    'It was the age of wisdom',
    'It was the age of foolishness'
]

for i, sentence in enumerate(sentences):
	tokenized= []
	for word in sentence.split(' '):
		word = word.split('.')[0]
		word = word.lower()
		tokenized.append(word)
	sentences[i] = tokenized

chosen_word = 'best'

model = word2vec.Word2Vec(sentences, workers = 1, size = 2, min_count = 1, window = 3, sg = 0)
similar_word = model.wv.most_similar(chosen_word)[0]
print("Most common word to " + chosen_word + " is: " + str(similar_word[0]))

