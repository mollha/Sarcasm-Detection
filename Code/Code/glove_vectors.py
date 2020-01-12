import time
import pandas as pd
from scipy import spatial


class GloVeConfig:
    def __init__(self, dataset: pd.Series):
        # ----------------- CONFIGURE DATASET -------------------
        print('Configuring Data...')
        self.dataset = dataset
        print('Data Configured Successfully.')
        self.glove_dict = self.refresh_dict()
        print('Glove Dictionary Built Successfully.')
        self.vectorized_data = self.vectorize()
        print('Data Vectorized Successfully....')

        print(self.find_closest_embeddings(self.glove_dict['cheese'], 3))
        self.print_stats()

    @staticmethod
    def refresh_dict() -> dict:
        print('Building Glove Dictionary....')
        first = time.time()
        with open('Datasets/GLOVEDATA/glove.twitter.27B.50d.txt', "r", encoding="utf-8") as file:
            gloveDict = {line.split()[0]: list(map(float, line.split()[1:])) for line in file}
            del gloveDict['0.45973']    # for some reason, this entry has 49 dimensions instead of 50
        # print(Counter([len(gloveDict[key]) for key in gloveDict]))
        print(len(gloveDict), "words in the GLOVE dictionary\n")
        print('Took ' + str(round(time.time() - first, 2)) + ' seconds to construct glove dictionary')
        return gloveDict

    def check_proportion(self):
        all_in, total_in, total_not = set(), 0, 0
        for line in self.dataset:
            for token in line:
                if token in self.glove_dict:
                    all_in.add(token)
                    total_in += 1
                else:
                    total_not += 1
        return total_in, total_not

    def find_closest_embeddings(self, embedding, n=1):
        return sorted(self.glove_dict.keys(),
                      key=lambda word: spatial.distance.euclidean(self.glove_dict[word], embedding))[0:n]

    def print_stats(self):
        print('\n---------------- Results ------------------')
        total_found, total_not_found = self.check_proportion()
        print('Total tokens found: ', total_found)
        print('Total tokens not found: ', total_not_found)
        print('Percentage found: ', 100 * round((total_found / (total_found + total_not_found)), 4))
        print('Percentage not found: ', 100 * round((total_not_found / (total_found + total_not_found)), 4))

    def get_glove_embedding(self, token: str) -> list:
        return [] if token not in self.glove_dict else self.glove_dict[token]

    def vectorize(self):
        print('Vectorizing textual data')

        def get_mean_embedding(row: list) -> list:
            tokenized_row = [self.get_glove_embedding(token) for token in row]
            valid_row = [list_value for list_value in tokenized_row if list_value]
            zipped_values = list(zip(*valid_row))
            return [sum(value) / len(zipped_values) for value in zipped_values]

        return self.dataset.apply(lambda x: get_mean_embedding(x))