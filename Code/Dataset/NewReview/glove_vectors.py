import time
import pandas as pd
import numpy as np

"""
sarcastic_data = pd.read_csv(path_to_sarcastic, encoding="ISO-8859-1")
sarcastic_data["title_and_review"] = sarcastic_data["title"] + '. ' + sarcastic_data["review"]

regular_data = pd.read_csv(path_to_regular, encoding="ISO-8859-1")
regular_data["title_and_review"] = regular_data["title"] + '. ' + regular_data["review"]
"""


class GloVeConfig:
    def __init__(self, dataset: pd.DataFrame):
        print('Configuring Data...')
        # DATASET must contain a label column with an int, 0 (non-sarcastic) and 1 (sarcastic)
        # Dataset must also contain a column called data, where each cell is a list of cleaned tokens
        # ----------------- CONFIGURE DATASET -------------------
        self.dataset = dataset
        self.sarcastic_data = dataset[dataset['label'] == 1]
        self.regular_data = dataset[dataset['label'] == 0]
        print('Data Configured Successfully.')
        self.glove_dict = self.refresh_dict()
        print('Glove Dictionary Built Successfully.')
        self.vectorized_data = self.vectorize()
        print(self.vectorized_data)
        print('Data Vectorized Successfully....')
        self.print_stats()

    @staticmethod
    def refresh_dict() -> dict:
        print('Building Glove Dictionary....')
        first = time.time()
        with open('GLOVEDATA/glove.twitter.27B.50d.txt', "r", encoding="utf-8") as file:
            gloveDict = {line.split()[0]: np.array(map(float, line.split()[1:])) for line in file}
        print(len(gloveDict), "words in the GLOVE dictionary\n")
        print('Took ' + str(round(time.time() - first, 2)) + ' seconds to construct glove dictionary')
        return gloveDict

    def check_proportion(self, data_type: str):
        if data_type == 'sarcastic':
            data = self.sarcastic_data['data'].tolist()
        elif data_type == 'regular':
            data = self.regular_data['data'].tolist()
        else:
            data = self.sarcastic_data['data'].tolist() + self.regular_data['data'].tolist()

        all_in, total_in, total_not = set(), 0, 0
        for line in data:
            for token in line:
                if token in self.glove_dict:
                    all_in.add(token)
                    total_in += 1
                else:
                    total_not += 1
        return total_in, total_not

    def print_stats(self):
        def get_stats(data_type: str):
            total_found, total_not_found = self.check_proportion(data_type)
            print('Total tokens found: ', total_found)
            print('Total tokens not found: ', total_not_found)
            print('Percentage found: ', 100 * round((total_found / (total_found + total_not_found)), 4))
            print('Percentage not found: ', 100 * round((total_not_found / (total_found + total_not_found)), 4))

        print('\n---------------- Sarcastic Results ------------------')
        get_stats('sarcastic')
        print('\n---------------- Regular Results ------------------')
        get_stats('regular')
        print('\n---------------- Both Results ------------------')
        get_stats('both')

    def vectorize(self):
        print('Vectorizing textual data')

        def individual(vector, glove_dict):
            vector_list = []
            for token in vector:
                if token in glove_dict:
                    vector_list.append(glove_dict[token])
            return vector_list

        return [individual(token, self.glove_dict) for token in self.dataset]