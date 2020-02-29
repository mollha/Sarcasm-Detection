import pickle
import ast
import pandas as pd


if __name__ == '__main__':
    dataset_paths = ["Datasets/Sarcasm_Amazon_Review_Corpus", "Datasets/news-headlines-dataset-for-sarcasm-detection"]
    path_to_root = dataset_paths[1]
    feature = 'sentiment'

    open(path_to_root + "/processed_data/Features/" + feature + ".pckl", 'wb').close()
    store_in = open(path_to_root + "/processed_data/Features/" + feature + ".pckl", 'ab')

    csv = pd.read_csv(path_to_root + "/processed_data/Features/" + feature + ".csv", encoding="ISO-8859-1", header=None)[0]
    csv['lists'] = csv.apply(lambda string: ast.literal_eval(string))

    pickle.dump(csv['lists'], store_in)
    store_in.close()
