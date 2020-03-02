import time
import pandas as pd
from Code.Main import get_clean_data_col, get_vector_col
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding


# I am using the softmax activation function


if __name__ == '__main__':
    start = time.time()
    dataset_paths = ["Datasets/Sarcasm_Amazon_Review_Corpus", "Datasets/news-headlines-dataset-for-sarcasm-detection"]

    # Choose a dataset from the list of valid data sets
    path_to_dataset_root = dataset_paths[1]
    print('Selected dataset: ' + path_to_dataset_root[9:])

    # Read in raw data
    data = pd.read_csv(path_to_dataset_root + "/processed_data/OriginalData.csv", encoding="ISO-8859-1")[:500]

    # Clean data, or retrieve pre-cleaned data
    data['clean_data'] = get_clean_data_col(data, False)

    # Vectorise data, or retrieve pre-computed vectors
    vector = 'glove'
    print('Vector Type: ' + vector)
    data['vector'] = get_vector_col(data, path_to_dataset_root, vector)

    # embed_dim = len(data['vector'][0])  # length of a single glove vector
    # lstm_out =

    embed_dim = 128
    lstm_out = 200
    batch_size = 32

    model = Sequential()
    model.add(Embedding(2500, embed_dim, input_length=data['vector'].shape[1], dropout=0.2))
    model.add(LSTM(lstm_out, dropout_U=0.2, dropout_W=0.2))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())


    # Create features, or retrieve pre-generated features
    # feature = 'sentiment'
    # print('Feature Type: ' + feature)
    # data['feature'] = get_feature_col(data, path_to_dataset_root, "sentiment")

#    sentiment_evaluate(data)

    # # Use feature INSTEAD of vector
    # data['vector'] = data['feature']

    # ---------------------------------------------------------------------------------------------------------------

    # print('Training ML models')
    # labels = data['sarcasm_label']
    # classifier_name, classifier = get_model(4)
    # print('Classifier: ' + classifier_name)
    #
    # scores = cross_val_score(classifier, data['vector'].apply(pd.Series), labels, cv=2, scoring='f1_macro')
    # five_fold_cross_validation = np.mean(scores)
    # print('Score: ', five_fold_cross_validation)
    # print('Time taken: ' + str(round((time.time() - start)/60, 2)) + ' minutes')