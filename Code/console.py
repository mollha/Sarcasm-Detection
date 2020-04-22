from warnings import filterwarnings; filterwarnings('ignore')
import numpy as np
import pandas as pd
from pathlib import Path
from Code.pkg.model_training.DLmodels import prepare_pre_vectors, load_model_from_file, get_custom_layers
from Code.pkg.vectors.create_vectors import ElMoVectorizer, GloVeVectorizer
from Code.pkg.vectors.create_features import SentimentAnnotator, PunctuationAnnotator
import matplotlib
from matplotlib.colors import rgb2hex
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
from keras.utils import CustomObjectScope
from keras.models import load_model
from keras.layers import Activation
import keras.backend as K


def colorise(token_list: list, color_array: list):
    """
    Given attention weights and tokens, visualise colour map
    :param token_list: list of tokens (strings)
    :param color_array: array of numbers between 0 and 1
    :return:
    """
    cmap = get_cmap('Reds')
    template = '<span class="barcode"; style="color: black; background-color: {}">{}</span>'
    colored_string = ''
    for t, color in zip(token_list, color_array):
        # if negative, set to white
        color_val = rgb2hex((1, 1, 1)) if color < 0 else rgb2hex(cmap(color)[:3])
        colored_string += template.format(color_val, '&nbsp' + t + '&nbsp')

    # save in html file and open in browser
    with open('colorise.html', 'w') as f:
        f.write(colored_string)
    print('A visualisation is now available at colorise.html')
    return colored_string


def get_attention(text: str, trained_model):
    tokens, sequence = prepare_pre_vectors(text, 'glove', 2, 'attention-lstm')

    get_full_attention = K.function([trained_model.layers[0].input], [trained_model.layers[3].output])
    attention_output = get_full_attention(sequence)  # 1 x 150
    attention_weights, context_vectors = attention_output.pop(0)
    attention_weights = attention_weights[0]
    #print(attention_weights)
    #print(attention_weights.shape)
    #attention_weights = (attention_weights - attention_weights.min())/(attention_weights.max()-attention_weights.min())

    attention_weights = np.interp(attention_weights, (attention_weights.min(), attention_weights.max()), (attention_weights.min()/2, attention_weights.max()*2))
    #attention_weights = attention_weights.clip(min=0)

    # TODO remove this at the end
    list_array = attention_weights.tolist()
    tuple_list = []
    for val in range(len(tokens)):
        attention_val = list_array[val]
        token = tokens[val]
        tuple_list.append((attention_val, token))

    return colorise(tokens, attention_weights[:len(tokens)])


def get_prediction(text: str, trained_model, d_num, m_name):
    tokens, sequence = prepare_pre_vectors(text, 'glove', d_num, m_name)
    prediction = trained_model.predict(sequence)
    return prediction


def get_trained_model():
    base_path = Path(__file__).parent
    model_name = 'attention-lstm'
    dataset_number = 1
    vector_type = 'glove'

    file_name = str(base_path / (
                'pkg/trained_models/' + model_name + '_with_' + vector_type + '_on_' + str(dataset_number) + '.h5'))
    custom_layers = get_custom_layers(model_name, vector_type)
    trained_model = load_model_from_file(file_name, custom_layers)
    trained_model._make_predict_function()
    return trained_model


if __name__ == "__main__":
    model = get_trained_model()
    # prediction = get_prediction('example string of text', model, dataset_number, model_name)
    while True:
        sentence = input('Type sentence:\n')
        get_attention(sentence, model)

        c = input('Continue? y/n\n')
        if c == 'n':
            break
