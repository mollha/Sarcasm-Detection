from warnings import filterwarnings; filterwarnings('ignore')
import numpy as np
import pandas as pd
from pathlib import Path
from Code.pkg.model_training.DLmodels import prepare_pre_vectors, load_model_from_file, get_custom_layers
from Code.pkg.vectors.create_vectors import ElMoVectorizer, GloVeVectorizer
from Code.pkg.vectors.create_features import SentimentAnnotator, PunctuationAnnotator
import matplotlib
from matplotlib.colors import rgb2hex, Normalize
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
from keras.utils import CustomObjectScope
from keras.models import load_model
from keras.layers import Activation
import keras.backend as K


def visualise(token_list: list, color_array: np.array, prediction=None):
    """
    Given attention weights and tokens, visualise colour map
    :param prediction: value to set slider to
    :param token_list: list of tokens (strings)
    :param color_array: array of numbers between 0 and 1
    :return:
    """
    cmap = get_cmap('Reds')
    template = '<span class="barcode"; style="color: black; background-color: {}">{}</span>'
    colored_string = '<div style="max-width: 300px; overflow: auto;">'
    for t, color in zip(token_list, color_array):
        # if negative, set to white
        color_val = rgb2hex((1, 1, 1)) if color < 0 else rgb2hex(cmap(color)[:3])
        colored_string += template.format(color_val, '&nbsp' + t + '&nbsp')

    colored_string += '</div>'

    if prediction:
        prediction *= 100
        tick_marks = '<datalist style="display: block" id="tickmarks"><option value="0" label="Non-Sarcastic"></option><option value="50" label="Neutral"></option><option value="100" label="Sarcastic"></option></datalist>'
        prediction_template = '<div style="background-color: grey; width: 300px; height: 50px;"><div class ="slidecontainer", style="{}"> <input type = "range" min = "0" max = "100" value = "{}" style="{}" class ="slider" id="myRange" list="tickmarks" ></div></div>'
        style = '<style>.slider::-webkit-slider-thumb {-webkit-appearance: none; border-radius: 50%; appearance: none; width: 100%;' \
                 ' height: 5px; background: #f0f0f0; cursor: default;}</style>'
        outer_css = '-webkit-appearance: none; width: 280px; margin-top: 10px; margin-left: 10px; background:#D8D8D8;	background: linear-gradient(to right, #ff4c38, #ffff66, #85ff93);'
        inner_css = '-webkit-appearance: none; height: 2px; width: 5px; outline: none;'
        script = "<script>" \
                 "document.addEventListener('DOMContentLoaded', (event) => {\
                 var slider = document.getElementById('myRange');\
                 slider.value=parseInt(document.getElementById('myRange').value);});" \
                 "</script>"

        colored_string += style
        #colored_string += tick_marks
        colored_string += prediction_template.format(outer_css, prediction, inner_css)
        colored_string += '<span> Prediction score: ' + str(round(prediction, 2)) + '%</span>'
        colored_string += script

    # save in html file and open in browser
    with open('colorise.html', 'w') as f:
        f.write(colored_string)
    print('A visualisation is now available at colorise.html')
    return colored_string


def get_attention(text: str, trained_model, v_type: str, m_name: str, d_num: int, prediction=None):
    tokens, sequence = prepare_pre_vectors(text, v_type, d_num, m_name)

    get_full_attention = K.function([trained_model.layers[0].input], [trained_model.layers[3].output])
    attention_output = get_full_attention(sequence)  # 1 x 150
    attention_weights, context_vectors = attention_output.pop(0)
    attention_weights = attention_weights[0]
    #print(attention_weights.shape)
    # attention_weights = (attention_weights - attention_weights.min())/(attention_weights.max()-attention_weights.min())
    # attention_weights = np.interp(attention_weights, (attention_weights.min(), attention_weights.max()), (attention_weights.min(), min(0.8, attention_weights.max()*20)))
    attention_weights = np.interp(attention_weights, (attention_weights.min(), attention_weights.max()), (attention_weights.min()/2, attention_weights.max()*2))
    #attention_weights = attention_weights.clip(min=0)

    # TODO remove this at the end
    list_array = attention_weights.tolist()
    tuple_list = []
    for val in range(len(tokens)):
        attention_val = list_array[val]
        token = tokens[val]
        tuple_list.append((attention_val, token))
    return visualise(tokens, attention_weights[:len(tokens)], prediction)


def get_prediction(text: str, trained_model, v_type, d_num, m_name):
    tokens, sequence = prepare_pre_vectors(text, v_type, d_num, m_name)
    return np.mean(trained_model.predict(sequence))


def get_trained_model(v_type: str, m_name: str, d_num: int):
    base_path = Path(__file__).parent

    file_name = str(base_path / (
                'pkg/trained_models/' + m_name + '_with_' + v_type + '_on_' + str(d_num) + '.h5'))
    custom_layers = get_custom_layers(m_name, v_type)
    trained_model = load_model_from_file(file_name, custom_layers)
    trained_model._make_predict_function()
    return trained_model


if __name__ == "__main__":
    model_name = 'attention-bi-gru'
    dataset_number = 1
    vector_type = 'glove'

    model = get_trained_model(vector_type, model_name, dataset_number)
    # prediction = get_prediction('example string of text', model, dataset_number, model_name)
    while True:
        sentence = input('Type sentence:\n')
        prediction = get_prediction(sentence, model, vector_type, dataset_number, model_name)
        print(prediction)
        get_attention(sentence, model, vector_type, model_name, dataset_number, prediction)

        c = input('Continue? y / n\n')
        if c == 'n':
            break
