import sys
import pathlib; base_path = pathlib.Path(__file__).parent.parent.resolve(); sys.path.insert(1, str(base_path))
from warnings import filterwarnings; filterwarnings('ignore')
import numpy as np
from pathlib import Path
from Code.pkg.model_training.DLmodels import prepare_pre_vectors, load_model_from_file, get_custom_layers
from Code.pkg.data_processing.cleaning import data_cleaning
from matplotlib.colors import rgb2hex
from matplotlib.cm import get_cmap
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
    template = '<span class="barcode"; style="color: black; font-size: 13px; background-color: {}">{}</span>'
    colored_string = '<div style="max-width: 175px; overflow: auto;">'
    for t, color in zip(token_list, color_array):
        # if negative, set to white
        color_val = rgb2hex((1, 1, 1)) if color < 0 else rgb2hex(cmap(color)[:3])
        colored_string += template.format(color_val, '&nbsp' + t + '&nbsp')

    colored_string += '</div>'

    if prediction is not None:
        prediction_template = '<div class ="slidecontainer", style="{}"> <input type = "range" min = "0" max = "100" value = "{}" style="{}" class ="slider" id="myRange"></div>'
        style = '<style>.slider::-webkit-slider-thumb {-webkit-appearance: none; border-radius: 0%; appearance: none; width: 0.7px; background: #000000; height: 6px; cursor: default;}</style>'
        outer_css = '-webkit-appearance: none; width: 171px; margin-left: 2px; background:#D8D8D8;	background: linear-gradient(to right, #ff4c38, #ffff66, #85ff93);'
        inner_css = '-webkit-appearance: none; height: 2px; width: 0px; outline: none; margin-left: {}px;'
        labels = '<div style="margin-left: 0px; height:10px; float: left; font-weight: bold; font-family: Arial, Helvetica, sans-serif; text-align: left;"><label style = "font-size: 4px;">▲ </label><br><label style = "font-size: 2.5px;">NON-SARCASTIC</label>' \
        '</div><div style = "margin-left: 60.35px; height:10px; float: left; font-weight: bold; font-family: Arial, Helvetica, sans-serif; text-align: center;"> <label style = "font-size: 4px;">▲ </label> <br> <label style = "font-size: 2.5px;"> NEUTRAL </label>'\
        '</div><div style = "margin-left: 66px; height:10px; float: left; font-weight: bold; font-family: Arial, Helvetica, sans-serif; text-align: right;"><label style = "font-size: 4px;">▲ </label> <br> <label style = "font-size: 2.5px; text-align: center;">' \
        'SARCASTIC&nbsp; </label></div>'

        prediction_score = str(round(prediction, 2))
        if len(prediction_score) == 3:
            prediction_score += '0'

        colored_string += style
        colored_string += '<p style="font-size: 6px; margin-left: 2px; margin-top: 10px; margin-bottom: 2px; font-family: \'Times New Roman\', Times, serif;"> Prediction score: ' + prediction_score + '</p>'
        colored_string += prediction_template.format(outer_css, prediction, inner_css.format(170 * prediction))
        colored_string += labels

    # save in html file and open in browser
    with open('colorise.html', 'w', encoding="utf-8") as f:
        f.write(colored_string)
    print('A visualisation is now available at colorise.html')
    return colored_string


def get_prediction(text: str, trained_model, v_type: str, m_name: str, d_num: int):
    tokens, sequence = prepare_pre_vectors(text, v_type, d_num, m_name)
    get_full_embeddings = K.function([trained_model.layers[0].input], [trained_model.layers[1].output])
    get_attention_from_embedding = K.function([trained_model.layers[2].input], [trained_model.layers[3].output])
    get_prediction_from_embedding = K.function([trained_model.layers[2].input], [trained_model.layers[5].output])

    embedding_output = get_full_embeddings(sequence)
    attention_output = get_attention_from_embedding(embedding_output)
    prediction = np.mean(get_prediction_from_embedding(embedding_output))
    print('Prediction: ', prediction)
    attention_weights, context_vectors = attention_output.pop(0)
    attention_weights = attention_weights[0]
    return visualise(tokens, attention_weights[:len(tokens)], prediction)


def get_trained_model(v_type: str, m_name: str, d_num: int):
    base_path = Path(__file__).parent

    file_name = str(base_path / (
                'pkg/trained_models/' + m_name + '_with_' + v_type + '_on_' + str(d_num) + '.h5'))
    custom_layers = get_custom_layers(m_name, v_type)
    trained_model = load_model_from_file(file_name, custom_layers)
    trained_model._make_predict_function()
    return trained_model


if __name__ == "__main__":
    model_name = 'attention-bi-lstm'
    dataset_number = 2
    vector_type = 'elmo'

    model = get_trained_model(vector_type, model_name, dataset_number)

    while True:
        sentence = input('Type sentence:\n')
        cleaned_sentence = data_cleaning(sentence)
        get_prediction(cleaned_sentence, model, vector_type, model_name, dataset_number)

        c = input('Continue? y / n\n')
        if c == 'n':
            break
