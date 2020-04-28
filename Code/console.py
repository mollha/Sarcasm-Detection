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
    template = '<span class="barcode"; style="color: black; background-color: {}">{}</span>'
    colored_string = '<div style="max-width: 300px; overflow: auto;">'
    for t, color in zip(token_list, color_array):
        # if negative, set to white
        color_val = rgb2hex((1, 1, 1)) if color < 0 else rgb2hex(cmap(color)[:3])
        colored_string += template.format(color_val, '&nbsp' + t + '&nbsp')

    colored_string += '</div>'

    if prediction:
        prediction_template = '<div class ="slidecontainer", style="{}"> <input type = "range" min = "0" max = "100" value = "{}" style="{}" class ="slider" id="myRange" list="tickmarks" ></div>'
        style = '<style>.slider::-webkit-slider-thumb {-webkit-appearance: none; border-radius: 50%; appearance: none; width: 100%;' \
                 ' height: 5px; background: #f0f0f0; cursor: default;}</style>'
        outer_css = '-webkit-appearance: none; width: 220px; margin-top: 10px; background:#D8D8D8;	background: linear-gradient(to right, #ff4c38, #ffff66, #85ff93);'
        inner_css = '-webkit-appearance: none; height: 2px; width: 5px; outline: none; margin-left: {}px;'
        script = "<script>" \
                 "document.addEventListener('DOMContentLoaded', (event) => {\
                 var slider = document.getElementById('myRange');\
                 slider.value=parseInt(document.getElementById('myRange').value);});" \
                 "</script>"

        colored_string += style
        colored_string += prediction_template.format(outer_css, prediction, inner_css.format(214 * prediction)
)
        colored_string += '<span style="font-size: 10px"> Prediction score: ' + str(round(prediction, 2)) + '</span>'
        colored_string += script

    # save in html file and open in browser
    with open('colorise.html', 'w') as f:
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


    # attention_output = get_full_attention(sequence)  # 1 x 150
    attention_weights, context_vectors = attention_output.pop(0)
    attention_weights = attention_weights[0]

    # attention_weights = np.clip(attention_weights, np.min(attention_weights), np.quantile(attention_weights, 0.98))


    #print(attention_weights.shape)
    # attention_weights = (attention_weights - attention_weights.min())/(attention_weights.max()-attention_weights.min())
    #attention_weights = np.interp(attention_weights, (attention_weights.min(), attention_weights.max()), (attention_weights.min(), min(0.8, attention_weights.max()*20)))
    print(attention_weights)
    #attention_weights = np.interp(attention_weights, (attention_weights.min(), attention_weights.max()), (attention_weights.min()/2, attention_weights.max()/2))
    #attention_weights = attention_weights.clip(min=0)

    # TODO remove this at the end
    list_array = attention_weights.tolist()
    tuple_list = []
    for val in range(len(tokens)):
        attention_val = list_array[val]
        token = tokens[val]
        tuple_list.append((attention_val, token))
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
        get_prediction(sentence, model, vector_type, model_name, dataset_number)

        c = input('Continue? y / n\n')
        if c == 'n':
            break
