import sys
from warnings import filterwarnings; filterwarnings('ignore')
import pathlib; base_path = pathlib.Path(__file__).parent.parent.resolve(); sys.path.insert(1, str(base_path))
from Code.console import get_attention, get_trained_model
from flask import Flask, render_template, request, jsonify, redirect

app = Flask(__name__, template_folder='static')

@app.route('/')
def login():
    """
    Configure route to home page
    :return: Render login page
    """
    return render_template('index.html')


@app.route('/attention', methods=['GET'])
def attention():
    print('here')
    print(request.args)
    text = request.args['text']
    print(text)

    if text is not None:
        print(text)
        print(get_attention(text))
        attention_map = get_attention(text)
        return attention_map
        #return render_template("index.html", data=attention_map)




# user_id_str = request.args.get('user')
#     try:
#         user_id = int(user_id_str.strip())
#         all_users = recommender.get_all_users()
#         if user_id in all_users:        # render user dashboard if a valid user id is provided
#             return render_template('recommendations.html', user=user_id_str)
#         return redirect("/")       # if user id is invalid, redirect to the login page
#     except AttributeError:
#         return redirect("/")       # if no user id is provided, redirect to the login page