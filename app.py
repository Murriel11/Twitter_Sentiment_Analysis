from flask import Flask, render_template, flash, request, url_for, redirect, session
import numpy as np
import re
import os
import tensorflow as tf
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
import pickle
from numpy import array

IMAGE_FOLDER = os.path.join('static', 'img_pool')

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = IMAGE_FOLDER

with open('sem.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


def init():
    global model, graph
    graph = tf.Graph()


@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template("index.html")


@app.route('/sentiment_prediction', methods=['POST', "GET"])
def sent_anly_prediction():
    if request.method == 'POST':
        text = request.form['text']

        tw = tokenizer.texts_to_sequences([text])
        tw = sequence.pad_sequences(tw, maxlen=200)
        # vector = np.array([tw.flatten
        with graph.as_default():
            # load the pre-trained Keras model
            model = load_model('sentiment_analysis.h5')

            probability = model.predict(tw)[0][0]
            prediction = int(model.predict(tw).round().item())
            # probability = model.predict(array([vector][0]))[0][0]
            # prediction = model.predict_classes(array([vector][0]))[0][0]

        if prediction == 0:
            sentiment = 'Negative'
            img_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'sad.gif')
        else:
            sentiment = 'Positive'
            img_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'happy.gif')
    return render_template('index.html', text=text, sentiment=sentiment, probability=probability, image=img_filename)


# init()
# app.run(debug=True)
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)

# from flask import Flask, render_template, flash, request, url_for, redirect, session
# import numpy as np
# import pandas as pd
# import re
# import os
# import tensorflow as tf
# from numpy import array
# from keras.datasets import imdb
# from keras.preprocessing import sequence
# from keras.models import load_model
#
# IMAGE_FOLDER = os.path.join('static', 'img_pool')
#
# app = Flask(__name__)
#
# app.config['UPLOAD_FOLDER'] = IMAGE_FOLDER
#
# def init():
#     global model,graph
#     # load the pre-trained Keras model
#     model = load_model('spam.h5')
#     graph = tf.Graph()
#
# #########################Code for Sentiment Analysis
# @app.route('/', methods=['GET', 'POST'])
# def home():
#
#     return render_template("home.html")
#
# @app.route('/sentiment_analysis_prediction', methods = ['POST', "GET"])
# def sent_anly_prediction():
#     if request.method=='POST':
#         text = request.form['text']
#         Sentiment = ''
#         max_review_length = 500
#         word_to_id = imdb.get_word_index()
#         strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
#         text = text.lower().replace("<br />", " ")
#         text=re.sub(strip_special_chars, "", text.lower())
#
#         words = text.split() #split string into a list
#         x_test = [[word_to_id[word] if (word in word_to_id and word_to_id[word]<=20000) else 0 for word in words]]
#         x_test = sequence.pad_sequences(x_test, maxlen=500) # Should be same which you used for training data
#         vector = np.array([x_test.flatten()])
#         with graph.as_default():
#             probability = model.predict(array([vector][0]))[0][0]
#             class1 = model.predict_classes(array([vector][0]))[0][0]
#         if class1 == 0:
#             sentiment = 'Negative'
#             img_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'Sad_Emoji.png')
#         else:
#             sentiment = 'Positive'
#             img_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'Smiling_Emoji.png')
#     return render_template('home.html', text=text, sentiment=sentiment, probability=probability, image=img_filename)
# #########################Code for Sentiment Analysis
#
# if __name__ == "__main__":
#     init()
#     app.run()
