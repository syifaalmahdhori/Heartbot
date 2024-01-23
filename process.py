import json
import nltk
import time
import random
import string
import pickle
import numpy as np
import pandas as pd
from io import BytesIO
import tensorflow as tf
import IPython.display as ipd
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Model
# from keras.utils.vis_utils import plot_model
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Input, Embedding, LSTM
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Flatten, Dense, GlobalMaxPool1D
# import tensorflow as tf
# from nltk.stem import WordNetLemmatizer
# from tensorflow import keras
# from tensorflow.keras.preprocessing.sequence import pad_sequences

global responses, lemmatizer, tokenizer, le, model, input_shape
input_shape = 14

# import dataset answer
def load_response():
    global responses
    responses = {}
    with open('heartbot2.json', encoding="utf-8") as content:
        data = json.load(content)
    for intent in data['intents']:
        responses[intent['tag']]=intent['responses']

# import model dan download nltk file
def preparation():
    load_response()
    global lemmatizer, tokenizer, le, model
    tokenizer = pickle.load(open('tokenizers.pkl', 'rb'))
    le = pickle.load(open('le.pkl', 'rb'))
    model = tf.keras.models.load_model('chat_model.h5', compile=False)
    lemmatizer = WordNetLemmatizer()
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

# hapus tanda baca
def remove_punctuation(text):
    texts_p = []
    text = [letters.lower() for letters in text if letters not in string.punctuation]
    text = ''.join(text)
    texts_p.append(text)
    return texts_p

# mengubah text menjadi vector
def vectorization(texts_p):
    vector = tokenizer.texts_to_sequences(texts_p)
    vector = np.array(vector).reshape(-1)
    vector = pad_sequences([vector], input_shape)
    return vector

# klasifikasi pertanyaan user
def predict(vector):
    output = model.predict(vector)
    output = output.argmax()
    response_tag = le.inverse_transform([output])[0]
    return response_tag

# menghasilkan jawaban berdasarkan pertanyaan user
def generate_response(text):
    texts_p = remove_punctuation(text)
    vector = vectorization(texts_p)
    response_tag = predict(vector)
    answer = random.choice(responses[response_tag])
    return answer