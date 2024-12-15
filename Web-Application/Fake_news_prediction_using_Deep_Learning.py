# importing libraies
import pickle
import nltk
import string
import re
import os

import streamlit as st
import pandas as pd

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from keras_preprocessing.sequence import pad_sequences
from keras.models import load_model

# downloading necessary packages
# nltk.download("stopwords", quiet=True)
# nltk.download("punkt", quiet=True)
# nltk.download("wordnet", quiet=True)
# nltk.download('punkt_tab', quiet=True)

# creating stemmer for stemming the text
stemmer = PorterStemmer()

# loading models
lstm_model = load_model(os.path.abspath("lstm_model.h5"))
cnn_model = load_model(os.path.abspath("cnn_model.h5"))
cobra_model = load_model(os.path.abspath("cobra_model.h5"))

# loading tokenizer of each model
with open(os.path.abspath("lstm_tokenizer.pkl"), "rb") as handle:
    lstm_tokenizer = pickle.load(handle)

with open(os.path.abspath("cnn_tokenizer.pkl"), "rb") as handle:
    cnn_tokenizer = pickle.load(handle)

with open(os.path.abspath("cobra_tokenizer.pkl"), "rb") as handle:
    cobra_tokenizer = pickle.load(handle)


# function to remove punctuations from the text
def remove_punctuations(data):
    # translator = str.maketrans("", "", string.punctuation)  # creating translator to replace punctuations with "None"

    punctuation_to_remove = string.punctuation.replace("'", "").replace('"', "")
    translator = str.maketrans("", "", punctuation_to_remove)

    return data.translate(translator)


# function to remove stop words
def remove_stop_words(data):
    stop_words = set(stopwords.words("english"))  # returns a list of stop words
    words = data.split()  # split the words
    cleaned_text = " ".join([word for word in words if word not in stop_words]).strip()

    return cleaned_text.strip()


def text_tokenizer(data):
    data = re.sub(r"@\w+|#", "", data)  # removes "@" and "#" from the text
    data = re.sub(r"[^\w\s]", "", data)  # removes punctuations
    text_tokens = word_tokenize(data)  # tokenizing the text
    stop_words = set(stopwords.words("english"))  # removing stopwords

    return " ".join([word for word in text_tokens if word not in stop_words])


# stemming the text
def stemming_text(data):
    return " ".join([stemmer.stem(word) for word in data.split()])


# UI
st.title("Fake News Prediction")

title = st.text_input("Title", placeholder="Enter News Title Here!")
title = fr"""{title}"""
text = st.text_input("Text", placeholder="Enter News Text Here!")
text = fr"""{text}"""

if st.button("Predict"):
    # cleaning the title and text and combining them
    title_new = stemming_text(text_tokenizer(remove_stop_words(remove_punctuations(title.lower()))))
    text_new = stemming_text(text_tokenizer(remove_stop_words(remove_punctuations(text.lower()))))
    combined_text = title_new + " " + text_new
    combined_text_new = stemming_text(text_tokenizer(remove_stop_words(remove_punctuations(combined_text.lower()))))

    # LSTM model
    lstm_sequence = lstm_tokenizer.texts_to_sequences([combined_text_new])
    lstm_padded = pad_sequences(lstm_sequence, maxlen=20, padding="post")
    lstm_prediction = lstm_model.predict(lstm_padded)
    lstm_prediction_class = (lstm_prediction > 0.5).astype("int32")
    lstm_result = "Real" if lstm_prediction_class[0][0] == 1 else "Fake"
    lstm_probability = "{:.2f}".format(lstm_prediction[0][0] * 100)

    # CNN model
    cnn_sequence = cnn_tokenizer.texts_to_sequences([combined_text_new])
    cnn_padded = pad_sequences(cnn_sequence, maxlen=20, padding="post")
    cnn_prediction = cnn_model.predict(cnn_padded)
    cnn_prediction_class = (cnn_prediction > 0.5).astype("int32")
    cnn_result = "Real" if cnn_prediction_class[0][0] == 1 else "Fake"
    cnn_probability = "{:.2f}".format(cnn_prediction[0][0] * 100)

    # COBRA model
    cobra_sequence = cobra_tokenizer.texts_to_sequences([combined_text_new])
    cobra_padded = pad_sequences(cobra_sequence, maxlen=20, padding="post")
    cobra_prediction = cobra_model.predict(cobra_padded)
    cobra_prediction_class = (cobra_prediction > 0.5).astype("int32")
    cobra_result = "Real" if cobra_prediction_class[0][0] == 1 else "Fake"
    cobra_probability = "{:.2f}".format(cobra_prediction[0][0] * 100)

    # Prediction
    results = pd.DataFrame({
        "Model": ["LSTM", "CNN", "COBRA"],
        "Prediction": [lstm_result, cnn_result, cobra_result],
        "Score": [lstm_probability, cnn_probability, cobra_probability]
    })

    results.reset_index(drop=True, inplace=True)
    results.index += 1

    st.write("### Prediction Results")
    st.write(results)

    # Display final result
    if (lstm_result.__eq__("Real") and cnn_result.__eq__("Real")) or (
            lstm_result.__eq__("Real") and cobra_result.__eq__("Real")) or (
            cnn_result.__eq__("Real") and cobra_result.__eq__("Real")):
        st.text("This News is expected to be True")
    else:
        st.text("This News is expected to be False")

st.text("Note: The lower the score, the more chances of the news being false.")
