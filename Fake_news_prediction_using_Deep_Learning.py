"""
1. Deployed Application Link: https://fake-news-detection-using-deep-learning-xfbxnzfhqwkpq5alx7ppwg.streamlit.app/
2. Github Code Link: https://github.com/sohail3031/Fake-News-Detection-Using-Deep-Learning

Steps to Run Deployed Application
1. Upload all the model files
2. Upload the tokenizer files

Steps to Run Web Application Locally:
1. Open a terminal
2. Run the following command to run the application locally
3. Command: streamlit run Fake_news_prediction_using_Deep_Learning.py
"""

# importing libraies
import pickle
import nltk
import string
import re

import streamlit as st
import pandas as pd

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from keras_preprocessing.sequence import pad_sequences
from keras.models import load_model

st.set_page_config(page_title="Fake News Prediction", initial_sidebar_state="expanded", page_icon="📰")
st.markdown("""
    <style>
        body {
            background-color: #ea925d;
            color: #000000;
        }
        
        .stApp {
            background-color: #ea925d;
        }
    </style>
""", unsafe_allow_html=True)

# downloading necessary packages
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download('punkt_tab', quiet=True)

# creating stemmer for stemming the text
stemmer = PorterStemmer()

# uploading files
lstm_model_upload = st.file_uploader("Upload LSTM Model", type=["h5"])
cnn_model_upload = st.file_uploader("Upload CNN Model", type=["h5"])
cobra_model_upload = st.file_uploader("Upload COBRA Model", type=["h5"])

lstm_token_upload = st.file_uploader("Upload LSTM Token", type=["pkl"])
cnn_token_upload = st.file_uploader("Upload CNN Token", type=["pkl"])
cobra_token_upload = st.file_uploader("Upload COBRA Token", type=["pkl"])

if lstm_model_upload is not None and cnn_model_upload is not None and cobra_model_upload is not None and lstm_token_upload is not None and cnn_token_upload is not None and cobra_token_upload is not None:
    # loading models
    with open("lstm_model.h5", "wb") as m1:
        m1.write(lstm_model_upload.getbuffer())

    with open("cnn_model.h5", "wb") as m2:
        m2.write(cnn_model_upload.getbuffer())

    with open("cobra_model.h5", "wb") as m3:
        m3.write(lstm_model_upload.getbuffer())

    lstm_model = load_model("lstm_model.h5")
    cnn_model = load_model("cnn_model.h5")
    cobra_model = load_model("cobra_model.h5")

    # loading tokenizer of each model
    lstm_tokenizer = pickle.load(lstm_token_upload)
    cnn_tokenizer = pickle.load(cnn_token_upload)
    cobra_tokenizer = pickle.load(cobra_token_upload)
    # with open("models/lstm_tokenizer.pkl", "rb") as handle:
    #     lstm_tokenizer = pickle.load(handle)

    # with open("models/cnn_tokenizer.pkl", "rb") as handle:
    #     cnn_tokenizer = pickle.load(handle)

    # with open("models/cobra_tokenizer.pkl", "rb") as handle:
    #     cobra_tokenizer = pickle.load(handle)


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
