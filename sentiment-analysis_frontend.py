# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 18:18:55 2025

@author: Sunita
"""

import streamlit as st
import re
import pickle
import numpy as np
from nltk.stem import PorterStemmer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Initialize PorterStemmer
ps = PorterStemmer()

# Preprocess function (same as in the notebook)
def preprocess(text):
    # Lowercase
    text = text.lower()
    # Remove special characters, keep only alphabets and spaces
    text = re.sub(r'[^a-z\s]', '', text)
    # Stemming word by word
    text = " ".join(ps.stem(word) for word in text.split())
    return text

# Load the tokenizer from pickle
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load the model from pickle (if saved via pickle.dump(model))
with open('model.pkl', 'rb') as handle:
    model = pickle.load(handle)

# Alternatively, if saved via model.save('model.h5') or 'model.keras', use:
# model = load_model('model.h5')  # or 'model.keras'

# Function to predict sentiment
def predict_sentiment(review):
    # Preprocess the review
    cleaned_review = preprocess(review)
    # Tokenize and pad
    sequence = tokenizer.texts_to_sequences([cleaned_review])  # Fixed line
    padded_sequence = pad_sequences(sequence, maxlen=200)
    # Predict
    prediction = model.predict(padded_sequence)[0][0]
    sentiment = "positive" if prediction > 0.5 else "negative"
    return sentiment, prediction  # Return sentiment and confidence score

# Streamlit interface
st.title("Movie Review Sentiment Analysis")
st.write("Enter a movie review to predict its sentiment (positive or negative).")

# Text input area for the review
user_input = st.text_area("Enter your movie review:", height=150)

# Button to trigger prediction
if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a review before predicting.")
    else:
        sentiment, score = predict_sentiment(user_input)
        st.success(f"**Predicted Sentiment:** {sentiment.capitalize()}")

# Optional: Add a clear button to reset the input
if st.button("Clear"):
    st.session_state.user_input = ""  # Clear the text area
    st.rerun()  # Refresh the app to reflect the cleared input

# Footer
st.markdown("---")
st.markdown("Built with Streamlit. Model trained on IMDB dataset for sentiment analysis.")
