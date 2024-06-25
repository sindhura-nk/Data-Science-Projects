# Import all packages
import streamlit as st
import pickle
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf1 =  TfidfVectorizer()

# Build the user interface
st.set_page_config(page_title='Restaurant Review',layout='wide')

# Add title to the body
st.title('Restaurant Review - Sindhura N')

# Add inputs for user
review_input = st.text_input("Review: ")

# Add a button to predict
submit = st.button("Predict whether the review is positive or negative")

# Load the pickle files: model files
with open("Restaurant Review\model.pkl","rb") as file1:
    model = pickle.load(file1)

# Logic for prediction
pattern = r'[^a-z\s]'
# If submit button is pressed
if submit:
    # convert the data before feeding to the model
    review_input = review_input.lower()
    review_input = re.sub(pattern,'',review_input)
    X_new = tfidf1.transform([review_input]).toarray()
    probs1 = model.predict(X_new)
    if probs1>=0.5:
        print('The given review is Positive')
    else:
        print('The given review is Negative')