# Import all packages
import streamlit as st
import pickle
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# Build the user interface
st.set_page_config(page_title='Restaurant Review',layout='wide')

# Add title to the body
st.title('Restaurant Review - Sindhura N')

# Add inputs for user
review_input = st.text_input("Write your Review: ")

# Add a button to predict
submit = st.button("Predict whether the review is positive or negative")

# Load the pickle files: tfidf file and model files. Using try except block to avoid errors
try:
    # Loading tfidf vectorizer
    with open(r"Restaurant Review\mtf.pkl","rb") as file1:
        tfidf = pickle.load(file1)
    # Loading neural network model
    with open(r"Restaurant Review\model.pkl","rb") as file2:
        model = pickle.load(file2)
except FileNotFoundError:
    st.error(f"Error: File not found.")
except (pickle.PickleError, AttributeError, TypeError) as e:
    st.error(f"Error loading pickle file: {e}")    


# Logic for prediction
# If submit button is pressed
if submit:
    if review_input.strip():  # Check if review_input is not empty
    # convert the data before feeding to the model
        review_input = review_input.lower()
        review_input = re.sub(r'[^a-z\s]','',review_input)
        X_new = tfidf.transform([review_input]).toarray()
        probs1 = model.predict(X_new)

        # Assuming probs1 is the probability of positive review
        if probs1>=0.5:
            st.subheader("The given review is Positive ")
        else:
            st.subheader("The given review is Negative ")
    else:
        st.warning("Please enter a review before predicting.")