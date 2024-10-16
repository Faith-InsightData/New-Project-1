import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import pandas as pd
import numpy as np

# Load the sentiment analysis model
model_path = 'C:/Users/HP/Downloads/New Project 1/ref/model.pkl'
try:
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Preprocessing function
def preprocess_text(text):
    if pd.isnull(text):
        return ''
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    return ' '.join(tokens)

# Streamlit app layout
st.title("Heart Prediction Sentiment Analysis")

# User input for review
user_input = st.text_area("Enter a Heart prediction:")

if st.button("Predict Sentiment"):
    if user_input:
        # Preprocess the input text
        cleaned_text = preprocess_text(user_input)
        
        # Predict sentiment
        try:
            prediction = model.predict([cleaned_text])
            confidence = np.max(model.predict_proba([cleaned_text])) * 100
            # Output prediction
            sentiment = "Positive" if prediction[0] == 1 else "Negative"
            st.success(f"The sentiment of the review is: {sentiment} with {confidence:.2f}% confidence")
        except Exception as e:
            st.error(f"Error in prediction: {e}")
    else:
        st.warning("Please enter a review before predicting.")

# Optional: Show sample data for demonstration
if st.checkbox("Show sample data"):
    sample_data = pd.DataFrame({
        'text_column': [
            'Hello, world! This is a test.',
            'NLTK is great for text processing.',
            'Preprocessing text data is essential.',
            'Python makes it easy to work with text.',
            'Let\'s clean this text data!'
        ]
    })
    
    # Apply preprocessing to the sample data
    sample_data['cleaned_text'] = sample_data['text_column'].apply(preprocess_text)
    st.write(sample_data)

# Optional: Show histogram of text lengths
if st.checkbox("Show text length histogram"):
    text_lengths = sample_data['cleaned_text'].apply(len)
    st.bar_chart(text_lengths)
