import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import re
import joblib
import base64

# Load the trained Naive Bayes model and CountVectorizer
model = joblib.load('naive_bayes_model.pkl')
vectorizer = joblib.load('count_vectorizer.pkl')

# Function to preprocess and predict hate speech
def predict_hate_speech_proba(text):
    # Preprocess text
    text = text.lower()
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    
    # Vectorize the text
    text_vector = vectorizer.transform([text])
    
    # Predict the probability of being hate speech
    proba = model.predict_proba(text_vector)[0]
    
    return proba[1]  # Probability of being hate speech

# Set the background image
# background_image = """
# <style>
#[data-testid="stAppViewContainer"] > .main {
#    background-image: url("https://images.unsplash.com/photo-1542358935821-e4e9f3f3c15d?q=80&w=2874&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
 #   background-size: 100vw 100vh;  # This sets the size to cover 100% of the viewport width and height
  #  background-position: center;  
   # background-repeat: no-repeat;
#}
# </style>
# """

# st.markdown(background_image, unsafe_allow_html=True)


# Streamlit app
def main():
    st.title('Is Your Post Homo/Transphobic? 🏳️‍🌈')
    st.write('Enter a text below to check the probability of it being hate speech. Minimum 3 words required.')

    # User input
    user_input = st.text_area(':rainbow[Input Text:]', '')

    if st.button('Analyse'):
        if user_input:
            # Check minimum length
            if len(user_input.split()) < 3:
                st.warning('Please enter a text with at least 3 words for analysis.')
            else:
                # Get probability
                probability = predict_hate_speech_proba(user_input)
                
                # Display result
                st.subheader('Analysis Result:')
                st.write(f'Probability of Hate Speech: {probability:.2%}')

                # Display messages based on probability range
                if probability < 0.15:
                    st.success("This looks ok!")
                elif 0.15 <= probability < 0.3:
                    st.warning("This could be considered hate speech, depending on context")
                elif 0.3 <= probability <= 0.7:
                    st.error("This doesn't sound nice :(")
                else:
                    st.error("Why are you so hateful uwu")
        else:
            st.warning('Please enter a text for analysis.')

if __name__ == '__main__':
    main()
