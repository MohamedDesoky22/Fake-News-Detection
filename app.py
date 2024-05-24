import streamlit as st
import string
import joblib
import os

# Load the model
model_file_path = 'best_model.pkl'
model = joblib.load(model_file_path)

def preprocess(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Return preprocessed text
    return text

# Define the function to make predictions
def predict_fake_news(news_text):
    if model is None:
        return None
    
    # Preprocess the input text
    preprocessed_text = preprocess(news_text)
    # Make prediction
    prediction = model.predict([preprocessed_text])[0]
    return prediction

# Define the Streamlit app
def main():
    # Set title and description
    st.title("Fake News Prediction")
    st.write("Enter the news text and click the button to check if it's fake or real.")

    # Create a text input for the user to enter the news text
    news_text = st.text_area("Enter the news text here:", "")

    # When the user clicks the predict button
    if st.button("Predict"):
        # Check if the input is not empty
        if news_text:
            # Make prediction
            prediction = predict_fake_news(news_text)
            if prediction is not None:
                # Display the prediction
                if prediction == 1:
                    st.error("Fake News Detected!")
                else:
                    st.success("Real News Detected!")
            else:
                st.warning("Model not loaded.")
        else:
            st.warning("Please enter some text.")
main()