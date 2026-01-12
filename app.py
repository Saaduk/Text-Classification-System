import streamlit as st
import pickle
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import os

# 1️⃣ Download stopwords (if not already)
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

# 2️⃣ Load trained model and vectorizer
if not os.path.exists('model/model.pkl') or not os.path.exists('model/vectorizer.pkl'):
    st.error("Model or vectorizer not found! Run Text_classifier.py first to generate them.")
else:
    model = pickle.load(open('model/model.pkl', 'rb'))
    vectorizer = pickle.load(open('model/vectorizer.pkl', 'rb'))

    # 3️⃣ Streamlit UI
    st.title("📨 SMS Spam Detector")
    st.write("Enter your message below to check if it is Spam or Ham.")

    # Input
    user_input = st.text_area("Type your message here:")

    # Predict button
    if st.button("Predict"):
        if user_input.strip() == "":
            st.warning("Please enter a message.")
        else:
            # Preprocessing function (same as training)
            def preprocess(text):
                text = re.sub('[^a-zA-Z]', ' ', text)
                text = text.lower()
                words = text.split()
                words = [ps.stem(word) for word in words if word not in stop_words]
                return " ".join(words)

            clean_text = preprocess(user_input)
            vect_text = vectorizer.transform([clean_text])
            prediction = model.predict(vect_text)[0]
            label = "Spam" if prediction == 1 else "Ham"
            st.success(f"Prediction: **{label}**")
