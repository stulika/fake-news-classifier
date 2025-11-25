import streamlit as st
import pickle

# Load model & vectorizer
model = pickle.load(open("fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

st.set_page_config(page_title="Fake News Classifier", page_icon="📰")

st.title("📰 Fake News Classifier")
st.subheader("Detect whether a news article is REAL or FAKE using Machine Learning")

# Input box
news_text = st.text_area("Enter News Article Text here:")

if st.button("Classify"):
    if news_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Transform input
        tfidf = vectorizer.transform([news_text])
        prediction = model.predict(tfidf)[0]

        if prediction == 1:
            st.success("✅ This news is REAL")
        else:
            st.error("❌ This news is FAKE")

