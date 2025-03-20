# pages/suggestions.py
from transformers import pipeline
from sentiment import simple_sentiment
import streamlit as st
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY)

st.title("AI Suggestions")

if 'df' not in st.session_state:
    st.warning("Please upload a file on the Home page.")
else:
    df = st.session_state['df']
    dicp = st.session_state['dicp']
    chat_text = " ".join(df['text'].dropna())

    if not chat_text:
        st.error("No valid chat text found for analysis.")
        st.stop()

    st.subheader("Sentiment Analysis")

    col1, col2, col3 = st.columns(3)

    # 🔹 Groq Sentiment Analysis
    groq_sentiment = "Unknown"
    try:
        sentiment_response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Analyze sentiment as 'Positive', 'Negative', or 'Neutral'."},
                {"role": "user", "content": f"Analyze sentiment: {chat_text[:2000]}"},
            ],
            max_tokens=100
        )
        groq_sentiment = sentiment_response.choices[0].message.content
        sentiment_label = "Positive" if "positive" in groq_sentiment.lower() else "Negative" if "negative" in groq_sentiment.lower() else "Neutral"
    except Exception as e:
        st.error(f"Groq Sentiment Error: {e}")
        sentiment_label = "Error"

    col1.metric("Groq Sentiment", sentiment_label)

    # 🔹 Transformers Sentiment Analysis
    try:
        sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        transformers_sentiment = sentiment_analyzer(chat_text[:512])

        if transformers_sentiment:
            avg_score = sum(1 if s['label'] == 'POSITIVE' else -1 for s in transformers_sentiment) / len(transformers_sentiment)
            trans_label = 'Positive' if avg_score >= 0 else 'Negative'
        else:
            avg_score = 0
            trans_label = "Neutral"

    except Exception as e:
        st.error(f"Transformers Sentiment Error: {e}")
        avg_score = 0
        trans_label = "Error"

    col2.metric("Transformers Sentiment", f"{avg_score:.2f}", trans_label)

    # 🔹 Simple Sentiment Analysis
    try:
        positive_words = {'happy': 1, 'good': 1, 'great': 1, 'love': 1, 'awesome': 1}
        negative_words = {'sad': 1, 'bad': 1, 'terrible': 1, 'hate': 1, 'awful': 1}

        simple_sent = simple_sentiment(chat_text, positive_words, negative_words)
        simple_label = 'Positive' if simple_sent >= 0 else 'Negative'

    except Exception as e:
        st.error(f"Simple Sentiment Error: {e}")
        simple_sent = 0
        simple_label = "Error"

    col3.metric("Simple Sentiment", f"{simple_sent:.2f}", simple_label)

    st.write("---")  # Adds a divider for clarity

    # 🔹 Communication Suggestions
    st.subheader("Communication Suggestions")
    try:
        suggestions_response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Provide 2-3 actionable communication suggestions based on sentiment."},
                {"role": "user", "content": f"Chat text: {chat_text[:2000]}\nSentiment: {groq_sentiment}"},
            ],
            max_tokens=300
        )
        groq_suggestions = suggestions_response.choices[0].message.content.split('\n')
        for suggestion in groq_suggestions:
            if suggestion.strip():
                st.write(f"- {suggestion.strip()}")
    except Exception as e:
        st.error(f"Error with Groq suggestions: {e}")

        # Fallback suggestions if AI fails
        fallback_suggestions = []
        if avg_score < 0:
            fallback_suggestions.append("Use more positive language to improve tone.")
        if "sorry" in dicp:
            fallback_suggestions.append("Minimize unnecessary apologies to convey confidence.")
        for s in fallback_suggestions:
            st.write(f"- {s}")