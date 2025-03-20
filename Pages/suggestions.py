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
    
    st.subheader("Sentiment Analysis")
    try:
        sentiment_response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # Use a versatile Groq model
            messages=[
                {"role": "system", "content": "You are a sentiment analysis expert. Analyze the sentiment of the given text and classify it as 'Positive', 'Negative', or 'Neutral'. Provide a brief explanation."},
                {"role": "user", "content": f"Analyze the sentiment of this text: {chat_text[:2000]}"},  # Truncate to avoid token limits
            ],
            max_tokens=200
        )
        groq_sentiment = sentiment_response.choices[0].message.content
        st.write(f"Groq Sentiment Analysis: {groq_sentiment}")
    except Exception as e:
        st.error(f"Error with Groq sentiment analysis: {e}")
        groq_sentiment = "Error in sentiment analysis"

    sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    sentiments = sentiment_analyzer(chat_text[:512])  # Truncate for model limit
    avg_score = sum(1 if s['label'] == 'POSITIVE' else -1 for s in sentiments) / len(sentiments)
    st.write(f"Transformers Sentiment Score: {avg_score:.2f} ({'Positive' if avg_score >= 0 else 'Negative'})")

    positive_words = {'happy': 1, 'good': 1, 'great': 1, 'love': 1, 'awesome': 1}
    negative_words = {'sad': 1, 'bad': 1, 'terrible': 1, 'hate': 1, 'awful': 1}
    simple_sent = simple_sentiment(chat_text, positive_words, negative_words)
    st.write(f"Simple Sentiment: {simple_sent}")

    st.subheader("Communication Suggestions")
    try:
        suggestions_response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a communication expert. Based on the sentiment and content of the given chat text, provide specific, actionable suggestions to improve communication. Consider the tone, language, and context."},
                {"role": "user", "content": f"Chat text: {chat_text[:2000]}\nSentiment: {groq_sentiment}\nProvide 2-3 communication suggestions."},
            ],
            max_tokens=300
        )
        groq_suggestions = suggestions_response.choices[0].message.content.split('\n')
        for suggestion in groq_suggestions:
            if suggestion.strip():
                st.write(f"- {suggestion.strip()}")
    except Exception as e:
        st.error(f"Error generating suggestions with Groq: {e}")
        # Fallback to basic suggestions
        suggestions = []
        if avg_score < 0:
            suggestions.append("Try using more positive language to improve group morale.")
        if "sorry" in dicp:
            suggestions.append("Reduce apologies unless necessary—confidence can enhance your tone.")
        for s in suggestions:
            st.write(f"- {s}")