# pages/profiles.py
import streamlit as st
from sentiment import get_user_threads
from groq import Groq
import os
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY)

st.title("User Profiles")
 

if 'df' not in st.session_state:
    st.warning("Please upload a file on the Home page.")
else:
    df = st.session_state['df']
    threads = get_user_threads(df['name'].to_dict())
    
    st.subheader("Save User Analysis")
    user_name = st.selectbox("Select user to analyze:", list(threads.keys()))
    if user_name:
        msg_count = threads[user_name]
        
        # Get the user's messages
        user_messages = " ".join(df[df['name'] == user_name]['text'].dropna())
        
        # Sentiment Analysis using Groq
        try:
            sentiment_response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a sentiment analysis expert. Analyze the sentiment of the given text and provide a score between -1 (negative) and 1 (positive)."},
                    {"role": "user", "content": f"Analyze the sentiment of this text: {user_messages[:2000]}"},
                ],
                max_tokens=100
            )
            sentiment_text = sentiment_response.choices[0].message.content
            # Extract the numerical score (assuming the response contains a number)
            sentiment_score = float(re.search(r'-?\d+\.?\d*', sentiment_text).group()) if re.search(r'-?\d+\.?\d*', sentiment_text) else 0
        except Exception as e:
            st.error(f"Error with Groq sentiment analysis: {e}")
            sentiment_score = 0
        
        # Personalized Suggestions using Groq
        try:
            suggestions_response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a communication expert. Provide personalized communication suggestions for the user based on their messages."},
                    {"role": "user", "content": f"User messages: {user_messages[:2000]}\nSentiment score: {sentiment_score}\nProvide 2-3 suggestions."},
                ],
                max_tokens=200
            )
            suggestions = suggestions_response.choices[0].message.content.split('\n')
            suggestions = [s.strip() for s in suggestions if s.strip()]
        except Exception as e:
            st.error(f"Error generating suggestions with Groq: {e}")
            suggestions = ["Try more positive language"]
        
        if st.button("Save Profile"):
            store_analysis(user_name, msg_count, sentiment_score, suggestions)
            st.success(f"Saved profile for {user_name}")
    
    st.subheader("Load User Profile")
    load_name = st.selectbox("Select user to load:", list(threads.keys()), key="load")
    if st.button("Load Profile"):
        profile = load_analysis(load_name)
        if profile:
            st.write(f"Name: {profile[0]}")
            st.write(f"Message Count: {profile[1]}")
            st.write(f"Sentiment: {profile[2]}")
            st.write(f"Suggestions: {profile[3]}")