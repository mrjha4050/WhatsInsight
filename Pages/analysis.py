# pages/analysis.py
import streamlit as st
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sentiment import get_word_frequency, get_hourly_activity
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

st.title("Chat Analysis")

if 'df' not in st.session_state:
    st.warning("Please upload a file on the Home page.")
else:
    df = st.session_state['df']
    dicp = st.session_state['dicp']
    
    chat_text = " ".join(df['text'].dropna())
    
    st.subheader("Most Used Words")
    word_freq = get_word_frequency(dicp)
    st.write(list(word_freq.items())[:10])
    
    st.subheader("Word Cloud")
    wordcloud = WordCloud(width=800, height=400).generate_from_frequencies(word_freq)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)
    
    st.subheader("Hourly Activity")
    hours = get_hourly_activity(df['hour'].to_dict())
    fig = px.bar(x=list(hours.keys()), y=list(hours.values()), labels={'x': 'Hour', 'y': 'Messages'})
    st.plotly_chart(fig)
    
    st.subheader("Main Topics Discussed")
    try:
        topics_response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a topic modeling expert. Identify the main topics discussed in the given chat conversation."},
                {"role": "user", "content": f"Identify the main topics in this chat: {chat_text[:2000]}"},
            ],
            max_tokens=200
        )
        topics = topics_response.choices[0].message.content
        st.write(topics)
    except Exception as e:
        st.error(f"Error identifying topics with Groq: {e}")
        st.write("Unable to identify topics at this time.")
    
    st.subheader("Overall Tone of the Conversation")
    try:
        tone_response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a tone analysis expert. Analyze the overall tone of the given chat conversation (e.g., formal, informal, emotional, professional)."},
                {"role": "user", "content": f"Analyze the tone of this chat: {chat_text[:2000]}"},
            ],
            max_tokens=100
        )
        tone = tone_response.choices[0].message.content
        st.write(tone)
    except Exception as e:
        st.error(f"Error analyzing tone with Groq: {e}")
        st.write("Unable to analyze tone at this time.")