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

# Initialize Groq client safely
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY is not set. Please add it to your .env file.")
    client = None
else:
    try:
        client = Groq(api_key=GROQ_API_KEY)
    except Exception as e:
        st.error(f"Failed to initialize Groq client: {e}")
        client = None

st.title("Chat Analysis 📉📊")

# Check session state
if "df" not in st.session_state or "dicp" not in st.session_state:
    st.warning("Please upload a file on the Home page.")
    st.stop()

df = st.session_state["df"]
dicp = st.session_state["dicp"]
chat_text = " ".join(df["text"].dropna())

st.subheader("Most Used Words")
word_freq = get_word_frequency(dicp)
st.write(list(word_freq.items())[:10])

st.subheader("Word Cloud")
wordcloud = WordCloud(width=800, height=400).generate_from_frequencies(word_freq)
fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(wordcloud, interpolation="bilinear")
ax.axis("off")
st.pyplot(fig)
plt.close(fig)

st.subheader("Hourly Activity")
hours = get_hourly_activity(df["hour"].to_dict())
fig = px.bar(
    x=list(hours.keys()),
    y=list(hours.values()),
    labels={"x": "Hour", "y": "Messages"},
)
st.plotly_chart(fig)

st.subheader("Main Topics Discussed")
if client:
    try:
        topics_response = client.chat.completions.create(
            model="llama3-70b-8192",  # Update if needed
            messages=[
                {
                    "role": "system",
                    "content": "You are a topic modeling expert. Identify the main topics discussed in the given chat conversation.",
                },
                {
                    "role": "user",
                    "content": f"Identify the main topics in this chat: {chat_text[:2000]}",
                },
            ],
            max_tokens=200,
        )
        topics = topics_response.choices[0].message.content
        st.write(topics)
    except Exception as e:
        st.error(f"Error identifying topics with Groq: {e}")
        st.write("Unable to identify topics at this time.")
else:
    st.write("Topic analysis unavailable: Groq API key missing.")

st.subheader("Overall Tone of the Conversation")
if client:
    try:
        tone_response = client.chat.completions.create(
            model="llama3-70b-8192",  # Update if needed
            messages=[
                {
                    "role": "system",
                    "content": "You are a tone analysis expert. Analyze the overall tone of the given chat conversation (e.g., formal, informal, emotional, professional).",
                },
                {
                    "role": "user",
                    "content": f"Analyze the tone of this chat and provide a summary in simple words using emoji: {chat_text[:2000]}",
                },
            ],
            max_tokens=100,
        )
        tone = tone_response.choices[0].message.content
        st.write(tone)
    except Exception as e:
        st.error(f"Error analyzing tone with Groq: {e}")
        st.write("Unable to analyze tone at this time.")
else:
    st.write("Tone analysis unavailable: Groq API key missing.")