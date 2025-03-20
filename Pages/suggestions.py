# pages/suggestions.py
import streamlit as st
from groq import Groq
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Groq client safely
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY is not set. Please add it to your environment.")
    client = None
else:
    try:
        client = Groq(api_key=GROQ_API_KEY)
    except Exception as e:
        st.error(f"Failed to initialize Groq client: {e}")
        client = None

st.title("Suggestions 💡")

# Check session state
if "df" not in st.session_state:
    st.warning("Please upload a file on the Home page.")
    st.stop()

df = st.session_state["df"]
chat_text = " ".join(df["text"].dropna())

# Get suggestions using Groq
if client:
    try:
        suggestion_response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": "Provide suggestions based on this chat conversation."},
                {"role": "user", "content": f"Chat: {chat_text[:2000]}"},
            ],
            max_tokens=200,
        )
        suggestions = suggestion_response.choices[0].message.content
        st.write(suggestions)
    except Exception as e:
        st.error(f"Error getting suggestions: {e}")
        st.write("Unable to get suggestions at this time.")
else:
    st.write("Suggestions unavailable: Groq API key missing.")