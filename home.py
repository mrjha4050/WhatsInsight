import streamlit as st
from sentiment import parse_chat, get_word_frequency, get_hourly_activity
from Whatsapp_analysis import analyze_chat
from groq import Groq
import os
from dotenv import load_dotenv
import plotly.express as px

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

# Initialize session state variables
if 'df' not in st.session_state:
    st.session_state['df'] = None
if 'dicp' not in st.session_state:
    st.session_state['dicp'] = None
if 'analysis' not in st.session_state:
    st.session_state['analysis'] = None
if 'chat_loaded' not in st.session_state:
    st.session_state['chat_loaded'] = False

def main():
    st.title("WhatsApp Analyzer 🕵️‍♂️")
    st.write("Upload your chat file below 📁:")

    uploaded_file = st.file_uploader("Choose a WhatsApp chat file (.txt)", type="txt")

    if uploaded_file:
        content = uploaded_file.read().decode('utf-8', errors='replace')
        with open("temp_chat.txt", "w", encoding="utf-8") as f:
            f.write(content)

        df, dicp, words = parse_chat("temp_chat.txt")
        analysis = analyze_chat("temp_chat.txt")

        st.session_state['df'] = df
        st.session_state['dicp'] = dicp
        st.session_state['analysis'] = analysis
        st.session_state['chat_loaded'] = True

        st.success("File uploaded successfully!")

    if st.session_state['chat_loaded']:
        st.write(f"**Total Messages:** {st.session_state['analysis']['total_messages']}")
        st.write("**User Word Counts:**", st.session_state['analysis']['user_word_counts'])

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Analysis 📊"):
                st.switch_page("pages/analysis.py")
        with col2:
            if st.button("Quick Analytics 📈"):
                st.session_state['show_analytics'] = True

        # Analytics Section
        if 'show_analytics' in st.session_state and st.session_state['show_analytics']:
            st.subheader("Quick Analytics 📉")
            
            st.write("**Most Used Words**")
            word_freq = get_word_frequency(st.session_state['dicp'])
            st.write(list(word_freq.items())[:10])

            st.write("**Hourly Activity**")
            hours = get_hourly_activity(st.session_state['df']["hour"].to_dict())
            fig = px.bar(
                x=list(hours.keys()),
                y=list(hours.values()),
                labels={"x": "Hour", "y": "Messages"},
                width=600,  # Smaller width for the main page
                height=300  # Smaller height for the main page
            )
            st.plotly_chart(fig)

        st.subheader("Suggestions 💁")
        if client and st.session_state['df'] is not None:
            chat_text = " ".join(st.session_state['df']["text"].dropna())
            try:
                if len(chat_text) > 2000:
                    st.warning("Chat text truncated to 2000 characters for suggestions.")
                suggestion_response = client.chat.completions.create(
                    model="llama3-70b-8192",
                    messages=[
                        {"role": "system", "content": "Provide suggestions based on this chat conversation and also at end show in short keywords."},
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
            st.write("Suggestions unavailable: Groq API key missing or chat not loaded.")

if __name__ == "__main__":
    main()