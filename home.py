import streamlit as st
from sentiment import parse_chat, get_word_frequency, get_hourly_activity
from Whatsapp_analysis import analyze_chat
from groq import Groq
import os
from dotenv import load_dotenv
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

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
if 'show_analysis' not in st.session_state:
    st.session_state['show_analysis'] = False

# Function to summarize chat text if it's too long
def summarize_chat(chat_text, max_chars=8000):
    if len(chat_text) <= max_chars:
        return chat_text, False
    if client:
        try:
            summary_response = client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[
                    {"role": "system", "content": "You are a summarization expert. Summarize the given chat conversation into a concise summary."},
                    {"role": "user", "content": f"Summarize this chat into {max_chars} characters or less: {chat_text[:16000]}"},
                ],
                max_tokens=2000,  # Adjust based on max_chars
            )
            summary = summary_response.choices[0].message.content
            return summary, True
        except Exception as e:
            st.error(f"Error summarizing chat: {e}")
            return chat_text[:max_chars], True
    return chat_text[:max_chars], True

def main():
    st.title("WhatsApp Analyzer 🕵️‍♂️")
    st.write("For getting the chat data, go to person profile and click the export chat button.")

    st.write("Upload a WhatsApp chat file (.txt) to get started 📁:-")

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
        st.session_state['show_analysis'] = False  # Reset analysis display on new upload

        st.success("File uploaded successfully!")

    if st.session_state['chat_loaded']:
        st.write(f"**Total Messages:** {st.session_state['analysis']['total_messages']}")
        st.write("**User Word Counts:**", st.session_state['analysis']['user_word_counts'])

        # Button to toggle analysis section
        if st.button("Analysis 📊"):
            st.session_state['show_analysis'] = True

        # Full Analysis Section
        if st.session_state['show_analysis']:
            st.subheader("Chat Analysis 📉📊")
            chat_text = " ".join(st.session_state['df']["text"].dropna())
            processed_text, was_summarized = summarize_chat(chat_text, max_chars=8000)
            if was_summarized:
                st.warning("Chat text was summarized to fit within the API limit.")

            # Most Used Words
            st.subheader("Most Used Words")
            word_freq = get_word_frequency(st.session_state['dicp'])
            st.write(list(word_freq.items())[:10])

            # Word Cloud
            st.subheader("Word Cloud")
            wordcloud = WordCloud(width=800, height=400).generate_from_frequencies(word_freq)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)
            plt.close(fig)

            # Hourly Activity
            st.subheader("Hourly Activity")
            hours = get_hourly_activity(st.session_state['df']["hour"].to_dict())
            fig = px.bar(
                x=list(hours.keys()),
                y=list(hours.values()),
                labels={"x": "Hour", "y": "Messages"},
            )
            st.plotly_chart(fig)

            # Main Topics Discussed
            st.subheader("Main Topics Discussed")
            if client:
                try:
                    topics_response = client.chat.completions.create(
                        model="llama3-70b-8192",
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a topic modeling expert. Identify the main topics discussed in the given chat conversation.",
                            },
                            {
                                "role": "user",
                                "content": f"Identify the main topics in this chat: {processed_text}",
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

            # Overall Tone of the Conversation
            st.subheader("Overall Tone of the Conversation")
            if client:
                try:
                    tone_response = client.chat.completions.create(
                        model="llama3-70b-8192",
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a tone analysis expert. Analyze the overall tone of the given chat conversation (e.g., formal, informal, emotional, professional).",
                            },
                            {
                                "role": "user",
                                "content": f"Analyze the tone of this chat and provide a summary in simple words using emoji: {processed_text}",
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

        # Suggestions Section
        st.subheader("Suggestions 💁")
        if client and st.session_state['df'] is not None:
            chat_text = " ".join(st.session_state['df']["text"].dropna())
            processed_text, was_summarized = summarize_chat(chat_text, max_chars=8000)
            if was_summarized:
                st.warning("Chat text was summarized to fit within the API limit for suggestions.")
            try:
                suggestion_response = client.chat.completions.create(
                    model="llama3-70b-8192",
                    messages=[
                        {"role": "system", "content": "Provide suggestions based on this chat conversation and also at end show in short keywords."},
                        {"role": "user", "content": f"Chat: {processed_text}"},
                    ],
                    max_tokens=300,
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