import streamlit as st
from sentiment import parse_chat
from Whatsapp_analysis import analyze_chat

st.title("WhatsApp Analyzer")
st.write("Upload your chat file below:")

st.sidebar.title("Navigation")
st.sidebar.write("Select a page to navigate:")


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

    st.success("File uploaded successfully!")
    st.write(f"**Total Messages:** {analysis['total_messages']}")
    st.write("**User Word Counts:**", analysis['user_word_counts'])

    # Navigation Buttons
    st.write("---")
    st.subheader("Explore Detailed Analysis:")
