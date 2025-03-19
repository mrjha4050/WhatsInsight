# pages/home.py
import streamlit as st
from sentiment import parse_chat
from Whatsapp_analysis import analyze_chat

st.title("WhatsApp Analyzer")
st.write("Upload your chat file below:")
uploaded_file = st.file_uploader("Choose a WhatsApp chat file (.txt)", type="txt")

if uploaded_file:
    # Save the file temporarily
    content = uploaded_file.read().decode('utf-8', errors='replace')
    with open("temp_chat.txt", "w", encoding="utf-8") as f:
        f.write(content)
    df, dicp, words = parse_chat("temp_chat.txt")
    analysis = analyze_chat("temp_chat.txt")
    
    # Store in session state
    st.session_state['df'] = df
    st.session_state['dicp'] = dicp
    st.session_state['analysis'] = analysis
    
    st.success("File uploaded! Navigate to other pages for detailed analysis.")
    st.write(f"Total Messages: {analysis['total_messages']}")
    st.write("User Word Counts:", analysis['user_word_counts'])