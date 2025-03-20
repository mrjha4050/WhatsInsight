import streamlit as st
from sentiment import parse_chat
from Whatsapp_analysis import analyze_chat

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
            if st.button("Suggestion 💁"):
                st.switch_page("pages/suggestions.py")

if __name__ == "__main__":
    main()