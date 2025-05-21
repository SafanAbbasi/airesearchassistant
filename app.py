import os
import streamlit as st
from ingest import load_and_embed, load_youtube_transcript
from qa_chain import ask_question

st.title("AI Research Assistant")

# Make sure the data directory exists
os.makedirs("data", exist_ok=True)

# Manually set the environment variable from secrets.toml
os.environ["OPENAI_API_KEY"] = st.secrets["openai"]["OPENAI_API_KEY"]

upload = st.file_uploader("Upload a PDF", type="pdf")
yt_link = st.text_input("Or paste a YouTube video link:")


question = st.text_input("Ask a question:")

if yt_link: 
    try:   
        vectordb = load_youtube_transcript(yt_link)
        st.session_state['doc_loaded'] = vectordb
        st.success("YouTube transcript loaded and indexed!")
    except Exception as e:
        st.error(f"Error loading YouTube transcript: {e}")

if upload:
    with open(f"data/{upload.name}", "wb") as f:
        f.write(upload.read())
    st.session_state['doc_loaded'] = load_and_embed(f"data/{upload.name}")

if question and st.session_state.get('doc_loaded'):
    answer = ask_question(question)
    st.markdown(answer)
