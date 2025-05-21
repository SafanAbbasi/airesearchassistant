from langchain_community.document_loaders import PyMuPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

from youtube_transcript_api import YouTubeTranscriptApi
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import re

# This function loads a PDF file, splits it into chunks, and stores the chunks in a Chroma vector store.
# It uses the OpenAI embeddings to create vector representations of the chunks.
def load_and_embed(file_path):
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    vectordb = Chroma.from_documents(chunks, OpenAIEmbeddings(), persist_directory="./chroma_db")
    vectordb.persist()
    return vectordb



def load_youtube_transcript(yt_url):
    # Extract video ID
    video_id = extract_youtube_id(yt_url)
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    text = " ".join([entry['text'] for entry in transcript])

    # Turn into a LangChain Document
    doc = Document(page_content=text, metadata={"source": yt_url})

    # Chunking
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents([doc])

    # Vector store
    vectordb = Chroma.from_documents(chunks, OpenAIEmbeddings(), persist_directory="./chroma_db")
    vectordb.persist()
    return vectordb

def extract_youtube_id(url):
    match = re.search(r"(?<=v=)[^&#]+|(?<=youtu.be/)[^&#]+", url)
    if match:
        return match.group(0)
    raise ValueError("Invalid YouTube URL")