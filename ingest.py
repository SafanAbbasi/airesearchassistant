from langchain_community.document_loaders import PyMuPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_embed(file_path):
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    vectordb = Chroma.from_documents(chunks, OpenAIEmbeddings(), persist_directory="./chroma_db")
    vectordb.persist()
    return vectordb
# This function loads a PDF file, splits it into chunks, and stores the chunks in a Chroma vector store.
# It uses the OpenAI embeddings to create vector representations of the chunks.