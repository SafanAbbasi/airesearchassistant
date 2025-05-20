from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings

def ask_question(query):
    vectordb = Chroma(persist_directory="./chroma_db", embedding_function=OpenAIEmbeddings())
    retriever = vectordb.as_retriever(search_type="similarity", k=3)
    llm = ChatOpenAI(temperature=0)
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa.run(query)
# This function takes a query, retrieves relevant chunks from the Chroma vector store, and uses a language model to generate an answer.
# It uses the OpenAI embeddings to create vector representations of the chunks and the ChatOpenAI model to generate the answer.