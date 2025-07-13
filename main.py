import os
import streamlit as st
import pickle
import time

from dotenv import load_dotenv

from gitdb.fun import chunk_size
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings  # Only if you're using OpenAI for embeddings
from langchain.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
import langchain

load_dotenv()
st.title("News Research Tool")

st.sidebar.title("News Article URLs")
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL{i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process Url")
file_path = "faiss_store_openai.pkl"

main_placefolder = st.empty()
llm = ChatGroq(model="llama-3.3-70b-versatile" , api_key=os.getenv("GROQ_API_KEY"))


if process_url_clicked:

    # load data from url
    loader = UnstructuredURLLoader(urls=urls)

    main_placefolder.text("Data Loading ... Started...")
    data = loader.load()

    # split the data

    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n','\n','.','.'],
        chunk_size=1000
    )

    main_placefolder.text("Data Splitting ... Started...")

    docs = text_splitter.split_documents(data)

    # create embeddings and save into faiss index

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectorstore_openapi = FAISS.from_documents(docs,embeddings)

    main_placefolder.text("Embedding vector started Building....")
    time.sleep(2)


    # save the faiss index to a pickle file

    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_openapi, f)

query = main_placefolder.text_input("Question: ")

if query:
    if os.path.exists(file_path):
        with open(file_path,"rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())

            result = chain({"question":query},return_only_outputs=True)

            st.header("Answer")
            st.subheader(result["answer"])

            sources = result.get("sources","")
            if sources:
                st.subheader("Sources")
                source_list = sources.split("\n")
                for source in source_list:
                    st.write(source)







