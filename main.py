import os
import streamlit as st
import pickle
import time
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import FAISS
from dotenv import load_dotenv
from langchain_community.llms import Ollama
from langchain.embeddings import OllamaEmbeddings  # Import Ollama embeddings and LLaMA LLM

load_dotenv()  # Load environment variables from .env

st.title("RockyBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

# Collect URLs from the user
urls = [url for i in range(3) if (url := st.sidebar.text_input(f"URL {i+1}"))]

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_ollama.pkl"

main_placeholder = st.empty()
llm = Ollama(model="llama3.1", temperature=0.9, max_tokens=500)  # Using LLaMA 3.1 model from Ollama

if process_url_clicked and urls:
    # Load data from the URLs
    try:
        loader = UnstructuredURLLoader(urls=urls)
        main_placeholder.text("Data Loading...Started...✅✅✅")
        data = loader.load()
    except Exception as e:
        main_placeholder.text(f"Error loading data from URLs: {e}")
        st.stop()

    # Process data if successfully loaded
    if data:
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=1000
        )
        main_placeholder.text("Text Splitter...Started...✅✅✅")
        docs = text_splitter.split_documents(data)

        try:
            # Generate embeddings using Ollama embeddings
            embeddings = OllamaEmbeddings()
            vectorstore_ollama = FAISS.from_documents(docs, embeddings)
            main_placeholder.text("Embedding Vector Started Building...✅✅✅")
            time.sleep(2)
        except Exception as e:
            main_placeholder.text(f"Error generating embeddings: {e}")
            st.stop()

        # Save the FAISS index to a pickle file
        with open(file_path, "wb") as f:
            pickle.dump(vectorstore_ollama, f)

# Handle queries from the user
query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            # Use the LLaMA 3.1 model via Ollama to process the query
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)

            # Display the answer and sources
            st.header("Answer")
            st.write(result["answer"])

            # Display sources, if available
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")  # Split the sources by newline
                for source in sources_list:
                    st.write(source)
