import os
import pickle
from bs4 import BeautifulSoup
import requests
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


# Save the vector store to disk
def save_vector_store(vector_store, path):
    with open(path, 'wb') as f:
        pickle.dump(vector_store, f)

# Load and process website content from sitemap
def load_and_process_website_content_from_sitemap(vector_store_path):
    urls =["http://www.ept.rnu.tn/","http://www.ept.rnu.tn/formation/cycle-dingenieur/voyage-detudes/"]
    
    all_docs = []

    for url in urls:
        try:
            loader = WebBaseLoader(url)
            documents = loader.load()
            print(f"Loaded {len(documents)} documents from {url}")

            # Split the content into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = text_splitter.split_documents(documents)
            all_docs.extend(docs)
            print(f"Split into {len(docs)} chunks")
        except Exception as e:
            print(f"Failed to load or process {url}: {e}")

    if not all_docs:
        print("No documents were loaded and processed.")
        return

    # Create embeddings for the chunks
    from langchain.embeddings import SentenceTransformerEmbeddings
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
   
    from langchain.vectorstores import Chroma
    db = Chroma.from_documents(docs, embeddings)

    if not embeddings:
        print("No embeddings were created.")
        return

    # Initialize the FAISS vector store
    vector_store = FAISS.from_documents(all_docs, embeddings)
    save_vector_store(vector_store, vector_store_path)
    print(f"Vector store saved to {vector_store_path}")

# Parameters
vector_store_path = 'vector_store.pkl'

# Generate and save the vector store
load_and_process_website_content_from_sitemap(vector_store_path)
