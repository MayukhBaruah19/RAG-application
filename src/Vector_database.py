from typing import List
import os
from langchain.vectorstores import Chroma
from src.models import get_embeddings

CHROMA_INDEX_DIR = "chroma_index"  # Path where Chroma DB is stored

def load_vector_db(persist_directory=CHROMA_INDEX_DIR):
    """Load an existing Chroma vector DB from a given directory."""
    try:
        if not os.path.exists(persist_directory):
            raise FileNotFoundError(f"Vector DB directory not found: {persist_directory}")

        print(f"Loading vector database from: {persist_directory}")
        embeddings = get_embeddings()
        vector_db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        print("Vector database loaded successfully!")
        return vector_db

    except Exception as e:
        print(f"Error loading vector database: {e}")
        raise e
    
    