import os
from langchain.vectorstores import Chroma
from src.models.embedding import get_embeddings

def load_vector_db(persist_directory="chroma_index"):
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)
        return Chroma(persist_directory=persist_directory, embedding_function=get_embeddings())
    return Chroma(persist_directory=persist_directory, embedding_function=get_embeddings())
