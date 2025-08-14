import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from models import get_embeddings

load_dotenv()
# Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")


def process_pdf_and_store(pdf_path: str, selected_class: str, selected_subject: str, persist_directory="chroma_db"):
    """
    Load a PDF, split into chunks, add class_subject metadata,
    and store in Chroma vector DB.
    """
    try:
        print(f"Loading PDF: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        print(f"Splitting {len(documents)} pages into chunks...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100)
        splitted_docs = splitter.split_documents(documents)

        # Add class_subject metadata
        for doc in splitted_docs:
            doc.metadata["class_subject"] = f"{selected_class}|{selected_subject}"

        print("Loading embedding model...")
        embeddings = get_embeddings()

        if os.path.exists(persist_directory):
            print("Adding to existing vector database...")
            vector_db = Chroma(
                persist_directory=persist_directory, embedding_function=embeddings)
            vector_db.add_documents(splitted_docs)
        else:
            print("Creating new vector database...")
            vector_db = Chroma.from_documents(
                splitted_docs, embeddings, persist_directory=persist_directory)

        vector_db.persist()
        print("Vector database updated successfully!")
        return vector_db

    except Exception as e:
        print(f"Error in process_pdf_and_store: {e}")
        raise e


def load_vector_db(persist_directory="chroma_db"):
    """Load an existing Chroma vector DB."""
    try:
        if not os.path.exists(persist_directory):
            raise FileNotFoundError(
                f"Vector DB directory not found: {persist_directory}")

        print(f"Loading vector database from: {persist_directory}")
        embeddings = get_embeddings()
        vector_db = Chroma(persist_directory=persist_directory,
                           embedding_function=embeddings)
        print("Vector database loaded successfully!")
        return vector_db

    except Exception as e:
        print(f"Error loading vector database: {e}")
        raise e
