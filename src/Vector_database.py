import os
import hashlib
from typing import List
from collections import abc
from langchain.schema import Document
from langchain.vectorstores import FAISS
from src.helper import load_pdf, split_documents, load_embeddings

FAISS_INDEX_DIR = "faiss_index"
HASHES_FILE = os.path.join(FAISS_INDEX_DIR, "pdf_hashes.txt")


def compute_file_hash(file_path: str) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


def load_existing_hashes() -> set:
    """Load hashes of already processed PDFs."""
    if not os.path.exists(HASHES_FILE):
        return set()
    with open(HASHES_FILE, "r") as f:
        return set(line.strip() for line in f.readlines())


def save_hashes(hashes: set):
    """Save updated hashes."""
    os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
    with open(HASHES_FILE, "w") as f:
        f.write("\n".join(hashes))


def update_faiss_db(data_folder: str = "data") -> FAISS:
    """Update FAISS vector database with new PDFs from data folder."""
    os.makedirs(FAISS_INDEX_DIR, exist_ok=True)

    # Load existing FAISS DB if exists
    if os.path.exists(os.path.join(FAISS_INDEX_DIR, "index.faiss")):
        vector_db = FAISS.load_local(FAISS_INDEX_DIR, 
                                     embeddings=load_embeddings(),
                                     allow_dangerous_deserialization=True)
        print("Loaded existing FAISS vector store.")
    else:
        vector_db = None
        print("No existing FAISS DB found. Creating a new one.")

    # Load existing PDF hashes
    existing_hashes = load_existing_hashes()

    # Track new hashes
    new_hashes = set()

    # Scan PDFs in data folder
    for file in os.listdir(data_folder):
        if not file.lower().endswith(".pdf"):
            continue
        file_path = os.path.join(data_folder, file)
        file_hash = compute_file_hash(file_path)

        if file_hash in existing_hashes:
            continue  # Already processed
        print(f"Processing new PDF: {file}")

        # Load and split
        docs = load_pdf(file_path)
        chunks = split_documents(docs)

        # Add to FAISS DB
        if vector_db:
            vector_db.add_documents(chunks)
        else:
            vector_db = FAISS.from_documents(chunks, load_embeddings())

        new_hashes.add(file_hash)

    # Update hash file
    all_hashes = existing_hashes.union(new_hashes)
    save_hashes(all_hashes)

    # Persist FAISS DB
    vector_db.save_local(FAISS_INDEX_DIR)
    print("FAISS vector store updated successfully.")
    return vector_db
