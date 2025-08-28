
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from src.models import get_embeddings

# Load PDF
def load_pdf(pdf_path: str) -> List[Document]:
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages from {pdf_path}")
    return documents    

#Split documents into chunks
def split_documents(documents: List[Document], chunk_size=500, chunk_overlap=50) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splitted_docs = splitter.split_documents(documents)

    # Keep track of original page numbers
    for i, doc in enumerate(splitted_docs):
        doc.metadata["chunk_id"] = i
    return splitted_docs    



#Load embeddings
def load_embeddings():
    embeddings = get_embeddings()
    print("Embedding Generated Successfully")
    return embeddings






