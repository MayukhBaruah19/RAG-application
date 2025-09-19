from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

def load_pdf(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    return loader.load()

def split_documents(docs, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)
    for i, doc in enumerate(chunks):
        doc.metadata["chunk_id"] = i
    return chunks
