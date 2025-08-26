from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from typing import List
from langchain.schema import Document


#Extract the documents from the PDF files
def load_pdf_file(data):
    loader= DirectoryLoader(data,
                            glob="*.pdf",
                            loader_cls=PyPDFLoader)

    documents=loader.load()

    return documents



def filter_important_texts(docs: List[Document]) -> List[Document]:
    imp_documents: List[Document] = []
    for doc in docs:  #  iterate over the argument, not the function
        src = doc.metadata.get("source")
        imp_documents.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )
    return imp_documents
