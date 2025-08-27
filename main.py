from collections import abc
import streamlit as st
from src.Vector_database import update_faiss_db
from src.models import get_llm
from langchain.chains import RetrievalQA


st.set_page_config(page_title="PDF RAG App", layout="wide")
st.title("📄 PDF RAG QA System")

# Step 1: Update / Load FAISS DB
with st.spinner("Loading FAISS vector store..."):
    vector_db = update_faiss_db(data_folder="data")

# Step 2: Create retriever
retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Step 3: Initialize LLM
llm = get_llm("llama2:latest")

# Step 4: Setup RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=False
)

# Step 5: User query input
query = st.text_input("Ask a question about your PDFs:")

if query:
    with st.spinner("Searching and generating answer..."):
        result = qa_chain({"query": query})

    # Display answer
    st.subheader("Answer")
    st.write(result["result"])

    
