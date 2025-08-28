import os
import streamlit as st
from src.helper import load_pdf, split_documents
from src.models import get_llm, get_embeddings, LLM_MODEL_LIST
from langchain.vectorstores import Chroma
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import RetrievalQA


DATA_FOLDER = "data"          #data source
CHROMA_INDEX_DIR = "chroma_index"  # VectorDB folder
TOP_K = 3               

os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(CHROMA_INDEX_DIR, exist_ok=True)

# Model selection option
selected_model = st.sidebar.selectbox("Select Model", LLM_MODEL_LIST)


@st.cache_resource(show_spinner=True)
def get_vector_db():
    embeddings = get_embeddings()
    
    # Load existing DB if exists
    vector_db = None
    db_exists = os.path.exists(os.path.join(CHROMA_INDEX_DIR, "chroma.sqlite"))
    if db_exists:
        vector_db = Chroma(persist_directory=CHROMA_INDEX_DIR, embedding_function=embeddings)
        print("Loaded existing Chroma vector store.")

    # Process PDFs in data folder
    pdf_files = [f for f in os.listdir(DATA_FOLDER) if f.lower().endswith(".pdf")]
    if not pdf_files:
        st.warning(f"No PDFs found in '{DATA_FOLDER}' folder. Add PDFs to use this app.")
        return vector_db 
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(DATA_FOLDER, pdf_file)
        already_added = False

        if vector_db:
            # Check if PDF is already added using metadata
            existing_metadatas = vector_db.get(include=["metadatas"])["metadatas"]
            for meta in existing_metadatas:
                if meta.get("source") == pdf_path:
                    already_added = True
                    break
        if already_added:
            continue

        # Load PDF and split into chunks
        docs = load_pdf(pdf_path)
        chunks = split_documents(docs)
        # Add metadata for source reference
        for chunk in chunks:
            chunk.metadata["source"] = pdf_path

        if vector_db:
            vector_db.add_documents(chunks)
        else:
            vector_db = Chroma.from_documents(
                chunks, embedding=embeddings, persist_directory=CHROMA_INDEX_DIR
            )

    # Persist DB
    if vector_db:
        vector_db.persist()
        print("Chroma vector store updated successfully.")

    return vector_db

# QA chain
@st.cache_resource(show_spinner=True)
def get_qa_chain():
    llm = get_llm(selected_model)
    vector_db = get_vector_db()
    if vector_db is None:
        return None

    retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": TOP_K})

    chat_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are a helpful assistant. Answer the user's question based on the provided context."
        ),
        HumanMessagePromptTemplate.from_template("{question}\n\nContext: {context}")
    ])

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": chat_prompt}
    )

    return qa_chain


# Streamlit App
st.set_page_config(page_title="PDF Question Answering", page_icon="📄")
st.title(" PDF Question Answering Bot")

user_question = st.text_input("Enter your question here:")

if user_question:
    qa_chain = get_qa_chain()
    if qa_chain is not None:
        with st.spinner("Generating answer..."):
            answer = qa_chain.run({"query": user_question})
        st.subheader("Answer")
        st.write(answer)
    else:
        st.warning("Vector database not loaded. Make sure the 'data/' folder contains PDFs.")
