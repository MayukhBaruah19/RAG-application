import os
import streamlit as st
from src.data.data_loader import load_pdf,split_documents
from src.models.llm import get_llm,LLM_MODEL_LIST
from src.models.embedding import get_embeddings
from src.QAchain.retrieval_qa import create_qa_chain
from langchain_community.vectorstores import Chroma
import logging
# from langchain.prompts.chat import ChatPromptTemplate,SystemMessagePromptTemplate,HumanMessagePromptTemplate
# from langchain.chains import RetrievalQA


DATA_FOLDER = "data"          
CHROMA_INDEX_DIR = "chroma_index"  # VectorDB folder
TOP_K = 3 

os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(CHROMA_INDEX_DIR, exist_ok=True)
# Model selection option
selected_model = st.sidebar.selectbox("Select Model", LLM_MODEL_LIST)

@st.cache_resource(show_spinner=True)
def get_vector_db():

    embedding=get_embeddings()

    # Check if database exists
    vector_db=None
    db_exists=os.path.exists(os.path.join(CHROMA_INDEX_DIR,'chroma.sqlite'))

    if db_exists:
        vector_db=Chroma(
            persist_directory=CHROMA_INDEX_DIR,
            embedding_function=embedding 
        )

    #process PDF file 
    pdf_files= [f for f in os.listdir(DATA_FOLDER) if f.lower().endswith(".pdf")]
    if not pdf_files:
        st.write("No PDF files found in the data folder")
        return vector_db

    for pdf_file in pdf_files:
        pdf_path=os.path.join(DATA_FOLDER,pdf_file)
        already_added=False

        if vector_db:
            existing_metadeta=vector_db.get(include=["metadatas"])["metadatas"]
            for meta in existing_metadeta:
                if meta.get("source")==pdf_path:
                    already_added=True
                    break
        if already_added:
            continue

        # Load pdf and split
        docs=load_pdf(pdf_path)
        chunks=split_documents(docs)

        for chunk in chunks:
            chunk.metadata["source"]=pdf_path

        if vector_db:
            vector_db.add_documents(chunks)    
        else:
            vector_db=Chroma.from_documents(
                chunks,
                embedding=embedding,
                persist_directory=CHROMA_INDEX_DIR
            )
            
    # persist db
    if vector_db:
        vector_db.persist()
        print("chroma db updated successfully")
    return vector_db

# QA chain    
@st.cache_resource(show_spinner=False)
def get_qa_chain():
    llm=get_llm(selected_model)
    vector_db=get_vector_db()
    if vector_db is None:
        return None
    
    retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": TOP_K})


    qa_chain=create_qa_chain(llm, retriever, top_k=TOP_K)

    return qa_chain

st.set_page_config(page_title="PDF Question Answering", page_icon="📄")
st.title(" PDF Question Answering Bot")

user_question = st.text_input("Enter your question here:")

if user_question:
    with st.spinner("Generating answer..."):
        qa_chain = get_qa_chain()
        response=qa_chain.invoke({"query":user_question})
        st.header("Answer")
        st.write(response['result'])

    






