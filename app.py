import streamlit as st
import os
from models import LLM_MODEL_LIST, get_llm
from data_loader import process_pdf_and_store, load_vector_db
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

PERSIST_DIR = "chroma_db"
os.makedirs(PERSIST_DIR, exist_ok=True)  # Ensure folder exists

# Sidebar selections
selected_model = st.sidebar.selectbox("Select Model", LLM_MODEL_LIST)
selected_class = st.sidebar.selectbox(
    "Select Class", ["Class 6", "Class 7", "Class 8", "Class 9", "Class 10"], index=3
)
selected_subject = st.sidebar.selectbox(
    "Select Subject", ["Science", "Math", "History", "Geography", "Civics", "English Grammar"], index=0
)

# PDF Upload
uploaded_pdf = st.sidebar.file_uploader("Upload a PDF", type="pdf")
vector_db = None

if uploaded_pdf:
    upload_path = os.path.join("uploaded_pdfs", uploaded_pdf.name)
    os.makedirs("uploaded_pdfs", exist_ok=True)
    with open(upload_path, "wb") as f:
        f.write(uploaded_pdf.read())

    st.sidebar.info(f"Processing {uploaded_pdf.name} and updating vector database...")
    vector_db = process_pdf_and_store(
        upload_path, selected_class, selected_subject, persist_directory=PERSIST_DIR
    )
    st.sidebar.success("PDF processed and stored successfully!")

# Load existing DB if possible
elif os.path.exists(PERSIST_DIR):
    try:
        vector_db = load_vector_db(PERSIST_DIR)
    except FileNotFoundError:
        st.warning("Vector DB exists but is empty. Please upload a PDF.")

# Title and input
st.title("NCERT-based PDF Chatbot")
question = st.text_input("Ask any Question:")

# Prompt Template
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are a helpful tutor for {selected_class} students in {selected_subject}.\n"
         "Only answer if the question is related to this class and subject.\n"
         "Use only the provided context from the vector database.\n"
         "If the question is unrelated, respond: 'I can only answer questions related to the selected class and subject.'"),
        ("user", "Context:\n{context}\n\nQuestion: {question}")
    ]
)

# Answer Generation
if st.button("Get Answer") and question:
    if vector_db is None:
        st.warning("No PDFs in the database yet. Please upload a PDF first.")
    else:
        try:
            class_subject = f"{selected_class}|{selected_subject}"

            # Search in vector DB with metadata filter
            results = vector_db.similarity_search(
                question,
                k=4,
                filter={"class_subject": {"$eq": class_subject}}
            )

            if not results:
                st.warning(
                    f"No relevant information found in the database for {selected_class} - {selected_subject}."
                )
            else:
                context_text = "\n".join([doc.page_content for doc in results])

                llm = get_llm(selected_model)
                parser = StrOutputParser()

                # Proper chain with variables
                chain = prompt_template | llm | parser

                final_answer = chain.invoke({
                    "selected_class": selected_class,
                    "selected_subject": selected_subject,
                    "context": context_text,
                    "question": question
                })

                st.success("Answer:")
                st.write(final_answer)

        except Exception as e:
            st.error(f"Error querying the vector DB: {e}")

