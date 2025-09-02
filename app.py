import os
import streamlit as st
from src.data.data_loader import load_pdf, split_documents
from src.models.llm import get_llm, LLM_MODEL_LIST
from src.models.embedding import get_embeddings
from src.QAchain.retrieval_qa import create_qa_chain
from langchain_community.vectorstores import Chroma
import logging

logger = logging.getLogger(__name__)

DATA_FOLDER = "data"          
CHROMA_INDEX_DIR = "chroma_index"
TOP_K = 3 

os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(CHROMA_INDEX_DIR, exist_ok=True)

# Define class and subject mapping to PDF files
CLASS_SUBJECT_MAPPING = {
    "General": {  # Add General option
        "All Subjects": ["chapter1.pdf", "Chemical Reactions.pdf", "Acids, Bases.pdf", "Metals and Non-metals.pdf", "Life Processes.pdf", "info2.pdf"]
    },
    "Class 9": {
        "Science": ["chapter1.pdf"],
        "Mathematics": ["chapter1.pdf"],
        "Social Science": ["chapter1.pdf"]
    },
    "Class 10": {
        "Science": ["Chemical Reactions.pdf", "Acids, Bases.pdf", "Metals and Non-metals.pdf", "Life Processes.pdf","info2.pdf"],
        "Mathematics": ["chapter1.pdf"],
        "History": ["info2.pdf"]
    },
    "Class 11": {   
        "Physics": ["chapter1.pdf" ],
        "Chemistry": ["chapter1.pdf"],
        "Biology": ["chapter1.pdf"]
    }
}


st.set_page_config(page_title="Edukracy Chatbot", page_icon="📚")
st.title("📚 Edukracy - Educational Chatbot")

# Sidebar for selections
st.sidebar.header("Academic Selection")

# Class selection with General option
class_options = ["General"] + list(CLASS_SUBJECT_MAPPING.keys())
class_options.remove("General")  # Remove duplicate
class_options = ["General"] + [c for c in class_options if c != "General"]
selected_class = st.sidebar.selectbox("Select Class", class_options)

# Subject selection based on class
if selected_class == "General":
    available_subjects = list(CLASS_SUBJECT_MAPPING[selected_class].keys())
    selected_subject = st.sidebar.selectbox("Select Subject", available_subjects)
else:
    available_subjects = list(CLASS_SUBJECT_MAPPING[selected_class].keys())
    selected_subject = st.sidebar.selectbox("Select Subject", available_subjects)

# Model selection
selected_model = st.sidebar.selectbox("Select AI Model", LLM_MODEL_LIST)

# Show selected PDF files
selected_pdfs = CLASS_SUBJECT_MAPPING[selected_class][selected_subject]
st.sidebar.info(f"**Selected Materials:**\n" + "\n".join([f"• {pdf}" for pdf in selected_pdfs]))

@st.cache_resource(show_spinner=False)
def get_vector_db(_selected_class, _selected_subject):
    """Get or create vector database for selected class and subject"""
    try:
        embedding = get_embeddings()
        
        # Get PDF files for selected class and subject
        pdf_files = CLASS_SUBJECT_MAPPING[_selected_class][_selected_subject]
        
        # Create a unique collection name for each class-subject combination
        if _selected_class == "General":
            collection_name = "general_knowledge"
        else:
            collection_name = f"{_selected_class.replace(' ', '_')}_{_selected_subject.replace(' ', '_')}".lower()
        
        # Check if we have an existing vector store
        vector_db = None
        try:
            vector_db = Chroma(
                persist_directory=CHROMA_INDEX_DIR,
                embedding_function=embedding,
                collection_name=collection_name
            )
            # Test if collection exists and has documents
            if vector_db._collection.count() > 0:
                st.sidebar.success("✓ Loaded existing knowledge base")
                return vector_db
        except:
            # Collection doesn't exist, we'll create it
            vector_db = None
        
        # Process PDF files
        all_chunks = []
        processed_files = set()  # Track processed files to avoid duplicates
        
        for pdf_file in pdf_files:
            if pdf_file in processed_files:
                continue
                
            pdf_path = os.path.join(DATA_FOLDER, pdf_file)
            
            if not os.path.exists(pdf_path):
                st.sidebar.warning(f"File not found: {pdf_file}")
                continue
                
            try:
                # Load and split PDF
                docs = load_pdf(pdf_path)
                chunks = split_documents(docs)
                
                # Add metadata
                for chunk in chunks:
                    chunk.metadata.update({
                        "source": pdf_file,
                        "class": _selected_class,
                        "subject": _selected_subject,
                        "full_path": pdf_path
                    })
                
                all_chunks.extend(chunks)
                processed_files.add(pdf_file)
                st.sidebar.success(f"✓ Processed {pdf_file}")
                
            except Exception as e:
                st.sidebar.error(f"Error processing {pdf_file}: {str(e)}")
                continue
        
        if not all_chunks:
            st.error("No documents were processed successfully.")
            return None
        
        # Create new vector store
        vector_db = Chroma.from_documents(
            documents=all_chunks,
            embedding=embedding,
            persist_directory=CHROMA_INDEX_DIR,
            collection_name=collection_name
        )
        
        vector_db.persist()
        st.sidebar.success("✓ Created new knowledge base")
        return vector_db
        
    except Exception as e:
        st.error(f"Error initializing vector database: {str(e)}")
        return None

def get_qa_chain(_selected_class, _selected_subject):
    """Create QA chain for selected class and subject"""
    try:
        llm = get_llm(selected_model)
        vector_db = get_vector_db(_selected_class, _selected_subject)
        
        if vector_db is None:
            st.error("No knowledge base available for the selected class and subject.")
            return None
        
        # Create retriever with metadata filtering
        retriever = vector_db.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": TOP_K}
        )
        
        # Use your existing QA chain function
        qa_chain = create_qa_chain(llm, retriever, TOP_K)
        return qa_chain
        
    except Exception as e:
        st.error(f"Error creating QA chain: {str(e)}")
        return None

# Main chat interface
if selected_class == "General":
    st.header("🌐 General Knowledge Assistant")
else:
    st.header(f"🧠 {selected_class} - {selected_subject} Assistant")

# Show mode information
if selected_class == "General":
    st.info("🔍 **General Mode**: I'll search through all available knowledge to answer your questions.")
else:
    st.info(f"🎯 **Specific Mode**: Focusing on {selected_class} - {selected_subject} materials.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask your question about the subject..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate and display assistant response
    with st.chat_message("assistant"):
        if selected_class == "General":
            spinner_text = "🔍 Searching through all knowledge..."
        else:
            spinner_text = f"🤔 Thinking about {selected_subject}..."
        
        with st.spinner(spinner_text):
            qa_chain = get_qa_chain(selected_class, selected_subject)
            
            if qa_chain:
                try:
                    response = qa_chain.invoke({"query": prompt})
                    answer = response['result']
                    
                    # Display answer
                    st.markdown(answer)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
            else:
                error_msg = "Unable to initialize the question answering system. Please check if the required PDF files are available."
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
