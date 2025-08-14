from langchain_community.llms import Ollama
from langchain_huggingface import HuggingFaceEmbeddings

# List of available LLMs in local Ollama server
LLM_MODEL_LIST = [
    "deepseek-r1:1.5b",
    "gemma3:latest"
]


def get_llm(model_name: str):
    return Ollama(model=model_name)


def get_embeddings():
    """
    Returns HuggingFaceEmbeddings object to be used with Chroma.
    """
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
