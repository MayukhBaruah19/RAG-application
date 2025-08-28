from langchain_ollama import ChatOllama, OllamaEmbeddings


LLM_MODEL_LIST = [
    "mistral:7b"
]

def get_llm(model_name=None):
    if model_name is None:
        model_name = LLM_MODEL_LIST[0]
    return ChatOllama(model=model_name)

def get_embeddings():
    """Load Ollama embeddings."""
    return OllamaEmbeddings(model="nomic-embed-text:v1.5")

