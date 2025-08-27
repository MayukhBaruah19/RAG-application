from langchain_ollama import ChatOllama,OllamaEmbeddings


LLM_MODEL_LIST = [
    "llama2:latest"
]


def get_llm(model_name: str):
    return ChatOllama(model=model_name)


def get_embeddings():
    """
    Returns OllamaEmbeddings object to be used with Chroma.
    """
    return OllamaEmbeddings(model="nomic-embed-text:v1.5")