from langchain_ollama import OllamaEmbeddings

def get_embeddings(model_name="nomic-embed-text:v1.5"):
    return OllamaEmbeddings(model=model_name)