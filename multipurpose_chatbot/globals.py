import os

BACKEND = os.environ.get("BACKEND", "mlx")
global MODEL_ENGINE

from multipurpose_chatbot.engines import load_multipurpose_chatbot_engine
from multipurpose_chatbot.demos import get_demo_class
# if MODEL_ENGINE is None:
MODEL_ENGINE = load_multipurpose_chatbot_engine(BACKEND)



RAG_CURRENT_FILE, RAG_EMBED, RAG_CURRENT_VECTORSTORE = None, None, None

RAG_EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def load_embeddings():
    global RAG_EMBED
    if RAG_EMBED is None:
        from langchain_community.embeddings import HuggingFaceEmbeddings, HuggingFaceBgeEmbeddings
        print(f'LOading embeddings: {RAG_EMBED_MODEL_NAME}')
        RAG_EMBED = HuggingFaceEmbeddings(model_name=RAG_EMBED_MODEL_NAME, model_kwargs={'trust_remote_code':True, "device": "cpu"})
    else:
        print(f'RAG_EMBED ALREADY EXIST: {RAG_EMBED_MODEL_NAME}: {RAG_EMBED=}')
    return RAG_EMBED


def get_rag_embeddings():
    return load_embeddings()


