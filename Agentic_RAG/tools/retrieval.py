from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import FAISS 
from pathlib import Path

VECTOR_DB_PATH = "D:/Practice/rag PRACTICE/vector_db"
TOP_K = 5 

#load embedder 
embeddings = OllamaEmbeddings(
    model="nomic-embed-text")

#load vector DB
vector_db = FAISS.load_local(
    str(VECTOR_DB_PATH),
    embeddings,
    allow_dangerous_deserialization=True
)
def data_retriever(search_query: str, k: int = TOP_K):
    results = vector_db.similarity_search_with_score(search_query, k=k)

    retrieved = []

    for item in results:
        doc = item[0]       # Document
        score = item[1]     # similarity score

        retrieved.append({
            "content": doc.page_content,
            "score": score,
            "metadata": doc.metadata
        })

    return retrieved