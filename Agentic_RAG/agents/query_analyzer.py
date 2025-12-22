import time
from langchain_ollama import OllamaLLM, OllamaEmbeddings

class QueryAnalyzer:
    def __init__(self):
        self.llm = OllamaLLM(
            model="llama3.1:8b",
            temperature=0.0
        )

        self.embedder = OllamaEmbeddings(
            model="nomic-embed-text"
        )

    def analyze(self, user_query: str):
        start_time = time.time()

        # 1️ Intent classification
        intent_prompt = f"""
        Classify the intent of the following query into ONE category only:

        - definition
        - explanation
        - comparison
        - factual_lookup
        - out_of_scope
        Query:
        {user_query}

        Return ONLY the category name.
        """
        intent = self.llm.invoke(intent_prompt).strip().lower()

        # 2️ Semantic query rewrite
        rewrite_prompt = f"""
        Rewrite the user query into a SINGLE concise semantic search query.

        Rules:
        - Output ONLY the rewritten query
        - DO NOT explain
        - DO NOT add examples
        - DO NOT use markdown
        - Max 20 words
        User query:
        {user_query}

        Rewritten search query:
        """
        optimized_query = self.llm.invoke(rewrite_prompt).strip().split("\n")[0]

        # Generate embedding
        embedding = self.embedder.embed_query(optimized_query)

        latency = round(time.time() - start_time, 3)

        return {
            "original_query": user_query,
            "intent": intent,
            "search_query": optimized_query,
            "embedding_vector_length": len(embedding),
            "latency_sec": latency
        }
