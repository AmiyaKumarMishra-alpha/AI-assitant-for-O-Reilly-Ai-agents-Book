import time
import langchain_ollama as OllamaLLM
from Agentic_RAG.tools.retrieval import data_retriever


class RetrievalAgent:
    def __init__(self):
        self.llm = OllamaLLM.OllamaLLM(
            model="llama3.1:8b",
            temperature=0.2
        )

    def answer(self, analysis_payload: dict):
        start_time = time.time()

        search_query = analysis_payload.get("search_query", "")

        #call retrival tool
        chunks = data_retriever(search_query)

        #build context
        context ="\n\n".join([c["content"] for c in chunks])

        #grounded prompt
        prompt = f"""
        Use the following context to answer the question.
        You are a book assistant.
        Rules:
        - Answer based ONLY on the provided context.
        - If the answer is not in the context, respond with "I don't know".
        Context:
        {context}
        Question:   
        {analysis_payload.get("original_query", "")}
        Answer:
        """
        answer = self.llm.invoke(prompt).strip()
        latency = round(time.time() - start_time, 3)
        return {
            "answer": answer.strip(),
            "sources": [
                c["metadata"].get("page", "N/A") for c in chunks
            ],
            "retrieval_scores": [c["score"] for c in chunks],
            "latency_sec": latency
        }


