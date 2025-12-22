import time
import numpy as np
from Agentic_RAG.agents.query_analyzer import QueryAnalyzer
from Agentic_RAG.agents.retrieval_agent import RetrievalAgent

class MasterAgent:
    def __init__(self):
        self.query_analyzer = QueryAnalyzer()
        self.retrieval_agent = RetrievalAgent()
        #Thressold 
        self.max_latency_sec = 30
        self.max_avg_distance = 450
        self.min_answer_length = 20


    def evaluates(self, retrieval_result: dict):
        scores = retrieval_result.get("retrieval_scores", [])
        answer = retrieval_result.get("answer", "")
        latency = retrieval_result.get("latency_sec", 0)
        avg_distance = np.mean(scores) if scores else float('inf')
        answer_length = len(answer)

        evaluation = {
            "avg_distance": round(avg_distance, 2),
            "latency_sec": latency,
            "answer_length": answer_length,
            "latency_ok": latency <= self.max_latency_sec,
            "retrieval_ok": avg_distance <= self.max_avg_distance,
            "answer_ok": answer_length >= self.min_answer_length
        }
        evaluation["accepted"] = all([
            evaluation["latency_ok"],
            evaluation["retrieval_ok"],
            evaluation["answer_ok"]
        ])
        return evaluation
    def run(self, user_query: str):
        start_time = time.time()

        # Step 1: Analyze the query
        analysis_payload = self.query_analyzer.analyze(user_query)

        # Step 2: Retrieve and answer
        retrieval_result = self.retrieval_agent.answer(analysis_payload)

        # Step 3: Evaluate the response
        evaluation = self.evaluates(retrieval_result)

        total_latency = round(time.time() - start_time, 3)

        return {
            "query": user_query,
            "analysis": analysis_payload,
            "answer": retrieval_result["answer"],
            "sources": retrieval_result["sources"],
            "evaluation": evaluation,
            "total_latency_sec": total_latency,
            "decision": "ACCEPT" if evaluation["accepted"] else "REVIEW"
        }