from Agentic_RAG.agents.query_analyzer import QueryAnalyzer
from Agentic_RAG.agents.retrieval_agent import RetrievalAgent

query_agent = QueryAnalyzer()
retrieval_agent = RetrievalAgent()

query = "What is an AI Agent according to this book?"

analysis = query_agent.analyze(query)
response = retrieval_agent.answer(analysis)

for k, v in response.items():
    print(f"{k}: {v}")
