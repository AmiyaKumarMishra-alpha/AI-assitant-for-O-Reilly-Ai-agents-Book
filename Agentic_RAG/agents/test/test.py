from Agentic_RAG.agents.query_analyzer import QueryAnalyzer

agent = QueryAnalyzer()

query = "What is an AI Agent according to this book?"
result = agent.analyze(query)

for k, v in result.items():
    print(f"{k}: {v}")
