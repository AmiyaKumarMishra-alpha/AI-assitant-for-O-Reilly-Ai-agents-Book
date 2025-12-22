from Agentic_RAG.agents.master_agent import MasterAgent

agent = MasterAgent()

query = "What is an AI Agent according to this book?"

result = agent.run(query)

for k, v in result.items():
    print(f"\n{k}:\n{v}")
