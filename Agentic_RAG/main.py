from Agentic_RAG.agents.master_agent import MasterAgent



def run_once():
    query = "What is an AI Agent according to this book?"

    master = MasterAgent()
    result = master.run(query)

    print("\n=== FINAL ANSWER ===\n")
    print(result["answer"])

    print("\n=== EVALUATION ===\n")
    print(result["evaluation"])

    print("\nDecision:", result["decision"])
    print("Total latency:", result["total_latency_sec"], "sec")


if __name__ == "__main__":
    run_once()
