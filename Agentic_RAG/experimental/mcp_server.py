from typing import List
from pydantic import BaseModel
from mcp.server import Server

from Agentic_RAG.tools.retrival import data_retriever


# -----------------------------
# Input schema
# -----------------------------
class RetrieveInput(BaseModel):
    query: str
    k: int


# -----------------------------
# Output schema
# -----------------------------
class RetrievalItem(BaseModel):
    content: str
    score: float
    metadata: dict


server = Server("VectorDB MCP Server")


def retrieve_from_vector_db(input: RetrieveInput) -> List[RetrievalItem]:
    results = data_retriever(input.query, input.k)
    return [RetrievalItem(**r) for r in results]


server.add_tool( #type: ignore
    name="retrieve_from_vector_db",
    description="Retrieve relevant chunks from the vector database",
    func=retrieve_from_vector_db,
    input_model=RetrieveInput,
    output_model=List[RetrievalItem],
)


if __name__ == "__main__":
    server.run_stdio() #type: ignore
