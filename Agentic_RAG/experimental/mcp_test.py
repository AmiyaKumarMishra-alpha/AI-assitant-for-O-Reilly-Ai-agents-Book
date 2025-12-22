import asyncio
from typing import List, TypedDict, cast
import anyio

from pydantic import BaseModel
from mcp.server.fastmcp import FastMCP
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters


# ============================================================
# Tool schemas
# ============================================================

class RetrieveInput(BaseModel):
    query: str
    k: int


class RetrievalItem(BaseModel):
    content: str
    score: float
    metadata: dict


# ============================================================
# MCP SERVER
# ============================================================

mcp = FastMCP("SingleFile MCP Server")


@mcp.tool(
    name="retrieve_from_vector_db",
    description="Retrieve relevant chunks from vector database"
)
def retrieve_from_vector_db(query: str, k: int) -> List[RetrievalItem]:
    # 🔹 mock data (replace with FAISS later)
    return [
        RetrievalItem(
            content=f"Mock answer for: {query}",
            score=0.95,
            metadata={"source": "mock"}
        )
    ]


async def run_server():
    await mcp.run_stdio_async()


# ============================================================
# MCP CLIENT
# ============================================================

class RetrievalItemDict(TypedDict):
    content: str
    score: float
    metadata: dict


async def run_client():
    server_params = StdioServerParameters(
        command="python",
        args=[__file__, "--server"]
    )

    async with stdio_client(server_params) as (r, w):
        async with ClientSession(r, w) as session:

            tools = await session.list_tools()
            print("Available tools:", tools)

            raw = await session.call_tool(
                "retrieve_from_vector_db",
                {"query": "What is an AI Agent?", "k": 3}
            )

            results = cast(List[RetrievalItemDict], raw)

            print("\nTool result:\n")
            for r in results:
                print(r["content"])


# ============================================================
# ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    import sys

    if "--server" in sys.argv:
        anyio.run(run_server)
    else:
        asyncio.run(run_client())
