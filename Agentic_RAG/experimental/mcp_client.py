import asyncio
from typing import TypedDict, List, cast

from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession


class RetrievalItem(TypedDict):
    content: str
    score: float
    metadata: dict


async def main():
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "Agentic_RAG.mcp_server"]
    )

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:

            tools = await session.list_tools()
            print("Available tools:", tools)

            raw_result = await session.call_tool(
                "retrieve_from_vector_db",
                {
                    "query": "What is an AI Agent according to this book?",
                    "k": 3
                }
            )

            results = cast(List[RetrievalItem], raw_result)

            print("\nTool result:\n")
            for item in results:
                print(item["content"][:200], "\n")


if __name__ == "__main__":
    asyncio.run(main())
