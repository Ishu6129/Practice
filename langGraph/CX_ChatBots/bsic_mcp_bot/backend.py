import asyncio
import os
import sys
from typing import Annotated, TypedDict

from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# -----------------------------
# Environment
# -----------------------------
load_dotenv()

os.environ["LANGSMITH_PROJECT"] = "chatbot_MCP"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SERVER_PATH = os.path.join(BASE_DIR, "arithMcp.py")

# -----------------------------
# MCP Client
# -----------------------------
client = MultiServerMCPClient(
    {
        "arith": {
            "transport": "stdio",
            "command": sys.executable,
            "args": [
                "-u",
                SERVER_PATH,
            ],
        }
    }
)

# -----------------------------
# LLM
# -----------------------------
llm = ChatGroq(
    model="openai/gpt-oss-20b",
)

# -----------------------------
# LangGraph State
# -----------------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# -----------------------------
# Build Graph
# -----------------------------
async def build_graph():

    tools = await client.get_tools()


    llm_with_tools = llm.bind_tools(tools)

    async def chatbot(state: ChatState):
        response = await llm_with_tools.ainvoke(state["messages"])
        return {"messages": [response]}

    graph = StateGraph(ChatState)

    graph.add_node("chatbot", chatbot)
    graph.add_node("tools", ToolNode(tools))

    graph.add_edge(START, "chatbot")

    graph.add_conditional_edges("chatbot",tools_condition)

    graph.add_edge("tools", "chatbot")

    return graph.compile()


# -----------------------------
# Main
# -----------------------------
async def main():

    app = await build_graph()

    result = await app.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content="factorial of 6 ?"
                )
            ]
        }
    )

    print("\n========================")
    print("RESPONSE")
    print("========================\n")

    print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())