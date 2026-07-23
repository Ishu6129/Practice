from langgraph.graph import StateGraph, START, END

from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_groq import ChatGroq

from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

from langgraph.prebuilt import ToolNode, tools_condition
from chatTools import *

from langgraph.graph.message import add_messages
from dotenv import load_dotenv


import  os
os.environ["LANGSMITH_PROJECT"]="chatbot_sqllite"

load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-20b")

#------------------TOOLS--------------------------------
tools = [search_tool, get_stock_price, calculator]
llm_with_tools = llm.bind_tools(tools)
#--------------------------------------------------------

#-----------------STATE-----------------------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
#---------------------------------------------------------

def chat_node(state: ChatState):
    """LLM node that may answer or request a tool call."""
    messages = state['messages']
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)

#----------------------Checkpointer----------------------
conn = sqlite3.connect(database='chatbot.db', check_same_thread=False) # False-allows to use same database with different-different threads
checkpointer = SqliteSaver(conn=conn)
#--------------------------------------------------------


#---------------------------GRAPH-------------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)
graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node",tools_condition)
graph.add_edge('tools', 'chat_node')
#----------------------------------------------------------

chatbot = graph.compile(checkpointer=checkpointer)

def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])

    return list(all_threads)
