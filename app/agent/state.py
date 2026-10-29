"""Define the messages and working values passed between graph steps."""
from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # Each node adds its new messages to the conversation.
    messages: Annotated[list[BaseMessage], add_messages]
    model_calls: int