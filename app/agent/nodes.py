"""Run model and tool steps, then choose the next step."""
import logging
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.agent.loop import InvokableModel, MAX_MODEL_CALLS, MAX_TOOL_CALLS_PER_RESPONSE
from app.agent.state import AgentState
from app.agent.tool_execution import execute_tool

logger = logging.getLogger(__name__)


def agent_node(state: AgentState, model: InvokableModel) -> dict:
    if state["model_calls"] >= MAX_MODEL_CALLS:
        raise ValueError("Agent reached its model-call limit.")
    response = model.invoke(context_messages(state["messages"]))
    if not isinstance(response, AIMessage):
        raise ValueError("Expected an AIMessage from the model.")
    if response.invalid_tool_calls:
        raise ValueError("The model returned malformed tool arguments.")
    # Keep tool calls but leave out the provider's reasoning data.
    response = response.model_copy(update={"additional_kwargs": {}})
    calls = state["model_calls"] + 1
    if response.tool_calls:
        if calls >= MAX_MODEL_CALLS:
            raise ValueError("Agent reached its model-call limit.")
        if len(response.tool_calls) > MAX_TOOL_CALLS_PER_RESPONSE:
            raise ValueError("Too many tool calls in one response.")
    elif not isinstance(response.content, str) or not response.content.strip():
        raise ValueError("The model returned no final answer.")
    else:
        logger.info("Final response generated.")
    return {"messages": [response], "model_calls": calls}


def tool_node(state: AgentState, tools=None) -> dict:
    response = state["messages"][-1]
    if not isinstance(response, AIMessage) or not response.tool_calls:
        raise ValueError("Tool node requires an assistant tool request.")
    request = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
    messages = state["messages"]
    latest_user = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
    searches = sum(isinstance(m, ToolMessage) and m.name == "web_search" for m in messages[latest_user:])
    results = []
    for call in response.tool_calls:
        if call["name"] == "web_search" and searches >= 1:
            results.append(ToolMessage(content="Search limit reached: one search per turn.", tool_call_id=call["id"], name="web_search"))
        else:
            results.append(execute_tool(call, tools=tools, request=request))
            searches += call["name"] == "web_search"
    return {"messages": results}


def route_after_agent(state: AgentState) -> Literal["tools", "end"]:
    response = state["messages"][-1]
    if not isinstance(response, AIMessage):
        raise ValueError("Routing requires an AIMessage.")
    return "tools" if response.tool_calls else "end"

def context_messages(messages: list) -> list:
    """Keep the latest 10 user turns without splitting tool-call/result groups."""
    starts = [i for i, message in enumerate(messages) if isinstance(message, HumanMessage)]
    if len(starts) <= 10:
        return list(messages)
    prefix = [message for message in messages[:starts[0]] if isinstance(message, SystemMessage)]
    return [*prefix, *messages[starts[-10]:]]