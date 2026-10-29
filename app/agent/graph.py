"""Build the LangGraph workflow used for a conversation."""
from collections.abc import Sequence
from functools import partial

from langchain_core.messages import AIMessage, BaseMessage
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agent.loop import InvokableModel, MAX_MODEL_CALLS
from app.agent.nodes import agent_node, route_after_agent, tool_node
from app.agent.state import AgentState
from app.llm.client import build_messages

GRAPH_RECURSION_LIMIT = 2 * MAX_MODEL_CALLS + 2


def build_graph(model: InvokableModel, checkpointer=None, tools=None) -> CompiledStateGraph:
    """The model is already bound to its tools; optional thread-level checkpointer."""
    builder = StateGraph(AgentState)
    builder.add_node("agent", partial(agent_node, model=model))
    builder.add_node("tools", partial(tool_node, tools=tools))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", route_after_agent, {"tools": "tools", "end": END})
    builder.add_edge("tools", "agent")
    return builder.compile(checkpointer=checkpointer)


def ask_graph(
    graph: CompiledStateGraph, question: str,
    history: Sequence[BaseMessage] | None = None,
) -> str:
    question = question.strip()
    if not question:
        raise ValueError("Question cannot be empty.")
    # Start each question with fresh working state and the supplied chat history.
    try:
        result = graph.invoke(
            {"messages": build_messages(question, history), "model_calls": 0},
            config={"recursion_limit": GRAPH_RECURSION_LIMIT},
        )
    except GraphRecursionError:
        raise ValueError("Graph reached its execution limit.") from None
    final = result["messages"][-1]
    if not isinstance(final, AIMessage) or final.tool_calls or not isinstance(final.content, str) or not final.content.strip():
        raise ValueError("Graph did not produce a final answer.")
    return final.content.strip()