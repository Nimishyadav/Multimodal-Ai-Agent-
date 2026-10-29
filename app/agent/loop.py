"""Earlier manual agent loop, kept as a reference alongside the LangGraph version."""
import logging
from collections.abc import Sequence
from typing import Protocol

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from app.agent.tool_execution import execute_tool

from app.llm.client import ask_model, build_messages
# This is the earlier manual loop; current chat uses the graph workflow.

logger = logging.getLogger(__name__)
MAX_MODEL_CALLS = 4
MAX_TOOL_CALLS_PER_RESPONSE = 4


class InvokableModel(Protocol):
    def invoke(self, messages: list[BaseMessage]) -> AIMessage: ...


def run_agent(
    model: InvokableModel, question: str,
    history: Sequence[BaseMessage] | None = None,
) -> str:
    """Return a final answer or stop safely at the model-call limit."""
    # Keep temporary tool messages within this request.
    response = ask_model(model, question, history)
    messages = build_messages(question.strip(), history)
    for iteration in range(MAX_MODEL_CALLS):
        if iteration:
            response = model.invoke(messages)
        if not isinstance(response, AIMessage):
            raise ValueError("Expected an AIMessage from the model.")
        if response.invalid_tool_calls:
            raise ValueError("The model returned malformed tool arguments.")
        messages.append(response)
        if not response.tool_calls:
            if not isinstance(response.content, str) or not response.content.strip():
                raise ValueError("The model returned no final answer.")
            logger.info("Final response generated.")
            return response.content.strip()
        if iteration == MAX_MODEL_CALLS - 1:
            raise ValueError("Agent reached its model-call limit.")
        if len(response.tool_calls) > MAX_TOOL_CALLS_PER_RESPONSE:
            raise ValueError("Too many tool calls in one response.")
        for call in response.tool_calls:
            messages.append(execute_tool(call))
    raise ValueError("Agent stopped without a final answer.")