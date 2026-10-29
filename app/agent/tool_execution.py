"""Run registered tools and check permission before changing saved preferences."""
import logging
import re
from collections.abc import Sequence

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from pydantic import ValidationError
from app.tools.calculator import calculator

logger = logging.getLogger(__name__)


def memory_write_allowed(name: str, request: str) -> bool:
    """Conservative command guard, not an intent or authentication system."""
    request = request.strip().lower()
    patterns = {
        "save_memory": r"^(?:please\s+)?(?:remember|save|store)\b",
        "delete_memory": r"^(?:please\s+)?(?:forget|delete|remove)\b",
    }
    if re.search(r"\b(?:don't|do not|never)\b", request):
        return False
    return bool(re.match(patterns[name], request))


def execute_tool(
    call: dict, tools: Sequence[BaseTool] | None = None, request: str = "",
) -> ToolMessage:
    call_id = call.get("id")
    if not isinstance(call_id, str) or not call_id:
        raise ValueError("Tool call has no valid ID.")
    registry = {tool.name: tool for tool in (tools if tools is not None else [calculator])}
    name = call.get("name")
    tool = registry.get(name)
    arguments = call.get("args")
    if tool is None:
        result = "Tool error: unknown tool."
    elif name in {"save_memory", "delete_memory"} and not memory_write_allowed(name, request):
        result = "Tool error: memory writes require a direct request starting with Remember/Save/Store or Forget/Delete/Remove."
    elif not isinstance(arguments, dict):
        result = "Tool error: arguments must be an object."
    else:
        fields = tool.args_schema.model_fields
        required = {key for key, field in fields.items() if field.is_required()}
        if set(arguments) - set(fields) or not required.issubset(arguments) or any(not isinstance(value, str) for value in arguments.values()):
            result = "Tool error: provide the defined string arguments only."
        else:
            logger.info("Tool selected: %s.", name)
            try:
                result = str(tool.invoke(arguments))
            except ValidationError:
                result = "Tool error: invalid tool arguments."
            except Exception:
                logger.warning("Tool execution failed: %s.", name)
                result = "Tool error: execution failed. Please try again."
            logger.info("Tool completed: %s.", name)
    return ToolMessage(content=result, tool_call_id=call_id, name=name if isinstance(name, str) else None)