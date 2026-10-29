"""Define the error shared by the agents and the chat interface."""


class AgentExecutionError(ValueError):
    """Safe validation error; excludes provider payloads and credentials."""
