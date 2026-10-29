"""Run the assistant from a terminal using the shared orchestrator."""
import logging
import sys
from openai import OpenAIError
from google.genai.errors import APIError as GoogleAPIError
from httpx import TimeoutException, ConnectError
from groq import APIConnectionError, APIStatusError, APITimeoutError, AuthenticationError, RateLimitError
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from app.config.settings import load_settings
from app.llm.client import MAX_HISTORY_MESSAGES, create_model, create_data_model
from app.agent.graph import ask_graph as ask_model, build_graph
from app.tools.calculator import calculator

EXIT_WORDS = {"exit", "quit"}


def friendly_error(error: Exception) -> str:
    """Show readable errors without exposing keys or raw API responses."""
    if isinstance(error, OpenAIError):
        status = getattr(error, "status_code", None)
        if status == 429:
            return "OpenRouter quota/rate limit reached. Check your OpenRouter account and wait for reset."
        if status in {401, 403}:
            return "OpenRouter access denied. Check OPENROUTER_API_KEY and model permissions."
        if status == 404:
            return "OpenRouter model unavailable. Check EXCEL_MODEL."
        return "OpenRouter request failed. Check provider access or try later."
    if type(error).__name__ == "AgentExecutionError":
        return str(error)
    google_error = error if isinstance(error, GoogleAPIError) else error.__cause__
    if isinstance(google_error, GoogleAPIError):
        if google_error.code in {401, 403}:
            return "Gemini access denied. Check GOOGLE_API_KEY and your AI Studio project."
        if google_error.code == 429:
            return "Gemini quota reached. Check AI Studio limits and wait for reset."
        if google_error.code == 404:
            return "Gemini model unavailable. Check GEMINI_MODEL and account access."
        return "Gemini could not complete the request. Try again later."
    if isinstance(error, AuthenticationError):
        return "Invalid Groq key. Check GROQ_API_KEY locally."
    if isinstance(error, RateLimitError):
        return "Groq quota reached. Check your account limits and wait for reset."
    if isinstance(error, (APITimeoutError, TimeoutException, TimeoutError)):
        return "Request timed out. Check your connection and try again."
    if isinstance(error, (APIConnectionError, ConnectError)):
        return "Connection failed. Check your network."
    if isinstance(error, APIStatusError):
        return f"API error ({error.status_code}). Check model access."
    return "Invalid input or no usable model response. Check your settings or try a shorter question."


def main() -> int:
    from pathlib import Path
    from app.agent.session import Conversation
    from app.history.store import HistoryStore
    from app.memory.store import MemoryStore
    from app.rag.documents import DocumentIndex
    from app.integrations.web_search import SearchClient
    from app.integrations.travel_prices import TravelPriceClient
    from app.config.settings import configured_key
    from app.tools.web_search import create_search_tool
    from app.tools.document_search import create_document_tool
    from app.integrations.web_reader import WebPageClient
    from app.rag.anything_rag import create_document_index
    from app.multi_agent.system import handle_multi_agent
    from app.spreadsheet.analysis import read_table, detect_header_row
    from app.spreadsheet.planner import create_excel_model
    try:
        settings = load_settings()
        model = create_model(settings)
        data_model = create_data_model(model)
        search = SearchClient(settings.tavily_api_key, configured_key("SERPAPI_API_KEY", "SERP_API_KEY"))
        prices = TravelPriceClient(configured_key("SERPAPI_API_KEY", "SERP_API_KEY"))
        web_client = WebPageClient()
        documents = create_document_index()
        conversation = Conversation(model, memory_store=MemoryStore(), history_store=HistoryStore(),
            additional_tools=[create_search_tool(search), create_document_tool(documents)],
            document_index=documents)
    except Exception:
        print("Configuration/storage error. Check local settings.", file=sys.stderr)
        return 1
    print(f"AI Agent | Orchestrator | Provider: {settings.provider}")
    print("Commands: /pdf PATH, /table PATH, /new, exit")
    pending, frame = None, None
    try:
        while True:
            try:
                question = input("\nYou: ").strip()
                if question.lower() in EXIT_WORDS:
                    break
                if not question:
                    continue
                if question == "/new":
                    conversation.clear()
                    pending = None
                    print("New conversation started.")
                    continue
                if question.startswith(("/pdf ", "/table ")):
                    command, filename = question.split(" ", 1)
                    path = Path(filename.strip().strip(chr(34)).strip(chr(39)))
                    if path.stat().st_size > 10 * 1024 * 1024:
                        raise ValueError("File exceeds 10 MB.")
                    content = path.read_bytes()
                    if command == "/pdf":
                        documents.add(path.name, content)
                        print("PDF indexed.")
                    else:
                        header = detect_header_row(content, path.name)
                        frame = read_table(content, path.name, header_row=header)
                        print(f"Table loaded: {len(frame)} rows; header {header}.")
                    continue
                answer, pending = handle_multi_agent(question, conversation, model, search, prices,
                                                      frame=frame, pending=pending, data_model=data_model, economy_mode=True, web_client=web_client)
                print("\nAI:", answer)
                for event in conversation.activity:
                    print("Activity:", event)
            except (KeyboardInterrupt, EOFError):
                break
            except (OSError, ValueError):
                print("Request/file could not be processed. Check details or try a smaller task.", file=sys.stderr)
            except Exception as error:
                print(friendly_error(error), file=sys.stderr)
    finally:
        conversation.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())