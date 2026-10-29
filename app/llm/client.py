"""Create model connections and prepare messages for model calls."""
from collections.abc import Sequence
import os
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models.chat_models import BaseChatModel
from app.config.settings import Settings, configured_key
from app.llm.fallback import FallbackModel
from langchain_openai import ChatOpenAI

# Limit history length. A small message count can still contain a lot of text.
MAX_HISTORY_MESSAGES = 20
SYSTEM_PROMPT = (
    "You are a helpful learning assistant. Explain in simple English "
    "unless another language is requested. Keep answers concise. If unsure, say so. " + "Use the calculator tool for arithmetic. Do not claim a result if the tool fails. " + "Answer ordinary conversation directly without a tool."
)


def provider_models(settings=None):
    """Create the configured model clients without calling an API yet."""
    models = {}
    google = configured_key("GOOGLE_API_KEY", "GEMINI_API_KEY")
    groq = configured_key("GROQ_API_KEY")
    if settings is not None:
        if settings.provider == 'gemini': google = settings.api_key
        if settings.provider == 'groq': groq = settings.api_key
    if google:
        models['gemini'] = ChatGoogleGenerativeAI(
            api_key=google, model=(settings.model if settings and settings.provider=='gemini' else
                                  os.getenv('GEMINI_MODEL', 'gemini-3.1-flash-lite')),
            temperature=0.2, max_tokens=800, thinking_level="minimal", timeout=20, max_retries=0, vertexai=False)
    if groq:
        models['groq'] = ChatGroq(api_key=groq,
            model=settings.model if settings and settings.provider=='groq' else os.getenv('GROQ_MODEL','openai/gpt-oss-20b'),
            temperature=0.2, max_tokens=1024, reasoning_effort="low", timeout=25, max_retries=0)
    router = configured_key('OPENROUTER_API_KEY')
    if router:
        name=os.getenv('CHAT_BACKUP_MODEL','openrouter/free').strip()
        if name != 'openrouter/free' and not name.endswith(':free'):
            raise ValueError('CHAT_BACKUP_MODEL must be a free OpenRouter model.')
        models['openrouter'] = ChatOpenAI(api_key=router,base_url='https://openrouter.ai/api/v1',
            model=name,temperature=0.2,max_tokens=800,timeout=25,max_retries=0)
    return models


def create_model(settings: Settings):
    models=provider_models(settings)
    order=list(dict.fromkeys([settings.provider,'gemini','groq','openrouter']))
    return FallbackModel([(name,models[name]) for name in order if name in models])


def create_data_model(primary):
    """Use OpenRouter for table planning, with the normal text models as backups."""
    from app.spreadsheet.planner import create_excel_model
    if not configured_key('OPENROUTER_API_KEY'):
        return primary
    models = [('openrouter', create_excel_model())]
    models += [(name,model) for name,model in primary.models if name != 'openrouter']
    return FallbackModel(models, primary.shared)


def build_messages(
    question: str, history: Sequence[BaseMessage] | None = None
) -> list[BaseMessage]:
    """System instruction, recent history, then the current question."""
    recent = list(history or [])[-MAX_HISTORY_MESSAGES:]
    return [SystemMessage(content=SYSTEM_PROMPT), *recent, HumanMessage(content=question)]


def ask_model(
    model: BaseChatModel, question: str, history: Sequence[BaseMessage] | None = None
) -> AIMessage:
    question = question.strip()
    if not question:
        raise ValueError("Question cannot be empty.")
    response = model.invoke(build_messages(question, history))
    if not isinstance(response, AIMessage):
        raise ValueError("Expected an AIMessage.")
    if not response.tool_calls and (not isinstance(response.content, str) or not response.content.strip()):
        raise ValueError("The model returned no usable text.")
    return response