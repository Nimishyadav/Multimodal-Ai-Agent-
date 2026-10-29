"""Read model settings and API keys from the local environment."""
import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    api_key: str = field(repr=False)
    model: str = "openai/gpt-oss-20b"
    tavily_api_key: str | None = field(default=None, repr=False)
    provider: str = "groq"


def configured_key(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name, "").strip()
        if value and not value.startswith("your_"):
            return value
    return None


def load_settings() -> Settings:
    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)
    provider = os.getenv("LLM_PROVIDER", "auto").strip().lower()
    google_key = configured_key("GOOGLE_API_KEY", "GEMINI_API_KEY")
    groq_key = configured_key("GROQ_API_KEY")
    if provider == "auto":
        provider = "groq" if groq_key else "gemini"
    if provider not in {"gemini", "groq"}:
        raise ValueError("LLM_PROVIDER must be auto, gemini or groq.")
    key = google_key if provider == "gemini" else groq_key
    if not key:
        variable = "GOOGLE_API_KEY" if provider == "gemini" else "GROQ_API_KEY"
        raise ValueError(f"Set {variable} in your local .env file.")
    if provider == "gemini":
        model = os.getenv("GEMINI_MODEL") or os.getenv("PDF_MODEL") or "gemini-3.1-flash-lite"
    else:
        model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    if not model.strip():
        raise ValueError("Model name cannot be empty.")
    return Settings(api_key=key, model=model.strip(), provider=provider,
                    tavily_api_key=configured_key("TAVILY_API_KEY"))