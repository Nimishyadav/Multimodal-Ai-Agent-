"""Search with Tavily and use SerpApi as a backup when needed."""
import json
import socket
import time
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlencode
from urllib.request import Request, urlopen

SEARCH_ENDPOINT = "https://api.tavily.com/search"
TIMEOUT_SECONDS = 12
CACHE_SECONDS = 300
MAX_RESPONSE_BYTES = 2_000_000


class SearchError(ValueError):
    """Safe provider error; never includes response bodies or credentials."""


def safe_source_url(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = urlsplit(value)
        return parsed.scheme in {"http", "https"} and bool(parsed.hostname) and not parsed.username and not parsed.password
    except ValueError:
        return False


class SearchClient:
    """Keep the key and bounded cache inside one browser session."""

    def __init__(self, api_key: str | None, backup_key: str | None = None) -> None:
        self._api_key = api_key
        self._backup_key = backup_key
        self._primary_cooldown = 0
        self._cache: dict[str, tuple[float, dict]] = {}

    def search(self, query: str) -> dict:
        query = query.strip()
        if not query or len(query) > 500:
            raise SearchError("Search query must contain 1–500 characters.")
        cache_key = " ".join(query.lower().split())
        cached = self._cache.get(cache_key)
        if cached and time.monotonic() - cached[0] < CACHE_SECONDS:
            return {**cached[1], "cached": True}
        if self._api_key and self._primary_cooldown <= time.monotonic():
            try:
                result = self._tavily_search(query)
                result['provider'] = 'Tavily'
                return result
            except SearchError:
                if not self._backup_key:
                    raise
                self._primary_cooldown = time.monotonic() + 60
        if not self._backup_key:
            raise SearchError("No available search API. Configure TAVILY_API_KEY or SERPAPI_API_KEY.")
        # Backup searches and travel lookups share the SerpApi quota.
        url = "https://serpapi.com/search.json?" + urlencode({
            'engine': 'google', 'q': query, 'num': 3, 'api_key': self._backup_key})
        try:
            with urlopen(url, timeout=TIMEOUT_SECONDS) as response:
                raw = response.read(MAX_RESPONSE_BYTES + 1)
            if len(raw) > MAX_RESPONSE_BYTES:
                raise SearchError("Backup search response was too large.")
            data = json.loads(raw)
            if not isinstance(data, dict) or data.get('error'):
                raise SearchError("Backup search failed. Check SerpApi quota/key.")
            rows = data.get('organic_results', [])
            if not isinstance(rows, list):
                raise SearchError("Backup search returned an invalid result format.")
            sources = [{'title': str(row.get('title', 'Source'))[:200], 'url': row['link'][:2000],
                        'snippet': str(row.get('snippet', ''))[:1000]} for row in rows[:3]
                       if isinstance(row, dict) and safe_source_url(row.get('link'))]
        except HTTPError as error:
            status = error.code
            error.close()
            raise SearchError(f"Backup search unavailable (HTTP {status}). Check SerpApi access/quota.") from None
        except (URLError, TimeoutError, OSError, ValueError) as error:
            if isinstance(error, SearchError):
                raise
            raise SearchError("Backup search could not connect or returned invalid data.") from None
        result = {'status': 'ok' if sources else 'no_results', 'query': query,
                  'searched_at': datetime.now(timezone.utc).isoformat(), 'cached': False,
                  'sources': sources, 'provider': 'SerpApi (backup)'}
        if len(self._cache) >= 50:
            self._cache.pop(next(iter(self._cache)))
        self._cache[cache_key] = (time.monotonic(), result)
        return result

    def _tavily_search(self, query: str) -> dict:
        query = query.strip()
        if not query or len(query) > 500:
            raise SearchError("Search query must contain 1–500 characters.")
        if not self._api_key:
            raise SearchError("Web search is disabled. Set TAVILY_API_KEY locally and restart the app.")
        cache_key = " ".join(query.lower().split())
        cached = self._cache.get(cache_key)
        if cached and time.monotonic() - cached[0] < CACHE_SECONDS:
            return {**cached[1], "cached": True}
        payload = {
            "query": query, "search_depth": "basic", "max_results": 3,
            "include_answer": False, "include_raw_content": False,
            "auto_parameters": False,
        }
        request = Request(
            SEARCH_ENDPOINT,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=TIMEOUT_SECONDS) as response:
                raw = response.read(MAX_RESPONSE_BYTES + 1)
            if len(raw) > MAX_RESPONSE_BYTES:
                raise SearchError("Search provider response was too large.")
            data = json.loads(raw)
        except HTTPError as error:
            messages = {
                401: "Invalid Tavily key. Check TAVILY_API_KEY in your local .env.",
                403: "Tavily search access was denied. Check your account.",
                429: "Tavily rate limit reached. Try again after the limit resets.",
                432: "Tavily plan quota reached. Check your free allowance.",
                433: "Tavily usage limit reached. Check your account limits.",
            }
            error.close()
            raise SearchError(messages.get(error.code, "Search provider could not complete the request.")) from None
        except (TimeoutError, socket.timeout):
            raise SearchError("Web search timed out. Try again later.") from None
        except URLError:
            raise SearchError("Web search could not connect. Check your network.") from None
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise SearchError("Search provider returned an invalid response.") from None
        if not isinstance(data, dict) or not isinstance(data.get("results"), list):
            raise SearchError("Search provider returned an invalid result format.")
        sources = []
        for row in data["results"][:3]:
            if not isinstance(row, dict) or not safe_source_url(row.get("url")):
                continue
            sources.append({
                "title": str(row.get("title") or "Source")[:200],
                "url": row["url"][:2000],
                "snippet": str(row.get("content") or "")[:1000],
            })
        result = {
            "status": "ok" if sources else "no_results",
            "query": query,
            "searched_at": datetime.now(timezone.utc).isoformat(),
            "cached": False,
            "sources": sources,
        }
        # Cache successful results only, and keep the cache size limited.
        if len(self._cache) >= 50:
            self._cache.pop(next(iter(self._cache)))
        self._cache[cache_key] = (time.monotonic(), result)
        return result