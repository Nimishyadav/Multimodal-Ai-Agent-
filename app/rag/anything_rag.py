"""Connect the app to Anything RAG. Parse files, keep source passages and retrieve indexed context."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import uuid
from app.rag.embeddings import profile_key
from app.rag.documents import (
    DocumentIndex,
    DocumentError,
    Chunk,
    requested_page,
    MAX_DOCUMENTS,
    MAX_UPLOAD_BYTES,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

ROOT = Path(__file__).resolve().parents[2]


def runtime_python():
    configured = os.getenv("RAG_PYTHON", "").strip()
    return (
        Path(configured)
        if configured
        else ROOT
        / ".venv-rag"
        / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    )


class RagBackendError(DocumentError):
    pass


class RagWorkerClient:
    """Give each chat its own storage scope."""

    def __init__(self, namespace=None):
        namespace = namespace or uuid.uuid4().hex[:12]
        if not namespace.isalnum():
            raise ValueError("Invalid RAG namespace.")
        self.directory = ROOT / "data" / "multimodal_rag" / namespace
        self.cloud_upload_approved = os.getenv('RAG_CLOUD_UPLOAD_APPROVED','false').lower() == 'true'

    def run(self, action, name, payload=None, query=""):
        python = runtime_python()
        if not python.is_file():
            raise RagBackendError(
                "Multimodal runtime is missing. Run scripts/setup_rag.ps1, then restart."
            )
        self.directory.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256((name + "\0").encode() + (payload or b"")).hexdigest()[
            :20
        ]
        if action in {"parse", "insert"}:
            folder = self.directory / digest
            folder.mkdir(exist_ok=True)
            if payload is not None:
                (folder / "source").mkdir(exist_ok=True)
                (folder / "source" / ("input" + Path(name).suffix.lower())).write_bytes(
                    payload
                )
        else:
            folder = Path(name).resolve()
            if not folder.is_relative_to(self.directory.resolve()):
                raise RagBackendError("RAG scope mismatch.")
        query_cache = None
        if action == 'query':
            marker = folder / 'index_ready.json'
            revision = str(marker.stat().st_mtime_ns) if marker.exists() else 'unready'
            cache_key = hashlib.sha256((profile_key() + '\0' + revision + '\0' + query).encode()).hexdigest()
            query_cache = folder / 'query_cache' / (cache_key + '.json')
            if query_cache.is_file():
                try:
                    cached = json.loads(query_cache.read_text(encoding='utf-8'))
                    if cached.get('ok') and cached.get('context'):
                        return {**cached, 'folder': str(folder), 'cache_hit': True}
                except (OSError, ValueError):
                    pass
        request = {
            "action": action,
            "folder": str(folder),
            "name": Path(name).name,
            "query": query,
            "cloud_upload_approved": getattr(self, "cloud_upload_approved", False) is True,
        }
        # The worker reads keys from the environment, not from command arguments or request files.
        with tempfile.TemporaryDirectory(prefix="rag_request_") as temporary:
            request_path = Path(temporary) / "request.json"
            response_path = Path(temporary) / "response.json"
            request_path.write_text(json.dumps(request), encoding="utf-8")
            environment = os.environ.copy()
            environment["PATH"] = (
                str(python.parent) + os.pathsep + environment.get("PATH", "")
            )
            environment["PYTHONIOENCODING"] = "utf-8"
            try:
                timeout = 360 if action in {"parse", "insert", "index"} else max(10, min(120, int(os.getenv('RAG_QUERY_TIMEOUT_SECONDS', '40'))))
                process = subprocess.run(
                    [
                        str(python),
                        "-B",
                        "-m",
                        "app.rag.anything_rag_worker",
                        str(request_path),
                        str(response_path),
                    ],
                    cwd=ROOT,
                    env=environment,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=timeout,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
                if not response_path.exists():
                    raise RagBackendError(
                        "Multimodal worker could not finish. Check optional dependencies/model downloads."
                    )
                result = json.loads(response_path.read_text(encoding="utf-8"))
            except subprocess.TimeoutExpired:
                raise RagBackendError(
                    "Multimodal processing timed out; no completed index is claimed."
                ) from None
            except (OSError, ValueError):
                raise RagBackendError(
                    "Could not read the multimodal worker result."
                ) from None
        if not result.get("ok"):
            raise RagBackendError(
                result.get("message", "Multimodal processing failed.")
            )
        if query_cache is not None and result.get('context'):
            try:
                query_cache.parent.mkdir(exist_ok=True)
                temporary_cache = query_cache.with_suffix('.' + uuid.uuid4().hex + '.tmp')
                temporary_cache.write_text(json.dumps(result, ensure_ascii=False), encoding='utf-8')
                temporary_cache.replace(query_cache)
            except OSError:
                pass  # A cache write must not discard a successful retrieval.
        return {**result, "folder": str(folder)}


class AnythingRAGIndex(DocumentIndex):
    """Add Anything RAG support without changing how the document tool is called."""

    def __init__(self, client=None):
        super().__init__()
        self.client = client or RagWorkerClient()
        self.backend_status = {}
        self._jobs = {}
        self._rag_queries = {}

    def add(self, name, content):
        return self.add_file(name, content)

    def add_file(self, name, content):
        """Read the document, keep its page evidence, then build the knowledge index."""
        name = Path(name.replace("\\", "/")).name
        digest = hashlib.sha256(content).hexdigest()
        if not content or len(content) > MAX_UPLOAD_BYTES:
            raise DocumentError("Use a non-empty document up to 10 MB.")
        if (name in self._documents and self._documents[name][0] == digest
                and self.backend_status.get(name, {}).get("retrieval") == "RAG-Anything"):
            return {"document": name, "status": "already_indexed"}
        if len(self._documents) >= MAX_DOCUMENTS and name not in self._documents:
            raise DocumentError("Upload at most five indexed documents per chat.")
        if name.lower().endswith(".pdf"):
            from pypdf import PdfReader
            from io import BytesIO

            try:
                pdf = PdfReader(BytesIO(content))
                if pdf.is_encrypted or len(pdf.pages) > 50:
                    raise DocumentError("Use an unencrypted PDF with at most 50 pages.")
            except DocumentError:
                raise
            except Exception:
                raise DocumentError("Invalid PDF file.") from None
        try:
            result = self.client.run("parse", name, content)
            self._install_pages(name, digest, result["pages"])
            self._jobs[name] = result["folder"]
            self._rag_queries = {key: value for key, value in self._rag_queries.items() if key[0] != name}
            self.backend_status[name] = {
                "parser": result.get("parser", "MinerU"),
                "retrieval": "parsed_only",
                "message": "Structured extraction completed.",
            }
            try:
                indexed = self.client.run("index", result["folder"])
                self.backend_status[name].update(
                    retrieval="RAG-Anything",
                    message=indexed.get("message", "Graph and vector index ready."),
                    embedding_profile=indexed.get('embedding_profile'),
                )
            except RagBackendError as error:
                self.backend_status[name]["message"] = (
                    "MinerU text is available; document indexing unavailable: "
                    + str(error)
                )
            return {
                "document": name,
                "status": "indexed",
                "backend": self.backend_status[name],
                "chunks": len(self._documents[name][1]),
            }
        except RagBackendError as error:
            self.backend_status[name] = {
                "parser": "local_fallback",
                "retrieval": "BM25",
                "message": str(error),
            }
            if name.lower().endswith(".pdf"):
                return DocumentIndex.add(self, name, content)
            raise

    def _install_pages(self, name, digest, pages):
        chunks = []
        for page in pages:
            number = page.get("page")
            if not isinstance(number, int) or number < 1:
                raise RagBackendError("Parser returned an invalid page reference.")
            text = " ".join(str(page.get("text", "")).split())
            for offset in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP):
                chunks.append(
                    Chunk(
                        name,
                        number,
                        f"{digest[:12]}-mineru-{number}-{offset}",
                        text[offset : offset + CHUNK_SIZE],
                    )
                )
                if offset + CHUNK_SIZE >= len(text):
                    break
        if not chunks:
            raise RagBackendError("MinerU returned no readable evidence.")
        self._documents[name] = (digest, chunks)

    def add_table_context(self, name, frame):
        """Index a small table preview for context; calculate totals from the full table."""
        preview = {
            "columns": list(frame.columns),
            "rows": len(frame),
            "preview": json.loads(frame.head(10).to_json(orient="records")),
            "coverage": "First 10 rows only. Exact calculations must use calculate_table over the full loaded table.",
        }
        payload = json.dumps(preview, ensure_ascii=False).encode()
        if len(self._documents) >= MAX_DOCUMENTS and name not in self._documents:
            self.backend_status[name] = {
                "parser": "pandas",
                "retrieval": "local_only",
                "message": "Knowledge-document limit reached; local table calculations remain available.",
            }
            return
        try:
            result = self.client.run("insert", name, payload)
            self._install_pages(
                name,
                hashlib.sha256(payload).hexdigest(),
                [{"page": 1, "text": json.dumps(preview, ensure_ascii=False)}],
            )
            self._jobs[name] = result["folder"]
            self.backend_status[name] = {
                "parser": "pandas",
                "retrieval": "RAG-Anything",
                "message": "Table metadata and first 10 rows indexed; full-table arithmetic remains local.",
            }
        except RagBackendError as error:
            self.backend_status[name] = {
                "parser": "pandas",
                "retrieval": "local_only",
                "message": str(error),
            }

    def retry_incomplete_indexes(self, rebuild_all=False):
        for name, folder in list(self._jobs.items()):
            if not rebuild_all and self.backend_status.get(name, {}).get("retrieval") != "parsed_only":
                continue
            try:
                indexed = self.client.run("index", folder)
                self._rag_queries.clear()
                self._query_cooldowns = {}
                self.backend_status[name].update(
                    retrieval="RAG-Anything", message=indexed.get('message', 'Document index ready.'),
                    embedding_profile=indexed.get('embedding_profile')
                )
            except RagBackendError as error:
                self.backend_status[name].update(retrieval='parsed_only', message=str(error))

    def search(self, query, document="", page=""):
        """Search the requested page, or add graph/vector context for a general question."""
        local = super().search(query, document, page)
        local['retrieval_info'] = {
            'backend': 'anything_rag',
            'hybrid_used': False,
            'reason': 'index_not_ready',
            'indexed_documents': [name for name, state in self.backend_status.items() if state.get('retrieval') == 'RAG-Anything'],
        }
        # For a page-specific question, search only that page.
        if page or requested_page(query) is not None:
            local['retrieval_info']['reason'] = 'exact_page'
            return local
        # Treat a whitespace/case variation as the same uploaded filename.
        if document:
            matches = [name for name in self._documents if name.strip().casefold() == document.strip().casefold()]
            if matches:
                document = matches[0]
                local = super().search(query, document, page) | {'retrieval_info': local['retrieval_info']}
        names = [document] if document else list(self._documents)
        ready = [
            name
            for name in names
            if self.backend_status.get(name, {}).get("retrieval") == "RAG-Anything"
        ][:2]
        context = []
        if document and document not in self._documents:
            local['retrieval_info']['reason'] = 'document_not_found'
        elif not ready and local['retrieval_info']['indexed_documents']:
            local['retrieval_info']['reason'] = 'selected_document_not_indexed'
        if ready:
            local['retrieval_info']['reason'] = 'no_hybrid_context'
        for name in ready:
            cooldowns = getattr(self, '_query_cooldowns', {})
            if cooldowns.get(name, 0) > time.monotonic():
                local['retrieval_info']['reason'] = 'query_cooldown'
                continue
            key = (name, query, profile_key())
            try:
                if key not in self._rag_queries:
                    self._rag_queries[key] = self.client.run("query", self._jobs[name], query=query)
                retrieved = self._rag_queries[key]
                if isinstance(retrieved,str):
                    retrieved = {'context':retrieved}
                if retrieved.get('context'):
                    context.append({'document':name,'context':retrieved['context'][:6000]})
            except RagBackendError:
                if not hasattr(self, '_query_cooldowns'):
                    self._query_cooldowns = {}
                self._query_cooldowns[name] = time.monotonic() + 60
                local['retrieval_info']['reason'] = 'query_failed'
                local["retrieval_notice"] = (
                    "Document retrieval unavailable; local source passages returned."
                )
        if context:
            local['retrieval_info'].update(hybrid_used=True, reason='hybrid_context_returned')
            local["method"] = "RAG-Anything hybrid + source passages"
            local["multimodal_context"] = context
            local["coverage"] = (
                "Graph/vector context augments actual source passages; it is not a verified spreadsheet calculation."
            )
        return local

    def clear(self):
        super().clear()
        self._jobs.clear()
        self.backend_status.clear()
        self._rag_queries.clear()
        self._query_cooldowns = {}
        # Start a new scope so this chat cannot reuse its cleared index by accident.
        self.client = RagWorkerClient()


def create_document_index():
    """Choose Anything RAG or the standard document index from the local setting."""
    return (
        AnythingRAGIndex()
        if os.getenv("DOCUMENT_RAG_BACKEND", "raganything").lower() == "raganything"
        else DocumentIndex()
    )
