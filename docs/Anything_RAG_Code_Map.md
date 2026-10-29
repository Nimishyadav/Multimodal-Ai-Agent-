# Anything RAG code map

The main project integration is named anything_rag.py. The official third-party framework remains imported as raganything; it is not a model/API name.

## Read these three files in order

1. app/rag/anything_rag.py: application-facing adapter. create_document_index selects the configured backend. AnythingRAGIndex.add_file parses a file, retains page evidence and attempts indexing. search keeps exact-page requests strict and augments general questions with hybrid context. RagWorkerClient runs the isolated optional process. Business rules and UI-facing status live here.
2. app/rag/anything_rag_worker.py: runtime dispatcher. execute handles parse, index, table-preview insert and query. pages_from_blocks preserves page references. validate_media_paths confines model-readable images to the current job. main returns sanitized JSON errors instead of credentials or raw provider bodies.
3. app/rag/anything_rag_models.py: actual model/API wiring. build_engine constructs official RAGAnything and RAGAnythingConfig. text_model_callback reuses create_model; vision_model_callback reuses describe_image; embedding_callback runs FastEmbed locally. These callbacks are the functions RAG-Anything calls when indexing/querying.

## Models and APIs

| Task | Model / component | Where it runs |
| --- | --- | --- |
| Text extraction from document layout | MinerU pipeline | Local CPU by current setting; no MinerU cloud key |
| Graph text interpretation / keyword and entity extraction | openai/gpt-oss-20b primary | Groq API, through app/llm/client.py |
| Text backups | gemini-3.1-flash-lite, openrouter/free | Google / OpenRouter APIs, compatible configured fallback |
| Figure interpretation | qwen/qwen3.8-27b primary | Groq vision, through app/attachments/vision.py |
| Vision backups | google/gemma-4-26b-a4b-it:free, compatible Gemini | OpenRouter / Google APIs |
| Embeddings | BAAI/bge-small-en-v1.5, 384 dimensions | Local FastEmbed/ONNX; no embedding API key |
| Retrieval framework | RAG-Anything + LightRAG | Local graph/vector storage and retrieval; model callbacks still consume API quota |

These are verified configuration values, not promises about perpetual free quotas or model availability. No new paid API service was added. The parser and embedding models may download public weights on first use.

## Simple explanation

anything_rag.py sochta nahi, application integration aur safety rules handle karta hai. anything_rag_worker.py heavy runtime ke andar actual jobs chalata hai. anything_rag_models.py batata hai ki framework text ke liye kaunsa API, figure ke liye kaunsa API aur embeddings ke liye kaunsa local model call karega. Orchestrator ab bhi LangGraph mein hai; exact Excel maths ab bhi Pandas mein hai.

## What is implemented

Real MinerU parsing, official RAG-Anything insertion, local graph/vector artifacts, hybrid retrieval context, existing page/source evidence, fallback and incomplete-index retry. Text/table/figure PDF processing and retrieval have been live-verified. Office conversion may need LibreOffice; the local text fallback is not full multimodal Office parsing. All upstream optional features, arbitrary file formats, audio/video and unlimited uninterrupted operation are not claimed.
