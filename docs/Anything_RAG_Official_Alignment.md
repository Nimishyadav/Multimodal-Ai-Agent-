# RAG-Anything implementation and upstream mapping

Reference: https://github.com/HKUDS/RAG-Anything
Example: https://github.com/HKUDS/RAG-Anything/blob/main/examples/raganything_example.py

The installed upstream `raganything==1.4.3` package implements the algorithms.
This app imports it; it does not maintain a replacement RAG algorithm.
GitHub main may evolve beyond the pinned package. This project is not a byte-for-byte
clone of the upstream repository and does not claim to enable every upstream example.

## Official API flow

1. `RAGAnythingConfig`: MinerU parser; image, table and equation processing enabled.
2. `RAGAnything`: receives text, vision and embedding callbacks.
3. Local parsing uses `rag.parse_document()` to retain page evidence for the UI.
4. Local indexing uses `rag.process_document_complete()` (official end-to-end pipeline).
   Its parser cache can reuse the earlier parse. Legacy indexes remain usable.
5. Approved cloud output uses official `rag.insert_content_list()` instead of local parsing.
6. Retrieval uses `rag.aquery(..., mode="hybrid")`.
7. `examples/anything_rag_example.py` demonstrates direct ingestion and generated answers,
   including upstream's automatic VLM-enhanced query behavior.

## Components required by the framework

MinerU is the upstream-supported parser. Its layout/OCR/table/formula models are internal
parser dependencies, not new agents or a second RAG framework. LightRAG constructs and
queries the graph/vector stores. A text LLM, vision LLM and embedding function are required.

The upstream example defaults to OpenAI gpt-4o-mini, gpt-4o and text-embedding-3-large.
Those are paid API choices, not mandatory RAG-Anything models. This app preserves existing
configured text/vision providers and local BAAI/bge-small-en-v1.5 embeddings to respect the
free-provider preference. It is therefore the official library workflow with different
model callbacks, not the identical OpenAI deployment.

## Application adapters

Streamlit, LangGraph delegation, SQLite chat/attachment persistence, CPU settings,
call budgets, API fallbacks, exact-page source checks and Pandas arithmetic are app-level
integration. They are not presented as upstream features. The unified chat retrieves
context with `only_need_context=True`, then its Document Agent composes the answer;
the standalone example lets RAG-Anything generate the answer itself. Exact-page queries
use page lookup so unrelated pages do not replace the requested evidence.

Cloud parsing remains explicitly opt-in. Cached parse/index results are reused. Local
source lookup is honestly labeled as fallback if indexing/querying cannot finish.
Tables indexed from the spreadsheet route contain a bounded preview, not all rows;
verified full-table totals remain Pandas operations.

## Windows command (from the project directory)

```powershell
.\.venv-rag\Scripts\python.exe examples/anything_rag_example.py "document.pdf" --question "Summarize this document briefly."
```

The example uses local MinerU; extracted text/figure context is sent to configured LLM
providers for indexing and answering. Existing upload limits still apply in Streamlit.
