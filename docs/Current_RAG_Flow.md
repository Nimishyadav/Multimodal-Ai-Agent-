# Current project flow

## Document questions

Upload → MinerU Cloud → structured text/tables/images → official RAGAnything
insert_content_list → LightRAG graph and vectors → aquery hybrid retrieval →
Document Agent → short answer with source references.

The RAG-Anything library is the real installed upstream package. Cloud parser output
enters its supported content-list API. Explicit local parsing can instead use the
upstream process_document_complete API. Text, image, table and equation processing
are enabled. Knowledge graph use is internal and approved; PyVis is not needed.
Ollama is not configured. Configured cloud callbacks supply the required model roles.

## Questions outside documents

Question → Orchestrator → Internet Agent → actual web search → retrieved snippets
and source links → answer. Uploaded files do not force every question into document
retrieval. A document miss can lead to a clearly labeled internet follow-up. If a
request says only the document, missing evidence is reported without internet fallback.
A URL uses the webpage reader; calculations and memory use their own specialists.

## Models

Text and graph extraction: Groq gpt-oss-20b with configured Gemini/OpenRouter backups.
Vision: existing compatible cloud vision chain. Embeddings: Gemini embedding 001,
768 dimensions. Parsing: MinerU Cloud. Storage, orchestration and Pandas remain local.
An extractive fallback is labeled; it is not falsely described as hybrid success.

## Main files

- app/ui/streamlit_app.py: one chat and cloud consent controls.
- app/ui/runtime.py: refreshes stale presentation imports after code reloads.
- app/multi_agent/requests.py: source-scope and external-question rules.
- app/multi_agent/system.py: orchestrator, document/internet agents and source separation.
- app/rag/anything_rag.py: worker adapter, cached context and source locations.
- app/rag/anything_rag_worker.py: official ingestion and hybrid query operations.
- app/rag/anything_rag_models.py: RAGAnything configuration/model callbacks.
- app/rag/mineru_cloud.py and embeddings.py: approved cloud requests.
- app/history/store.py: saves chat answers and activity in SQLite.

Default settings are raganything backend and hybrid retrieval, with approved cloud
inference. Sidebar cloud consent can be unchecked per chat. For old FAISS/local indexes,
use Document processing status → Rebuild document indexes once after restarting.
The old vector-only documentation describes the previous architecture.


## Verification

The final regression suite ran 163 tests: 161 passed, 2 optional fixture tests skipped.
A synthetic live document passed MinerU Cloud parsing, official RAG-Anything indexing,
and hybrid retrieval; a LightRAG graph artifact was created. A separate live public
web search returned 3 sources. Streamlit startup check produced zero exceptions and
confirmed the simple_activity helper import. These checks do not guarantee future API
availability or unlimited speed/quota.

## Files changed in this update

- app/ui/runtime.py and streamlit_app.py: cached-import compatibility and source/consent UI.
- app/multi_agent/requests.py and system.py: document scope, Internet Agent, tool forcing,
  evidence-miss follow-up and distinct source labels.
- app/rag/embeddings.py, anything_rag.py and anything_rag_worker.py: official hybrid defaults,
  cloud embedding profile, approved cloud client and native ingestion dispatch.
- .env: only the non-secret backend/mode/approval settings were updated; keys were not printed.
- .env.example: portable hybrid/cloud configuration with consent disabled for new users.
- examples/anything_rag_example.py: official content ingestion and direct hybrid query.
- requirements-rag.txt: official multimodal runtime dependencies.
- README.md and this guide: current architecture and setup.
- tests/test_source_routing.py plus explicit local fixtures in test_large_documents.py and
  test_official_rag_flow.py: routing, scope and migration/import regression coverage.


## RAG cleanup
The obsolete FAISS-only adapter, its tests, optional requirements and obsolete guide
were removed. Official RAG-Anything is the only multimodal indexing path. Agent
orchestration, tools, model fallbacks and source-routing behavior were preserved.
