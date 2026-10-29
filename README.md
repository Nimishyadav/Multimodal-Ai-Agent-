# Multimodal AI Agent

A unified AI assistant built with Python, LangChain, and LangGraph.

The application coordinates specialist agents for document questions,
spreadsheet analysis, image understanding, web research, public webpage
reading, and travel searches.

## Features

- LangGraph orchestrator and specialist agents
- Official RAG-Anything and LightRAG document retrieval
- MinerU Cloud document parsing
- Gemini API embeddings
- Verified spreadsheet calculations with Pandas
- Image understanding through compatible vision APIs
- Web search and public webpage reading
- Persistent chat history and profile memory
- Compatible model-provider fallback

## Architecture

User → Streamlit → Orchestrator → Specialist Agent → Tool/API
     → Evidence → Answer → SQLite storage

Document flow:
Upload → MinerU Cloud → RAG-Anything → LightRAG index
       → Relevant evidence → Document Agent → Answer

## Technology Stack

Python, Streamlit, LangChain, LangGraph, RAG-Anything,
LightRAG, Pandas, SQLite, BeautifulSoup, and Pillow.

## Configuration

Copy `.env.example` to `.env` and configure the required services.
Never commit actual API keys.

Cloud document processing sends uploaded files to MinerU and
extracted text to the configured embedding service with user consent.

## Limitations

API quotas and availability affect cloud features.
Local retrieval fallbacks may provide reduced coverage.
Travel listings do not guarantee booking availability or final prices.
