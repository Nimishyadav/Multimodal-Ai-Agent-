"""Set up the text model, vision model and local embeddings used by Anything RAG."""

import asyncio
import base64
from io import BytesIO
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


async def build_engine(folder):
    """Create the RAG-Anything engine with our text, vision and embedding functions."""
    from raganything import RAGAnything, RAGAnythingConfig
    from lightrag.utils import EmbeddingFunc
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
    from app.config.settings import load_settings
    from app.llm.client import create_model
    from app.attachments.vision import describe_image
    import numpy as np
    from app.rag.embeddings import embedding_profile, cloud_embeddings
    profile = embedding_profile()

    text_model = create_model(load_settings())
    remaining = [int(os.getenv("RAG_MAX_MODEL_CALLS", "16"))]

    def consume():
        remaining[0] -= 1
        if remaining[0] < 0:
            raise RuntimeError("RAG model-call budget reached.")

    # Reuse the normal text model and its configured backups.
    async def text_model_callback(
        prompt, system_prompt=None, history_messages=None, **kwargs
    ):
        consume()
        messages = [SystemMessage(content=system_prompt)] if system_prompt else []
        for item in (history_messages or [])[-4:]:
            cls = AIMessage if item.get("role") == "assistant" else HumanMessage
            messages.append(cls(content=item.get("content", "")))
        messages.append(HumanMessage(content=prompt))
        response = await asyncio.to_thread(
            text_model.invoke_with_budget, messages, consume
        )
        return response.content

    # Reuse the vision model setup for document figures.
    async def vision_model_callback(
        prompt,
        system_prompt=None,
        history_messages=None,
        image_data=None,
        messages=None,
        **kwargs
    ):
        if messages:
            from app.attachments.vision import create_vision_model
            consume()
            vision = create_vision_model()
            converted = []
            for item in messages:
                if not item:
                    continue
                cls = {'system': SystemMessage, 'assistant': AIMessage}.get(item.get('role'), HumanMessage)
                converted.append(cls(content=item.get('content', '')))
            response = await asyncio.to_thread(vision.invoke_with_budget, converted, consume)
            return response.content
        if not image_data:
            return await text_model_callback(prompt, system_prompt, history_messages)
        consume()
        from PIL import Image

        raw = base64.b64decode(image_data, validate=True)
        with Image.open(BytesIO(raw)) as image:
            mime = Image.MIME.get(image.format, "image/jpeg")
        return await asyncio.to_thread(
            describe_image,
            "data:" + mime + ";base64," + image_data,
            (system_prompt or "") + "\n" + prompt,
            extra_attempt=consume,
        )

    # Cloud profiles never import or load the local embedding model.
    embedding = [None]

    async def embedding_callback(texts, **kwargs):
        if profile['provider'] == 'gemini':
            return await asyncio.to_thread(cloud_embeddings, list(texts))
        def encode():
            from fastembed import TextEmbedding
            if embedding[0] is None:
                embedding[0] = TextEmbedding(
                    model_name="BAAI/bge-small-en-v1.5",
                    cache_dir=str(ROOT / "data" / "rag_models"),
                    threads=2,
                )
            return np.asarray(list(embedding[0].embed(texts)), dtype=np.float32)

        return await asyncio.to_thread(encode)

    # RAG-Anything uses these callbacks; LightRAG builds the graph and vector index.
    config = RAGAnythingConfig(
        working_dir=str(folder / ("index_cloud_gemini_768" if profile["provider"] == "gemini" else "index")),
        parser="mineru",
        parse_method="auto",
        parser_output_dir=str(folder / "parsed"),
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
        display_content_stats=False,
        max_concurrent_files=1,
    )
    rag = RAGAnything(
        config=config,
        llm_model_func=text_model_callback,
        vision_model_func=vision_model_callback,
        embedding_func=EmbeddingFunc(
            embedding_dim=profile["dimensions"], max_token_size=512, func=embedding_callback
        ),
        lightrag_kwargs={
            "llm_model_max_async": 1,
            "embedding_func_max_async": 1,
            "max_parallel_insert": 1,
            "chunk_token_size": 400,
            "chunk_overlap_token_size": 40,
            "entity_extract_max_gleaning": 0,
            "embedding_batch_num": 8,
            "enable_llm_cache": True,
        },
    )
    return rag
