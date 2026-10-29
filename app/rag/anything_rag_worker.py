"""Run parsing, indexing and search jobs in the separate Anything RAG environment."""

import asyncio
from collections import defaultdict
import json
import os
from pathlib import Path
import sys

from app.rag.embeddings import cloud_mode, embedding_profile, CloudProcessingError

ROOT = Path(__file__).resolve().parents[2]


def pages_from_blocks(blocks):
    """Convert MinerU page numbers to the one-based numbers shown to the user."""
    pages = defaultdict(list)
    for item in blocks:
        index = item.get("page_idx", 0)
        if not isinstance(index, int) or index < 0:
            continue
        pieces = []
        for key in [
            "text",
            "table_body",
            "latex",
            "equation_caption",
            "image_caption",
            "image_footnote",
            "table_caption",
            "table_footnote",
        ]:
            value = item.get(key)
            if isinstance(value, list):
                value = " ".join(str(v) for v in value)
            if isinstance(value, str) and value.strip():
                pieces.append(value)
        if pieces:
            pages[index + 1].append("\n".join(pieces))
    return [
        {"page": number, "text": "\n".join(pages[number])} for number in sorted(pages)
    ]


def validate_media_paths(blocks, folder):
    """Allow image files only from the current document job."""
    for item in blocks:
        for key in ["img_path", "image_path", "table_img_path"]:
            path = item.get(key)
            if path:
                resolved = Path(path).resolve()
                if (
                    not resolved.is_relative_to(folder.resolve())
                    or not resolved.is_file()
                ):
                    raise ValueError("Parser media path outside document scope.")


async def execute(request):
    """Run the requested parsing, indexing or search step."""
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
    folder = Path(request["folder"]).resolve()
    allowed = (ROOT / "data" / "multimodal_rag").resolve()
    if not folder.is_relative_to(allowed):
        raise ValueError("Invalid RAG storage scope.")
    folder.mkdir(parents=True, exist_ok=True)
    action = request["action"]
    api_inference = cloud_mode() or embedding_profile()['provider']=='gemini'
    if action == "parse":
        os.environ["MINERU_DEVICE_MODE"] = os.getenv("MINERU_DEVICE", "cpu")

        name = Path(request["name"]).name
        source = folder / "source" / ("input" + Path(name).suffix.lower())
        if not source.is_file() or source.stat().st_size > 10 * 1024 * 1024:
            raise ValueError("Invalid parser input.")
        # Reuse parsing results only for this exact source file.
        cache = folder / "content.json"
        parser_used = 'MinerU local'
        parser_cache = folder / 'parser_status.json'
        if cache.exists():
            if parser_cache.exists():
                parser_used = json.loads(parser_cache.read_text(encoding='utf-8')).get('parser', parser_used)
            blocks = json.loads(cache.read_text(encoding="utf-8"))
        else:
            token = os.getenv('MINERU_API_TOKEN', '').strip()
            blocks = None
            if api_inference and (not token or request.get('cloud_upload_approved') is not True):
                raise CloudProcessingError('Cloud parsing needs MINERU_API_TOKEN and the MinerU Cloud upload checkbox. Local AI parsing is disabled in cloud mode.')
            if token and request.get('cloud_upload_approved') is True:
                from app.rag.mineru_cloud import parse_cloud, MinerUCloudError
                try:
                    blocks = await asyncio.to_thread(parse_cloud, source, name, folder, token, upload_approved=True)
                    parser_used = 'MinerU Cloud API'
                except MinerUCloudError as error:
                    if api_inference:
                        raise CloudProcessingError(str(error) + " No local AI parser was started.") from None
                    parser_used = 'MinerU local (cloud unavailable)'
            if blocks is None:
                from app.rag.anything_rag_models import build_engine
                rag = await build_engine(folder)
                try:
                    initialized = await rag._ensure_lightrag_initialized()
                    if not initialized.get('success'):
                        raise RuntimeError('RAG-Anything storage initialization failed.')
                    # RAG-Anything owns parsing and its cache, as in the upstream pipeline.
                    blocks, _ = await rag.parse_document(
                        file_path=str(source), output_dir=str(folder / 'parsed'),
                        parse_method='auto', display_stats=False,
                        backend='pipeline', timeout=150,
                    )
                finally:
                    if rag.lightrag is not None:
                        await rag.lightrag.finalize_storages()
            if not isinstance(blocks, list) or len(blocks) > 5000:
                raise ValueError("Parser output exceeds block limit.")
            validate_media_paths(blocks, folder)
            cache.write_text(json.dumps(blocks, ensure_ascii=False), encoding="utf-8")
            parser_cache.write_text(json.dumps({"parser": parser_used}), encoding="utf-8")
        pages = pages_from_blocks(blocks)
        (folder / "metadata.json").write_text(
            json.dumps({"name": name}), encoding="utf-8"
        )
        return {
            "ok": True,
            "pages": pages,
            "blocks": len(blocks),
            "message": "MinerU structured extraction completed.",
            "parser": parser_used,
        }
    if api_inference and request.get('cloud_upload_approved') is not True:
        raise CloudProcessingError('Cloud indexing and query embeddings require approval to send extracted text to Google. Enable the cloud consent checkbox.')
    metadata = folder / "metadata.json"
    if action == "insert":
        name = Path(request["name"]).name
        payload = json.loads(
            (folder / "source" / ("input" + Path(name).suffix.lower())).read_text(
                encoding="utf-8"
            )
        )
        blocks = [
            {
                "type": "table",
                "table_body": json.dumps(payload, ensure_ascii=False),
                "table_caption": [
                    "Bounded preview only; never a full-table calculation."
                ],
                "page_idx": 0,
            }
        ]
        (folder / "content.json").write_text(
            json.dumps(blocks, ensure_ascii=False), encoding="utf-8"
        )
        metadata.write_text(
            json.dumps({"name": name, "coverage": "First 10 rows only"}),
            encoding="utf-8",
        )
    if not metadata.exists():
        raise ValueError("Document metadata unavailable.")
    name = json.loads(metadata.read_text(encoding="utf-8"))["name"]
    from app.rag.anything_rag_models import build_engine

    profile = embedding_profile()
    marker = folder / 'index_ready.json'
    if action == 'query' and marker.exists():
        stored = json.loads(marker.read_text(encoding='utf-8'))
        stored_profile = stored.get('embedding_profile', {'provider':'local','model':'BAAI/bge-small-en-v1.5','dimensions':384})
        if stored_profile != profile:
            raise CloudProcessingError('This document uses an older embedding index. Open Document processing status and rebuild indexes for the current processing mode.')
    rag = await build_engine(folder)
    try:
        if action in {"index", "insert"}:
            if marker.exists():
                stored = json.loads(marker.read_text(encoding='utf-8'))
                stored_profile = stored.get('embedding_profile', {'provider':'local','model':'BAAI/bge-small-en-v1.5','dimensions':384})
                if stored_profile == profile:
                    return {"ok": True, "message": "Reused completed multimodal index.", 'embedding_profile':profile}
            blocks = json.loads((folder / "content.json").read_text(encoding="utf-8"))
            validate_media_paths(blocks, folder)
            parser_status = folder / 'parser_status.json'
            parser_used = json.loads(parser_status.read_text(encoding='utf-8')).get('parser', '') if parser_status.exists() else ''
            if action == 'index' and parser_used.startswith('MinerU local') and not api_inference:
                source = folder / 'source' / ('input' + Path(name).suffix.lower())
                # Official end-to-end API: parse cache → text/multimodal processing → LightRAG.
                await rag.process_document_complete(
                    file_path=str(source), output_dir=str(folder / 'parsed'),
                    parse_method='auto', display_stats=False,
                    file_name=name, backend='pipeline', timeout=150,
                )
            else:
                # Official content-list API for approved cloud output and table previews.
                await rag.insert_content_list(blocks, file_path=name, display_stats=False)
            (folder / "index_ready.json").write_text(
                json.dumps({"ready": True, "embedding": profile["model"], "embedding_profile":profile}),
                encoding="utf-8",
            )
            return {
                "ok": True,
                "message": "RAG-Anything graph/vector indexing completed.",
                "embedding_profile": profile,
            }
        if action == "query":
            if not (folder / "index_ready.json").exists():
                raise ValueError("No completed RAG index.")
            initialized = await rag._ensure_lightrag_initialized()
            if not initialized.get("success"):
                raise RuntimeError("Storage initialization failed.")
            context = await rag.aquery(
                request["query"][:500],
                mode="hybrid",
                only_need_context=True,
                vlm_enhanced=False,
                enable_rerank=False,
                top_k=5,
                chunk_top_k=3,
                max_total_tokens=1800,
            )
            return {"ok": True, "context": str(context)[:6000] if context else ""}
        raise ValueError("Unknown RAG action.")
    finally:
        if rag.lightrag is not None:
            await rag.lightrag.finalize_storages()


def main():
    request_path, response_path = map(Path, sys.argv[1:3])
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
        result = asyncio.run(execute(request))
    except Exception as error:
        # Return a short error message; raw errors may contain private request details.
        kind = type(error).__name__
        result = {
            "ok": False,
            "message": (
                str(error) if isinstance(error, CloudProcessingError) else "MinerU/RAG-Anything could not finish (" + kind + "). "
                "Check optional packages, parser/model availability, LibreOffice for Office conversion, or provider quotas."
            ),
        }
    response_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
