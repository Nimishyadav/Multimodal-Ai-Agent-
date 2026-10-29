"""Official RAG-Anything ingestion/query with approved cloud parsing and model callbacks."""
import argparse
import asyncio
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


async def run(file,question):
    from app.config.settings import load_settings
    from app.rag.anything_rag import RagWorkerClient
    from app.rag.anything_rag_models import build_engine
    load_settings()
    client=RagWorkerClient();client.cloud_upload_approved=True
    path=Path(file)
    parsed=await asyncio.to_thread(client.run,'parse',path.name,path.read_bytes())
    await asyncio.to_thread(client.run,'index',parsed['folder'])
    rag=await build_engine(Path(parsed['folder']))
    try:
        initialized=await rag._ensure_lightrag_initialized()
        if not initialized.get('success'):raise RuntimeError('RAG storage unavailable.')
        print(await rag.aquery(question,mode='hybrid'))
    finally:
        if rag.lightrag is not None:await rag.lightrag.finalize_storages()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Official RAG-Anything content ingestion and hybrid query. No Ollama.')
    parser.add_argument('file');parser.add_argument('--question',default='What is this document about?')
    parser.add_argument('--allow-cloud',action='store_true',help='Approve file upload to MinerU and extracted text/images to configured AI providers.')
    args=parser.parse_args()
    if not args.allow_cloud:parser.error('Add --allow-cloud only for a document approved for these cloud services.')
    asyncio.run(run(args.file,args.question))
