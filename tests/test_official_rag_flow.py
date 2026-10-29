import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch
from app.rag import anything_rag_worker as worker


@patch.dict('os.environ', {'RAG_EMBEDDING_PROVIDER':'local','RAG_PROCESSING_MODE':'local'})
class OfficialFlowTests(unittest.TestCase):
    def run_index(self, parser_name):
        engine = AsyncMock()
        engine.lightrag = None
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / 'data' / 'multimodal_rag' / 'chat' / 'job'
            (folder/'source').mkdir(parents=True)
            (folder/'source'/'input.pdf').write_bytes(b'synthetic fixture')
            (folder/'metadata.json').write_text(json.dumps({'name':'guide.pdf'}))
            (folder/'content.json').write_text(json.dumps([{'type':'text','text':'Guide','page_idx':0}]))
            (folder/'parser_status.json').write_text(json.dumps({'parser':parser_name}))
            with patch.object(worker,'ROOT',root), patch('app.rag.anything_rag_models.build_engine', new=AsyncMock(return_value=engine)):
                result = asyncio.run(worker.execute({'action':'index','folder':str(folder)}))
            self.assertTrue(result['ok'])
            self.assertTrue((folder/'index_ready.json').exists())
        return engine

    def test_local_ingestion_uses_upstream_complete_api(self):
        engine = self.run_index('MinerU local')
        engine.process_document_complete.assert_awaited_once()
        engine.insert_content_list.assert_not_awaited()
        self.assertEqual(engine.process_document_complete.call_args.kwargs['backend'], 'pipeline')

    def test_cloud_content_uses_upstream_content_list_api(self):
        engine = self.run_index('MinerU Cloud API')
        engine.insert_content_list.assert_awaited_once()
        engine.process_document_complete.assert_not_awaited()

    def test_failed_complete_does_not_mark_ready(self):
        engine = AsyncMock()
        engine.lightrag = None
        engine.process_document_complete.side_effect = ValueError('incomplete')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);folder=root/'data'/'multimodal_rag'/'chat'/'job'
            folder.mkdir(parents=True)
            for name, data in [('metadata.json',{'name':'guide.pdf'}),('content.json',[]),('parser_status.json',{'parser':'MinerU local'})]:
                (folder/name).write_text(json.dumps(data))
            with patch.object(worker,'ROOT',root),patch('app.rag.anything_rag_models.build_engine',new=AsyncMock(return_value=engine)):
                with self.assertRaises(ValueError):
                    asyncio.run(worker.execute({'action':'index','folder':str(folder)}))
            self.assertFalse((folder/'index_ready.json').exists())
