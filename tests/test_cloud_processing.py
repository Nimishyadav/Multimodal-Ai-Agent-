import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
import numpy as np
from app.rag.embeddings import cloud_embeddings, embedding_profile, CloudProcessingError
from app.rag import anything_rag_worker as worker
from app.rag.mineru_cloud import MinerUCloudError


class CloudProcessingTests(unittest.TestCase):
    def test_cloud_profile_never_selects_local_embedding(self):
        with patch.dict('os.environ',{'RAG_PROCESSING_MODE':'cloud','RAG_EMBEDDING_PROVIDER':'gemini'}):
            self.assertEqual(embedding_profile()['dimensions'],768)
        with patch.dict('os.environ',{'RAG_PROCESSING_MODE':'cloud','RAG_EMBEDDING_PROVIDER':'local'}):
            with self.assertRaises(CloudProcessingError):embedding_profile()

    def test_cloud_vectors_are_normalized_and_validate_shape(self):
        reply=Mock();reply.json.return_value={'embeddings':[{'values':[1.0]*768}]}
        with patch('app.config.settings.configured_key',return_value='test-key'),patch('app.rag.embeddings.requests.post',return_value=reply) as post:
            values=cloud_embeddings(['synthetic fixture'])
            self.assertEqual(values.shape,(1,768))
            self.assertAlmostEqual(float(np.linalg.norm(values[0])),1.0,places=5)
            self.assertEqual(post.call_args.kwargs['timeout'],20)
            self.assertEqual(post.call_args.kwargs['json']['requests'][0]['outputDimensionality'],768)
        reply.json.return_value={'embeddings':[{'values':[1.0]*384}]}
        with patch('app.config.settings.configured_key',return_value='test-key'),patch('app.rag.embeddings.requests.post',return_value=reply):
            with self.assertRaises(CloudProcessingError):cloud_embeddings(['synthetic fixture'])

    def test_cloud_failure_does_not_start_local_parser(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);folder=root/'data'/'multimodal_rag'/'job'
            (folder/'source').mkdir(parents=True);(folder/'source'/'input.pdf').write_bytes(b'fixture')
            with patch.object(worker,'ROOT',root),patch.dict('os.environ',{'RAG_PROCESSING_MODE':'cloud','RAG_EMBEDDING_PROVIDER':'gemini','MINERU_API_TOKEN':'test-token'}),patch('app.rag.mineru_cloud.parse_cloud',side_effect=MinerUCloudError('failed')),patch('app.rag.anything_rag_models.build_engine',new_callable=AsyncMock) as local:
                with self.assertRaises(CloudProcessingError):
                    asyncio.run(worker.execute({'action':'parse','folder':str(folder),'name':'fixture.pdf','cloud_upload_approved':True}))
                local.assert_not_awaited()

    def test_cloud_indexing_requires_explicit_consent(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);folder=root/'data'/'multimodal_rag'/'job'
            with patch.object(worker,'ROOT',root),patch.dict('os.environ',{'RAG_PROCESSING_MODE':'cloud','RAG_EMBEDDING_PROVIDER':'gemini'}),patch('app.rag.anything_rag_models.build_engine',new_callable=AsyncMock) as build:
                with self.assertRaises(CloudProcessingError):
                    asyncio.run(worker.execute({'action':'index','folder':str(folder)}))
                build.assert_not_awaited()

    def test_local_index_must_be_rebuilt_for_cloud_vectors(self):
        engine=AsyncMock();engine.lightrag=None
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);folder=root/'data'/'multimodal_rag'/'job';folder.mkdir(parents=True)
            (folder/'metadata.json').write_text(json.dumps({'name':'fixture.pdf'}))
            (folder/'content.json').write_text(json.dumps([{'type':'text','text':'Synthetic guide','page_idx':0}]))
            (folder/'parser_status.json').write_text(json.dumps({'parser':'MinerU local'}))
            marker=folder/'index_ready.json';marker.write_text(json.dumps({'ready':True,'embedding':'BAAI/bge-small-en-v1.5'}))
            with patch.object(worker,'ROOT',root),patch.dict('os.environ',{'RAG_PROCESSING_MODE':'cloud','RAG_EMBEDDING_PROVIDER':'gemini'}),patch('app.rag.anything_rag_models.build_engine',new=AsyncMock(return_value=engine)):
                with self.assertRaises(CloudProcessingError):asyncio.run(worker.execute({'action':'query','folder':str(folder),'query':'summary','cloud_upload_approved':True}))
                result=asyncio.run(worker.execute({'action':'index','folder':str(folder),'cloud_upload_approved':True}))
            self.assertEqual(result['embedding_profile']['dimensions'],768)
            engine.insert_content_list.assert_awaited_once()
            engine.process_document_complete.assert_not_awaited()
            self.assertEqual(json.loads(marker.read_text())['embedding_profile']['provider'],'gemini')
