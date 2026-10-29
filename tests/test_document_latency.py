import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock
from langchain_core.messages import AIMessage
from app.rag import anything_rag as backend
from app.rag.anything_rag import AnythingRAGIndex, RagBackendError
from test_multi_agent import Model, call
import test_economy_workflows as helpers


class LatencyTests(unittest.TestCase):
    def test_query_cache_avoids_worker_and_invalidates_on_reindex(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            python=root/'python.exe';python.write_bytes(b'fixture')
            with patch.object(backend,'ROOT',root), patch.object(backend,'runtime_python',return_value=python), patch.dict(os.environ,{'RAG_QUERY_TIMEOUT_SECONDS':'40'}):
                client=backend.RagWorkerClient('chat')
                job=client.directory/'job';job.mkdir(parents=True)
                marker=job/'index_ready.json';marker.write_text('{"ready":true}')
                def run(args,**kwargs):
                    Path(args[-1]).write_text(json.dumps({'ok':True,'context':'Verified synthetic evidence.'}))
                    return Mock(returncode=0)
                with patch.object(backend.subprocess,'run',side_effect=run) as launch:
                    self.assertIn('context',client.run('query',str(job),query='summary'))
                    second=client.run('query',str(job),query='summary')
                    self.assertTrue(second['cache_hit'])
                    self.assertEqual(launch.call_count,1)
                    self.assertEqual(launch.call_args.kwargs['timeout'],40)
                    stamp=marker.stat().st_mtime_ns
                    os.utime(marker,ns=(stamp+1000000000,stamp+1000000000))
                    client.run('query',str(job),query='summary')
                    self.assertEqual(launch.call_count,2)

    def test_failed_hybrid_query_does_not_repeat_immediately(self):
        class Worker:
            def __init__(self):self.calls=0
            def run(self,*args,**kwargs):
                self.calls+=1
                raise RagBackendError('timed out')
        client=Worker();index=AnythingRAGIndex(client)
        index._install_pages('guide.pdf','digest',[{'page':1,'text':'Python tutorial explains functions.'}])
        index.backend_status['guide.pdf']={'retrieval':'RAG-Anything'}
        index._jobs['guide.pdf']='job'
        first=index.search('summary');second=index.search('summary')
        self.assertEqual(client.calls,1)
        self.assertEqual(second['retrieval_info']['reason'],'query_cooldown')
        self.assertTrue(first['passages']);self.assertTrue(second['passages'])

    def test_summary_forces_final_answer_after_evidence(self):
        class Track(Model):
            def __init__(self,*outputs):super().__init__(*outputs);self.choices=[]
            def bind_tools(self,tools,**kwargs):
                self.choices.append(kwargs.get('tool_choice'))
                return super().bind_tools(tools,**kwargs)
        files=helpers.EconomyTests.files(self)
        model=Track(call('delegate',{'agent':'documents','task':'Summarize PDF'},'h'),
                    call('search_documents',{'query':'summary'},'s'),
                    AIMessage(content='The document is a Python tutorial.'))
        answer,activity,_=helpers.EconomyTests.run_case(self,model,'explain the pdf in short',files)
        self.assertIn('none',model.choices)
        self.assertEqual(activity.count('Document Agent selected tool: search_documents'),1)
        self.assertIn('Python',answer)
        self.assertTrue(any('retrieved in' in event for event in activity))
