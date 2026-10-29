import json
import unittest
from langchain_core.messages import AIMessage
from app.rag.documents import DocumentIndex
from app.rag.anything_rag import AnythingRAGIndex
from app.rag.retrieval_status import document_retrieval_activity
from app.tools.document_search import create_document_tool
from app.multi_agent.system import handle_multi_agent
from test_multi_agent import Model,Conversation,call
from test_workflows import FakeSearch
from test_chat_router import PriceStub

class Client:
 def run(self,*args,**kwargs):return {'context':'Tax relates to the recorded sale.'}

class RetrievalActivityTests(unittest.TestCase):
 def index(self,ready):
  index=AnythingRAGIndex(Client())
  index._install_pages('notes.pdf','digest',[{'page':1,'text':'The tax amount is 25.'}])
  index.backend_status['notes.pdf']={'retrieval':'RAG-Anything' if ready else 'parsed_only'}
  index._jobs['notes.pdf']='job'
  return index
 def test_hybrid_use_is_reported_only_with_context(self):
  result=self.index(True).search('tax')
  self.assertTrue(result['retrieval_info']['hybrid_used'])
  self.assertIn('Anything RAG hybrid',document_retrieval_activity(result))
 def test_incomplete_index_is_reported_as_fallback(self):
  result=self.index(False).search('tax')
  self.assertFalse(result['retrieval_info']['hybrid_used'])
  self.assertIn('not ready',document_retrieval_activity(result))
  self.assertNotIn('Anything RAG hybrid',document_retrieval_activity(result))
 def test_page_lookup_is_reported_as_page_lookup(self):
  result=self.index(True).search('tax','notes.pdf','1')
  self.assertEqual(result['retrieval_info']['reason'],'exact_page')
  self.assertIn('hybrid search skipped',document_retrieval_activity(result))
 def test_filename_case_and_space_use_existing_hybrid_index(self):
  result=self.index(True).search('summary', ' NOTES.PDF ')
  self.assertTrue(result['retrieval_info']['hybrid_used'])
  self.assertTrue(result['passages'])
  self.assertIn('Document Agent → search_documents → RAG-Anything',document_retrieval_activity(result))
 def test_unknown_filename_does_not_claim_index_failure(self):
  result=self.index(True).search('summary', 'missing.pdf')
  self.assertEqual(result['retrieval_info']['reason'],'document_not_found')
  self.assertFalse(result['retrieval_info']['hybrid_used'])
  self.assertIn('does not match',document_retrieval_activity(result))
 def test_standard_tool_labels_standard_backend(self):
  index=DocumentIndex();index.add_sections('notes.txt',['Tax is 25.'])
  data=json.loads(create_document_tool(index).invoke({'query':'tax'}))
  self.assertEqual(data['retrieval_info']['backend'],'standard')
 def test_real_workflow_records_retrieval_detail(self):
  index=self.index(True);conv=Conversation();conv.document_index=index;conv.tools=[create_document_tool(index)]
  model=Model(call('delegate',{'agent':'documents','task':'Explain PDF tax'},'d'),
              call('search_documents',{'query':'tax'},'s'),AIMessage(content='The tax amount is 25.'))
  handle_multi_agent('Explain the PDF tax amount',conv,model,FakeSearch(),PriceStub(),economy_mode=True)
  self.assertTrue(any('Anything RAG hybrid' in event for event in conv.records[0][2]))

if __name__=='__main__':unittest.main()
