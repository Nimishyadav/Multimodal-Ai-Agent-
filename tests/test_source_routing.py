import unittest
from unittest.mock import patch
from types import SimpleNamespace
from langchain_core.messages import AIMessage
from app.ui import runtime
from app.rag.documents import DocumentIndex
from app.tools.document_search import create_document_tool
from app.multi_agent.requests import external_factual_question,external_requested,strict_document_scope
from app.multi_agent.system import handle_multi_agent
from test_multi_agent import Model,Conversation,call
from test_workflows import FakeSearch
from test_chat_router import PriceStub


class SourceRoutingTests(unittest.TestCase):
    def conversation(self):
        conv=Conversation();index=DocumentIndex()
        index.add_sections('Python Tutorial.pdf',['Python variables store values. Functions organize Python code.'])
        conv.document_index=index;conv.tools=[create_document_tool(index)]
        return conv

    def test_document_question_uses_retrieval_without_web(self):
        model=Model(call('delegate',{'agent':'documents','task':'Explain Python from PDF'},'d'),
                    call('search_documents',{'query':'Python'},'s'),AIMessage(content='Python variables store values.'))
        search=FakeSearch();conv=self.conversation()
        answer,_=handle_multi_agent('Explain variables from the PDF',conv,model,search,PriceStub(),economy_mode=True)
        self.assertIn('Python',answer);self.assertEqual(search.queries,[])
        self.assertIn('Document Agent selected tool: search_documents',conv.records[0][2])

    def test_unrelated_fact_switches_general_to_internet(self):
        model=Model(call('delegate',{'agent':'general','task':'Explain museum attractions'},'d'),
                    call('search_web',{'query':'museum attractions'},'s'),AIMessage(content='A museum and park are attractions [S1].'),AIMessage(content='Should not replace verified result.'))
        search=FakeSearch();conv=self.conversation()
        answer,_=handle_multi_agent('What are museum attractions?',conv,model,search,PriceStub(),economy_mode=True)
        self.assertEqual(len(search.queries),1)
        self.assertIn('Internet Agent selected tool: search_web',conv.records[0][2])
        self.assertNotIn('Document Agent selected tool: search_documents',conv.records[0][2])
        self.assertIn('https://example.org/source',answer)

    def test_explicit_external_question_does_not_force_document_role(self):
        model=Model(call('delegate',{'agent':'internet','task':'Check current museum information'},'d'),
                    call('search_web',{'query':'current museum information'},'s'),AIMessage(content='Museum and park information [S1].'))
        search=FakeSearch();conv=self.conversation()
        answer,_=handle_multi_agent('Ignore the PDF and search the internet for museum information',conv,model,search,PriceStub(),economy_mode=True)
        self.assertEqual(len(search.queries),1)
        self.assertIn('Museum',answer)

    def test_missing_document_fact_gets_labeled_web_followup(self):
        model=Model(call('delegate',{'agent':'documents','task':'Does PDF discuss museums?'},'d'),
                    call('search_documents',{'query':'summary'},'s'),AIMessage(content='DOCUMENT_NOT_FOUND'),
                    call('search_web',{'query':'museum attractions'},'w'),AIMessage(content='A museum and park are attractions [S2].'))
        search=FakeSearch();conv=self.conversation()
        answer,_=handle_multi_agent('Does the PDF discuss museum attractions?',conv,model,search,PriceStub(),economy_mode=True)
        self.assertIn('Document evidence:',answer);self.assertIn('Internet evidence:',answer)
        self.assertEqual(len(search.queries),1)

    def test_document_only_does_not_fallback_to_web(self):
        model=Model(call('delegate',{'agent':'documents','task':'Only use PDF'},'d'),
                    call('search_documents',{'query':'summary'},'s'),AIMessage(content='DOCUMENT_NOT_FOUND'))
        search=FakeSearch();conv=self.conversation()
        answer,_=handle_multi_agent('According to the PDF, explain museum attractions only',conv,model,search,PriceStub(),economy_mode=True)
        self.assertIn('not found',answer);self.assertEqual(search.queries,[])

    def test_document_request_corrects_general_handoff_without_web(self):
        model=Model(call('delegate',{'agent':'general','task':'Explain PDF'},'d'),
                    call('search_documents',{'query':'summary'},'s'),AIMessage(content='Python variables store values.'))
        search=FakeSearch();conv=self.conversation()
        answer,_=handle_multi_agent('Explain the PDF in short',conv,model,search,PriceStub(),economy_mode=True)
        self.assertIn('Python',answer);self.assertEqual(search.queries,[])

    def test_document_only_blocks_wrong_internet_handoff(self):
        model=Model(call('delegate',{'agent':'internet','task':'Search externally'},'wrong'),
                    call('delegate',{'agent':'documents','task':'Only use PDF'},'right'),
                    call('search_documents',{'query':'summary'},'s'),AIMessage(content='Python variables store values.'))
        search=FakeSearch();conv=self.conversation()
        answer,_=handle_multi_agent('Only the PDF: explain variables',conv,model,search,PriceStub(),economy_mode=True)
        self.assertEqual(search.queries,[]);self.assertIn('Python',answer)

    def test_private_facts_and_math_do_not_force_internet(self):
        self.assertFalse(external_factual_question('What is my name?'))
        self.assertFalse(external_factual_question('What is 2 + 2?'))
        self.assertTrue(external_requested('latest news'))
        self.assertTrue(strict_document_scope('Use only the PDF'))

    def test_stale_presentation_module_is_reloaded(self):
        stale=SimpleNamespace(public_activity=lambda events:events)
        ready=SimpleNamespace(**{name:lambda *args:[] for name in ('public_activity','simple_activity','public_answer','public_error')})
        with patch.object(runtime,'presentation_module',stale),patch.object(runtime.importlib,'reload',return_value=ready) as reload:
            self.assertIs(runtime.load_presentation(),ready)
            reload.assert_called_once_with(stale)
