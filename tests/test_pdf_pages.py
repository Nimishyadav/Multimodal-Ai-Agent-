import tempfile
import unittest
from pathlib import Path
from langchain_core.messages import AIMessage
from app.rag.documents import DocumentIndex, requested_page
from app.agent.session import Conversation
from app.history.store import HistoryStore
from app.tools.document_search import create_document_tool
from app.multi_agent.system import handle_multi_agent
from test_multi_agent import Model,call
from test_workflows import FakeSearch
from test_chat_router import PriceStub

class PageTests(unittest.TestCase):
    def test_page_recognition(self):
        for text,expected in [('first page',1),('page 20',20),('2nd page',2),('pehle page',1)]:
            self.assertEqual(requested_page(text),expected)

    def test_exact_page_beats_unrelated_keyword_matches(self):
        index=DocumentIndex()
        index.add_sections('guide.pdf',['Python Tutorial Release 3.7.0','First page examples and lists'])
        result=index.search('first page examples',page='1')
        self.assertEqual(result['method'],'exact_page')
        self.assertEqual({p['page'] for p in result['passages']},{1})
        missing=index.search('examples',page='99')
        self.assertEqual(missing['passages'],[])

    def test_ambiguous_document_does_not_choose_arbitrarily(self):
        index=DocumentIndex();index.add_sections('a.pdf',['Title A']);index.add_sections('b.pdf',['Title B'])
        self.assertEqual(index.search('first page')['status'],'ambiguous_document')

    def test_original_page_overrides_agent_query_and_one_line_stays_short(self):
        index=DocumentIndex();index.add_sections('guide.pdf',['Python Tutorial Release 3.7.0','Python lists and strings'])
        model=Model(call('delegate',{'agent':'documents','task':'Explain PDF'},'h'),
            call('search_documents',{'query':'lists','page':'2'},'s'),
            AIMessage(content='This title page identifies the Python Tutorial, release 3.7.0. Extra sentence.'),
            AIMessage(content='Wrong supervisor summary.'))
        with tempfile.TemporaryDirectory() as folder:
            conv=Conversation(model,history_store=HistoryStore(Path(folder)/'history.sqlite3'),
                document_index=index,additional_tools=[create_document_tool(index)])
            try:
                answer,_=handle_multi_agent('Explain first page of PDF in one line',conv,model,FakeSearch(),PriceStub())
                self.assertIn('title page',answer)
                self.assertNotIn('Extra sentence',answer)
                self.assertNotIn('Retrieved passage',answer)
                self.assertNotIn('Retrieved sources',answer)
                self.assertTrue(all(s['page']==1 for s in conv.sources))
            finally:conv.close()
