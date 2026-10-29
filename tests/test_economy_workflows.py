import tempfile
import unittest
from pathlib import Path
from langchain_core.messages import AIMessage
from app.attachments.store import AttachmentStore
from app.rag.documents import DocumentIndex
from app.tools.document_search import create_document_tool
from app.agent.session import Conversation
from app.history.store import HistoryStore
from app.multi_agent.system import handle_multi_agent
from app.multi_agent.requests import file_roles, wants_calculation, wants_one_line
from app.ui.presentation import public_activity, public_answer, public_error
from app.multi_agent.errors import AgentExecutionError
from test_multi_agent import Model,call
from test_workflows import FakeSearch
from test_chat_router import PriceStub

class EconomyTests(unittest.TestCase):
    def files(self):
        index=DocumentIndex();index.add_sections('guide.pdf',['Python Tutorial Release 3.7.0','Unrelated page 2'])
        files=AttachmentStore(index);files.add('sales.csv',b'Order,Tax\nA,10\nB,20\n')
        return files

    def run_case(self,model,question,files,economy=True):
        with tempfile.TemporaryDirectory() as folder:
            conv=Conversation(model,history_store=HistoryStore(Path(folder)/'history.sqlite3'),
                document_index=files.documents,additional_tools=[create_document_tool(files.documents)])
            try:
                answer,_=handle_multi_agent(question,conv,model,FakeSearch(),PriceStub(),attachments=files,economy_mode=economy)
                return answer,list(conv.activity),list(conv.sources)
            finally:conv.close()

    def test_both_files_first_page_summary_uses_four_model_calls(self):
        # Supervisor makes two handoffs in one response; Data needs only inspect;
        # Documents retrieves once and explains. No discarded final model calls.
        handoffs=AIMessage(content='',tool_calls=[
            {'name':'delegate','args':{'agent':'data','task':'Describe first sheet'},'id':'h1','type':'tool_call'},
            {'name':'delegate','args':{'agent':'documents','task':'Explain first PDF page briefly'},'id':'h2','type':'tool_call'}])
        model=Model(handoffs,call('inspect_table',{},'i'),
            call('search_documents',{'query':'summary','page':'2'},'s'),
            AIMessage(content='The title page identifies the Python Tutorial release 3.7.0.'))
        answer,activity,sources=self.run_case(model,'ek line dono file k first page k baare mai batao?',self.files())
        self.assertEqual(model.outputs,[])
        self.assertIn('2 records',answer)
        self.assertIn('p. 1',answer)
        self.assertNotIn('No table calculation',answer)
        self.assertNotIn('Retrieved passage',answer)
        self.assertNotIn('\n',answer)
        self.assertEqual({s['page'] for s in sources},{1})
        self.assertFalse(any('calculate_table' in event for event in activity))

    def test_economy_calculation_keeps_verified_sum_with_three_calls(self):
        model=Model(call('delegate',{'agent':'data','task':'Calculate Excel tax'},'h'),
            call('inspect_table',{},'i'),call('calculate_table',{'plan_json':'{"operation":"sum","column":"Tax"}'},'c'))
        answer,_,_=self.run_case(model,'Calculate Excel total tax',self.files())
        self.assertIn('30',answer);self.assertEqual(model.outputs,[])

    def test_duplicate_handoff_does_not_rerun_specialist(self):
        model=Model(call('delegate',{'agent':'data','task':'Describe Excel'},'h1'),call('inspect_table',{},'i'),
            AIMessage(content='Table summary'),call('delegate',{'agent':'data','task':'Describe Excel again'},'h2'),AIMessage(content='Done'))
        answer,activity,_=self.run_case(model,'Describe Excel',self.files(),economy=False)
        self.assertEqual(activity.count('Data Analyst Agent selected tool: inspect_table'),1)
        self.assertIn('Reused completed data specialist result.',activity)
        self.assertNotIn('Handoff limit',answer)

    def test_attachment_footer_does_not_request_extra_agents(self):
        question='Summarize Excel\n\nAttached files: sales.csv, guide.pdf'
        roles=file_roles(question,[{'document':'guide.pdf'}],{'sales.csv':None},{})
        self.assertEqual(roles,['data'])
        self.assertFalse(wants_calculation('Tell me about the sales table'))
        self.assertTrue(wants_one_line('ek line dono files batao'))

    def test_public_view_hides_provider_activity_and_errors(self):
        events=['Model provider used: groq','gemini unavailable; trying configured backup.',
                'General Agent selected tool: calculator']
        self.assertEqual(public_activity(events),events[-1:])
        self.assertEqual(public_answer('A photo.\n\nModel provider used: gemini-vision'),'A photo.')
        self.assertNotIn('Groq',public_error(AgentExecutionError('Groq unavailable')))
