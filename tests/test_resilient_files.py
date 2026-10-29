import unittest
from langchain_core.messages import AIMessage
from app.multi_agent.errors import AgentExecutionError
from test_multi_agent import Model,call
import test_economy_workflows as helpers

class ResilienceTests(unittest.TestCase):
    files=helpers.EconomyTests.files
    run_case=helpers.EconomyTests.run_case
    def test_exact_title_tax_two_line_question_finishes_without_document_final_call(self):
        outputs=AIMessage(content='',tool_calls=[
            {'name':'delegate','args':{'agent':'documents','task':'PDF page 1 title'},'id':'h1','type':'tool_call'},
            {'name':'delegate','args':{'agent':'data','task':'Sum Tax'},'id':'h2','type':'tool_call'}])
        model=Model(outputs,call('search_documents',{'query':'title'},'s'),
            call('inspect_table',{},'i'),call('calculate_table',{'plan_json':'{"operation":"sum","column":"Tax"}'},'c'))
        answer,activity,sources=self.run_case(model,'PDF page 1 ka title aur Excel ka total Tax batao—sirf 2 lines mein',self.files())
        self.assertEqual(len(answer.splitlines()),2)
        self.assertIn('PDF title: Python Tutorial',answer)
        self.assertIn('30',answer)
        self.assertNotIn('call limit',answer)
        self.assertEqual({item['page'] for item in sources},{1})

    def test_all_inference_unavailable_still_returns_local_title_and_verified_tax(self):
        class Unavailable:
            def bind_tools(self,*args,**kwargs):return self
            def invoke(self,*args,**kwargs):raise AgentExecutionError('All APIs unavailable')
        answer,activity,sources=self.run_case(Unavailable(),
            'PDF page 1 ka title aur Excel ka total Tax batao—sirf 2 lines mein',self.files())
        self.assertIn('Python Tutorial',answer)
        self.assertIn('30',answer)
        self.assertEqual(len(answer.splitlines()),2)
        self.assertTrue(any('local recovery' in event for event in activity))

    def test_document_failure_does_not_drop_valid_spreadsheet_result(self):
        class SometimesFails(Model):
            def invoke(self,messages):
                result=self.outputs.pop(0)
                if isinstance(result,Exception):raise result
                return result
        handoffs=AIMessage(content='',tool_calls=[
            {'name':'delegate','args':{'agent':'documents','task':'Read page 1'},'id':'h1','type':'tool_call'},
            {'name':'delegate','args':{'agent':'data','task':'Total Tax'},'id':'h2','type':'tool_call'}])
        model=SometimesFails(handoffs,AgentExecutionError('Document Agent reached its call limit'),
            call('inspect_table',{},'i'),call('calculate_table',{'plan_json':'{"operation":"sum","column":"Tax"}'},'c'))
        answer,_,_=self.run_case(model,'PDF page 1 title and Excel total tax in 2 lines',self.files())
        self.assertIn('Python Tutorial',answer);self.assertIn('30',answer)

    def test_document_explanation_finalizes_without_another_tool_loop(self):
        class TrackChoices(Model):
            def __init__(self,*outputs):super().__init__(*outputs);self.choices=[]
            def bind_tools(self,tools,**kwargs):
                self.choices.append(kwargs.get('tool_choice'));return super().bind_tools(tools,**kwargs)
        model=TrackChoices(call('delegate',{'agent':'documents','task':'Explain PDF page 1'},'h'),
            call('search_documents',{'query':'first page'},'s'),AIMessage(content='This is the Python Tutorial title page.'))
        answer,_,_=self.run_case(model,'Explain PDF page 1 briefly',self.files())
        self.assertIn('none',model.choices)
        self.assertIn('title page',answer)

    def test_image_recovery_uses_vision_when_text_coordination_fails(self):
        from io import BytesIO
        from PIL import Image
        files=self.files();data=BytesIO();Image.new('RGB',(10,10),'red').save(data,format='PNG')
        files.add('red.png',data.getvalue())
        class Unavailable:
            def bind_tools(self,*args,**kwargs):return self
            def invoke(self,*args,**kwargs):raise AgentExecutionError('Text unavailable')
        from unittest.mock import patch
        with patch('app.attachments.vision.describe_image',return_value='A red square.') as vision:
            answer,_,_=self.run_case(Unavailable(),'Describe this image in one line',files)
        self.assertIn('red square',answer)
        self.assertEqual(vision.call_count,1)

    def test_sheet_metadata_survives_all_text_api_failures(self):
        class Unavailable:
            def bind_tools(self,*args,**kwargs):return self
            def invoke(self,*args,**kwargs):raise AgentExecutionError('Text unavailable')
        answer,_,_=self.run_case(Unavailable(),'Summarize Excel in one line',self.files())
        self.assertIn('2 records',answer)
        self.assertNotIn('No table calculation',answer)
