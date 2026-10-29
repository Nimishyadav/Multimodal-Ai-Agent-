import json
import unittest
from langchain_core.messages import ToolMessage,AIMessage
from langchain_core.tools import tool
from app.agent.activity import tool_result_activity
from app.multi_agent.system import run_tool_agent,CallBudget
from test_multi_agent import Model,call

class ActivityTests(unittest.TestCase):
 def result(self,name,value):return ToolMessage(content=json.dumps(value),tool_call_id='test',name=name)
 def test_sensitive_output_is_not_logged(self):
  events=tool_result_activity('Data Agent','calculate_table',self.result('calculate_table',{'status':'error','message':'private-api-key-value'}))
  self.assertIn('failed',events[0]);self.assertNotIn('private-api-key-value',' '.join(events))
 def test_pandas_operation_is_visible(self):
  events=tool_result_activity('Data Agent','calculate_table',self.result('calculate_table',{'result_preview':[{'result':25}],'result_rows':1}))
  self.assertTrue(any('Pandas' in event for event in events))
 def test_legacy_document_tool_reports_actual_method(self):
  events=tool_result_activity('General Agent','search_documents',self.result('search_documents',{'status':'ok','method':'BM25','retrieval_info':{'backend':'standard'}}),document_details=True)
  self.assertTrue(any('standard backend' in event for event in events))
 def test_agent_loop_logs_selection_and_result(self):
  @tool
  def demo_tool(value: str) -> str:
   """Return a synthetic result for a local test."""
   return 'done'
  model=Model(call('demo_tool',{'value':'x'},'tool-1'),AIMessage(content='Completed.'))
  events=[]
  run_tool_agent('Test Agent',model,[demo_tool],[],CallBudget(),events)
  self.assertTrue(any('selected tool: demo_tool' in event for event in events))
  self.assertTrue(any('tool result: demo_tool' in event for event in events))

if __name__=='__main__':unittest.main()
