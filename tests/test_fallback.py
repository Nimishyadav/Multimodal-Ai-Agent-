import unittest
from unittest.mock import patch
from io import BytesIO
import json
from langchain_core.messages import AIMessage
from app.llm.fallback import FallbackModel
from app.multi_agent.errors import AgentExecutionError
from app.integrations.web_search import SearchClient, SearchError

class Failure(Exception):
    def __init__(self, status): self.status_code=status

class Provider:
    def __init__(self, output): self.output=output; self.calls=0; self.tools=None; self.options={}
    def bind_tools(self,tools,**kwargs): self.tools=tools;self.options=kwargs;return self
    def invoke(self,messages):
        self.calls+=1
        if isinstance(self.output,Exception):raise self.output
        return self.output

class FallbackTests(unittest.TestCase):
    def test_quota_failover_preserves_tool_binding_and_cools_failed_provider(self):
        primary=Provider(Failure(429));backup=Provider(AIMessage(content='OK',tool_calls=[{'name':'sample','args':{},'id':'s','type':'tool_call'}]))
        model=FallbackModel([('primary',primary),('backup',backup)])
        bound=model.bind_tools(['sample'],tool_choice='any')
        attempts=[]
        self.assertEqual(bound.invoke_with_budget([],lambda:attempts.append(1)).content,'OK')
        self.assertEqual(backup.tools,['sample'])
        self.assertEqual(backup.options,{'tool_choice':'any'})
        model.invoke([])
        self.assertEqual(primary.calls,1)
        self.assertEqual(backup.calls,2)
        self.assertEqual(attempts,[1])
        self.assertIn('backup',' '.join(model.drain_events()))

    def test_bad_input_does_not_trigger_provider_fallback(self):
        first=Provider(Failure(400));second=Provider(AIMessage(content='bad'))
        with self.assertRaises(Failure):FallbackModel([('first',first),('second',second)]).invoke([])
        self.assertEqual(second.calls,0)

    def test_all_failed_is_safe_and_bounded(self):
        first=Provider(Failure(429));second=Provider(Failure(503))
        model=FallbackModel([('first',first),('second',second)])
        for _ in range(2):
            with self.assertRaisesRegex(AgentExecutionError,'All configured'): model.invoke([])
        self.assertEqual(first.calls,1);self.assertEqual(second.calls,1)

    def test_extra_attempt_consumes_shared_budget_before_call(self):
        first=Provider(Failure(429));second=Provider(AIMessage(content='bad'))
        def stop():raise AgentExecutionError('Budget reached')
        with self.assertRaisesRegex(AgentExecutionError,'Budget'):
            FallbackModel([('first',first),('second',second)]).invoke_with_budget([],stop)
        self.assertEqual(second.calls,0)

    def test_search_fallback_and_cache_preserve_actual_links(self):
        client=SearchClient('primary','backup')
        result={'organic_results':[{'title':'Example','link':'https://example.org','snippet':'Evidence'},
                                   {'title':'Unsafe','link':'javascript:bad'}]}
        with patch.object(client,'_tavily_search',side_effect=SearchError('Quota')) as primary, \
             patch('app.integrations.web_search.urlopen',return_value=BytesIO(json.dumps(result).encode())) as backup:
            first=client.search('test query');second=client.search('test query')
        self.assertEqual(primary.call_count,1);self.assertEqual(backup.call_count,1)
        self.assertEqual(first['provider'],'SerpApi (backup)')
        self.assertEqual(first['sources'][0]['url'],'https://example.org')
        self.assertEqual(len(first['sources']),1);self.assertTrue(second['cached'])

    def test_invalid_query_never_calls_backup(self):
        with patch('app.integrations.web_search.urlopen') as request:
            with self.assertRaises(SearchError):SearchClient(None,'backup').search('')
        request.assert_not_called()


class GeminiTextBlockTests(unittest.TestCase):
    def test_text_blocks_normalize_and_tool_calls_survive(self):
        answer=AIMessage(content=[{'type':'text','text':'OK'}],tool_calls=[
            {'name':'calculator','args':{'expression':'2+2'},'id':'call-1','type':'tool_call'}])
        result=FallbackModel([('gemini',Provider(answer))]).invoke([])
        self.assertEqual(result.content,'OK')
        self.assertEqual(result.tool_calls[0]['name'],'calculator')


class ToolFormatFallbackTests(unittest.TestCase):
    def test_provider_rejected_tool_output_uses_backup(self):
        failure=Failure(400)
        failure.body={'error':{'code':'tool_use_failed'}}
        first=Provider(failure);second=Provider(AIMessage(content='OK'))
        result=FallbackModel([('first',first),('second',second)]).invoke([])
        self.assertEqual(result.content,'OK')
        self.assertEqual(first.calls,1)
        self.assertEqual(second.calls,1)


class InvalidOutputFallbackTests(unittest.TestCase):
    def test_ignored_required_tool_uses_backup_without_executing_any_tool(self):
        first=Provider(AIMessage(content='User Safety: safe'))
        second=Provider(AIMessage(content='',tool_calls=[
            {'name':'calculator','args':{'expression':'2+2'},'id':'x','type':'tool_call'}]))
        result=FallbackModel([('first',first),('second',second)]).bind_tools(['calculator'],tool_choice='any').invoke([])
        self.assertEqual(result.tool_calls[0]['name'],'calculator')
        self.assertEqual(first.calls,1);self.assertEqual(second.calls,1)

    def test_empty_output_uses_backup(self):
        result=FallbackModel([('first',Provider(AIMessage(content=''))),
                              ('second',Provider(AIMessage(content='OK')))]).invoke([])
        self.assertEqual(result.content,'OK')


class FinalAnswerFallbackTests(unittest.TestCase):
    def test_tool_call_during_finalization_uses_backup(self):
        first=Provider(AIMessage(content='',tool_calls=[{'name':'search_documents','args':{'query':'again'},'id':'x','type':'tool_call'}]))
        second=Provider(AIMessage(content='Verified final explanation.'))
        result=FallbackModel([('first',first),('second',second)]).bind_tools(['search_documents'],tool_choice='none').invoke([])
        self.assertEqual(result.content,'Verified final explanation.')
