import unittest
from langchain_core.messages import AIMessage
from app.rag.intents import is_document_overview
from app.rag.documents import DocumentIndex
from test_multi_agent import Model, call
import test_economy_workflows as helpers

class OverviewTests(unittest.TestCase):
    files = helpers.EconomyTests.files
    run_case = helpers.EconomyTests.run_case

    def test_overview_phrasing(self):
        for text in ['explain the document in short', 'Summarize the uploaded PDF', 'what is this document about?', 'give me a brief summary of the document']:
            self.assertTrue(is_document_overview(text), text)
        self.assertFalse(is_document_overview('Explain tax in the document'))
        self.assertFalse(is_document_overview('Explain PDF page 1'))

    def test_natural_summary_retrieves_local_passages(self):
        index = DocumentIndex()
        index.add_sections('manual.docx', ['Solar panels convert sunlight to electricity.', 'Batteries store electricity.'])
        result = index.search('explain the document in short')
        self.assertTrue(result.get('passages'))

    def test_coordinator_value_error_recovers_from_source(self):
        class Unavailable:
            def bind_tools(self, *args, **kwargs): return self
            def invoke(self, *args, **kwargs): raise ValueError('No usable model response')
        answer, activity, sources = self.run_case(Unavailable(), 'explain the document in short', self.files())
        self.assertIn('Python Tutorial', answer)
        self.assertIn('From sampled document excerpts', answer)
        self.assertTrue(sources)
        self.assertTrue(any('local recovery' in event for event in activity))
        self.assertLess(len(answer.split()), 120)

    def test_unsupported_numeric_summary_uses_evidence(self):
        model = Model(call('delegate', {'agent':'documents', 'task':'Summarize'}, 'h'),
                      call('search_documents', {'query':'summary'}, 's'),
                      AIMessage(content='This document contains 999999 examples.'))
        answer, _, _ = self.run_case(model, 'explain the document in short', self.files())
        self.assertNotIn('999999', answer)
        self.assertIn('Python Tutorial', answer)


class CurrentFileTests(OverviewTests):
    def test_pdf_request_overrides_stale_excel_tool_argument(self):
        files = self.files()
        files.documents.add_sections('old.xlsx', ['EXCEL SALES ONLY'])
        model = Model(call('delegate', {'agent':'documents','task':'Summarize PDF'}, 'h'),
                      call('search_documents', {'query':'summary','document':'old.xlsx'}, 's'),
                      AIMessage(content='The Python Tutorial introduces Python.'))
        answer, _, sources = self.run_case(model, 'explain the pdf in short\n\nAttached files: guide.pdf', files)
        self.assertTrue(sources)
        self.assertEqual({s['document'] for s in sources}, {'guide.pdf'})
        self.assertNotIn('EXCEL SALES', answer)

    def test_new_attachment_scopes_generic_summary(self):
        from app.multi_agent.requests import document_targets
        self.assertEqual(document_targets('explain the file in short\n\nAttached files: new.docx',
                                         ['old.xlsx','new.docx']), ['new.docx'])

    def test_truncated_table_fallback_is_readable(self):
        from app.rag.intents import brief_source_overview
        answer = brief_source_overview([{'excerpt':'{"columns": ["Order", "Tax (USD)"], "rows": 7, "preview": [{"Order":'}])
        self.assertIn('7 rows', answer)
        self.assertIn('Tax (USD)', answer)
        self.assertNotIn('{', answer)

    def test_reattach_retries_failed_backend(self):
        from unittest.mock import Mock
        files = self.files()
        files.documents.backend_status = {'sales.csv': {'retrieval':'local_only'}}
        files.documents.add_table_context = Mock()
        files.add('sales.csv', b'Order,Tax\nA,10\nB,20\n')
        files.documents.add_table_context.assert_called_once()
