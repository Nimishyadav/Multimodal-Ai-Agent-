import unittest
from app.ui.presentation import simple_activity


class SimpleActivityTests(unittest.TestCase):
    def test_document_trace_is_short_and_honest(self):
        events=['Upload accepted: Python Tutorial.pdf',
                'Upload processing: Python Tutorial.pdf — local_fallback / BM25',
                'Orchestrator Agent selected tool: delegate',
                'Orchestrator assigned documents specialist.',
                'Document Agent selected tool: search_documents',
                'Document retrieval: local sampled_overview fallback; Anything RAG index is not ready.',
                'Document Agent tool result: search_documents — returned output.',
                'Document Agent completed its response.',
                'Orchestrator Agent tool result: delegate — returned output.',
                'Orchestrator Agent completed its response.']
        lines=simple_activity(events)
        self.assertEqual(len(lines),4)
        self.assertIn('retrieval unavailable',lines[2])
        self.assertEqual(lines[-1],'Response ready.')
        self.assertNotIn('delegate',' '.join(lines))

    def test_tools_are_kept_and_duplicates_removed(self):
        lines=simple_activity(['Data Analyst Agent selected tool: inspect_table',
                              'Data Analyst Agent selected tool: calculate_table',
                              'Data Analyst Agent selected tool: calculate_table',
                              'Data Analyst Agent tool result: calculate_table — failed.'])
        self.assertEqual(len(lines),3)
        self.assertIn('could not finish',lines[-1])

    def test_real_hybrid_and_exact_page_remain_distinct(self):
        self.assertEqual(simple_activity(['Document retrieval: Anything RAG hybrid graph/vector context + local source passages.']),
                         ['Used RAG-Anything to find relevant content.'])
        self.assertEqual(simple_activity(['Document retrieval: exact PDF page 2; hybrid search skipped.']),['Read PDF page 2.'])

    def test_provider_details_are_not_exposed(self):
        self.assertEqual(simple_activity(['Model provider used: groq','Document Agent selected tool: search_documents']),
                         ['Document Agent: Document search.'])
