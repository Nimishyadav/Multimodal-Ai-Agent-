import tempfile
import unittest
from pathlib import Path
from app.attachments.store import AttachmentStore
from app.attachments.persistence import snapshot_attachments, restore_attachments
from app.rag.documents import DocumentIndex
from app.history.store import HistoryStore
from app.memory.store import MemoryStoreError

class PersistenceTests(unittest.TestCase):
    def test_restart_restores_document_and_calculable_table(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'history.sqlite3'
            history = HistoryStore(path)
            chat = history.create('alice')
            store = AttachmentStore(DocumentIndex())
            store.add('guide.txt', b'This document explains renewable energy and solar panels.')
            store.add('sales.csv', b'Item,Tax\nA,10.25\nB,20.50\n')
            store.images['picture.png'] = 'data:image/jpeg;base64,dGVzdA=='
            history.save_attachments('alice', chat, snapshot_attachments(store))
            # A fresh connection and attachment store simulate a server restart.
            reopened = HistoryStore(path)
            restored = restore_attachments(AttachmentStore(DocumentIndex()), reopened.load_attachments('alice', chat))
            self.assertTrue(restored.documents.search('explain the document in short')['passages'])
            self.assertAlmostEqual(restored.tables['sales.csv']['Tax'].sum(), 30.75)
            self.assertEqual(restored.images, store.images)
            self.assertEqual(restored.manifest(), store.manifest())

    def test_restored_document_answers_followup_when_model_is_unavailable(self):
        import test_economy_workflows as helpers
        class Unavailable:
            def bind_tools(self, *args, **kwargs): return self
            def invoke(self, *args, **kwargs): raise ValueError('No model response')
        original = AttachmentStore(DocumentIndex())
        original.add('guide.txt', b'This guide explains Python variables and functions.')
        restored = restore_attachments(AttachmentStore(DocumentIndex()), snapshot_attachments(original))
        answer, activity, sources = helpers.EconomyTests.run_case(self, Unavailable(), 'Explain the document in short.', restored)
        self.assertIn('Python variables', answer)
        self.assertTrue(sources)
        self.assertNotIn('Please upload', answer)

    def test_chat_and_profile_isolation_and_delete(self):
        with tempfile.TemporaryDirectory() as folder:
            history = HistoryStore(Path(folder) / 'history.sqlite3')
            first = history.create('alice')
            second = history.create('alice')
            store = AttachmentStore(DocumentIndex())
            store.add('notes.txt', b'Private chat notes about Python.')
            history.save_attachments('alice', first, snapshot_attachments(store))
            self.assertIsNone(history.load_attachments('alice', second))
            with self.assertRaises(MemoryStoreError): history.load_attachments('bob', first)
            store.clear()
            history.save_attachments('alice', first, snapshot_attachments(store))
            restored = restore_attachments(AttachmentStore(DocumentIndex()), history.load_attachments('alice', first))
            self.assertEqual(restored.manifest(), [])
            history.delete('alice', first)
            with history._connect() as connection:
                self.assertEqual(connection.execute('SELECT COUNT(*) FROM chat_attachments').fetchone()[0], 0)
