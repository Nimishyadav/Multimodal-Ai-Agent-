import tempfile
import unittest
from pathlib import Path
from app.history.store import HistoryStore
from app.history.migration import upgrade_history_store


class HistoryMigrationTests(unittest.TestCase):
    def test_stale_object_keeps_same_database_and_chat(self):
        with tempfile.TemporaryDirectory() as directory:
            history = HistoryStore(Path(directory) / 'history.sqlite3')
            chat = history.create('local-user')
            class OldHistory:
                path = history.path
            upgraded = upgrade_history_store(OldHistory())
            self.assertEqual(upgraded.path, history.path)
            self.assertEqual(upgraded.get_chat('local-user', chat)['id'], chat)
            self.assertIsNone(upgraded.load_attachments('local-user', chat))
            upgraded.save_attachments('local-user', chat, {'version': 1})
            self.assertEqual(upgraded.load_attachments('local-user', chat), {'version': 1})

    def test_current_store_is_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            history = HistoryStore(Path(directory) / 'history.sqlite3')
            self.assertIs(upgrade_history_store(history), history)
