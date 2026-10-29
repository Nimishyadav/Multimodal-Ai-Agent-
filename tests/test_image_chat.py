import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch
from PIL import Image
from langchain_core.messages import AIMessage
from app.attachments.store import AttachmentStore
from app.rag.documents import DocumentIndex
from app.agent.session import Conversation
from app.history.store import HistoryStore
from app.multi_agent.system import handle_multi_agent
from app.ui.presentation import public_answer
from test_multi_agent import Model,call
from test_workflows import FakeSearch
from test_chat_router import PriceStub

class ImageChatTests(unittest.TestCase):
    def files(self):
        files=AttachmentStore(DocumentIndex());data=BytesIO()
        Image.new('RGB',(10,10),'red').save(data,format='PNG');files.add('photo.png',data.getvalue())
        return files

    def test_same_question_reuses_analysis_despite_tool_paraphrase(self):
        files=self.files()
        with tempfile.TemporaryDirectory() as folder, patch('app.attachments.vision.describe_image',return_value='A red square.') as vision:
            for i,focus in enumerate(['Explain the image','Summarize this photo briefly']):
                model=Model(call('delegate',{'agent':'images','task':'Explain the image'},'h'),
                    call('analyze_image',{'filename':'photo.png','question':focus},'v'))
                conv=Conversation(model,history_store=HistoryStore(Path(folder)/'history.sqlite3'))
                try:
                    answer,_=handle_multi_agent('Explain the image in 1 line',conv,model,FakeSearch(),PriceStub(),attachments=files,economy_mode=True)
                    self.assertEqual(answer,'A red square.')
                    self.assertEqual(len(answer.splitlines()),1)
                finally:conv.close()
        self.assertEqual(vision.call_count,1)
        self.assertIn('photo.png',files.image_notes)

    def test_image_context_isolated_and_clear_removes_notes(self):
        one=self.files();two=self.files()
        one.remember_image_answer('photo.png','describe','A red square.')
        self.assertIsNone(two.cached_image_answer('photo.png','describe'))
        one.clear()
        self.assertFalse(one.images);self.assertFalse(one.image_notes)
        self.assertIsNone(one.cached_image_answer('photo.png','describe'))

    def test_legacy_provider_error_is_not_displayed_verbatim(self):
        message=public_answer('All configured compatible model APIs are unavailable or cooling down. Check quotas/keys.')
        self.assertNotIn('model APIs',message)
        self.assertIn('attachments remain in this chat',message)

    def test_upgrade_preserves_existing_chat_image(self):
        from types import SimpleNamespace
        from app.attachments.store import upgrade_attachment_store
        current=self.files()
        old=SimpleNamespace(documents=current.documents,items=current.items,tables=current.tables,images=current.images)
        upgraded=upgrade_attachment_store(old)
        self.assertIn('photo.png',upgraded.images)
        self.assertEqual(upgraded.items,current.items)
        self.assertIsNone(upgraded.cached_image_answer('photo.png','describe'))
        self.assertIs(upgrade_attachment_store(upgraded),upgraded)

    def test_migration_import_does_not_require_new_helper_on_cached_store_module(self):
        import importlib
        from app.attachments import store as store_module
        from app.attachments import migration
        from unittest.mock import patch
        with patch.object(store_module,'upgrade_attachment_store',new=None):
            del store_module.upgrade_attachment_store
            module=importlib.reload(migration)
            files=self.files()
            self.assertIs(module.upgrade_attachment_store(files),files)
