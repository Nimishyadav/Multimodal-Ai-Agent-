import tempfile
import unittest
from pathlib import Path
from langchain_core.messages import AIMessage
from app.agent.session import Conversation
from app.history.store import HistoryStore
from app.memory.store import MemoryStore

class FakeModel:
    def bind_tools(self, tools): return self
    def invoke(self, messages):
        self.messages = messages
        return AIMessage(content="Test answer.")

class HistoryTests(unittest.TestCase):
    def test_reopen_and_name_context(self):
        with tempfile.TemporaryDirectory() as folder:
            history=HistoryStore(Path(folder)/"history.sqlite3")
            memory=MemoryStore(Path(folder)/"memory.sqlite3")
            model=FakeModel()
            first=Conversation(model,memory_store=memory,history_store=history,user_id="alice")
            first.ask("My name is Nimish")
            chat_id=first.thread_id
            first.close()
            restored=Conversation(model,memory_store=memory,history_store=history,user_id="alice",chat_id=chat_id)
            self.assertEqual(len(history.transcript("alice",chat_id)),2)
            restored.ask("Continue")
            restored.close()
            fresh=Conversation(model,memory_store=memory,history_store=history,user_id="alice")
            self.assertIn("Nimish",fresh.system_message("what is my name?").content)
            self.assertIsNone(history.previous_user_name("bob"))
            memory.save("alice","name","Nimish")
            self.assertIn('"name": "Nimish"',fresh.system_message().content)
            memory.delete("alice","name")
            self.assertNotIn('"name": "Nimish"',fresh.system_message().content)
            fresh.close()


    def test_automatic_cross_chat_recall(self):
        with tempfile.TemporaryDirectory() as folder:
            history=HistoryStore(Path(folder)/"history.sqlite3")
            model=FakeModel()
            first=Conversation(model,history_store=history,user_id="alice")
            first.ask("I am building a supermarket dashboard for my internship.")
            first.close()
            fresh=Conversation(model,history_store=history,user_id="alice")
            fresh.ask("What dashboard am I building?")
            self.assertIn("supermarket dashboard",model.messages[0].content)
            self.assertEqual(history.relevant_history("bob","dashboard","other"),[])
            self.assertFalse(any("supermarket" in row["text"] for row in history.relevant_history("alice","dashboard",first.thread_id)))
            fresh.close()


    def test_broad_recap_all_chats(self):
        with tempfile.TemporaryDirectory() as folder:
            history=HistoryStore(Path(folder)/"history.sqlite3")
            model=FakeModel()
            first=Conversation(model,history_store=history,user_id="alice")
            first.ask("Explain PDF retrieval")
            first.close()
            second=Conversation(model,history_store=history,user_id="alice")
            second.ask("Calculate sales tax")
            second.close()
            recap=Conversation(model,history_store=history,user_id="alice")
            context=recap.system_message("what talks till now and what all things I have asked you?").content
            self.assertIn("Explain PDF retrieval",context)
            self.assertIn("Calculate sales tax",context)
            self.assertIn('"total_saved_questions": 2',context)
            self.assertEqual(history.history_overview("bob")["total_saved_questions"],0)
            recap.close()


    def test_history_recap_does_not_depend_on_model(self):
        with tempfile.TemporaryDirectory() as folder:
            history=HistoryStore(Path(folder)/"history.sqlite3")
            model=FakeModel()
            old=Conversation(model,history_store=history,user_id="alice")
            old.ask("Explain PDF retrieval")
            old.close()
            fresh=Conversation(model,history_store=history,user_id="alice")
            answer=fresh.ask("Summarize what we discussed across all my previous chats.")
            self.assertIn("Explain PDF retrieval",answer)
            self.assertIn("no LLM",fresh.activity[0])
            self.assertEqual(len(history.transcript("alice",fresh.thread_id)),2)
            fresh.close()

if __name__=="__main__": unittest.main()
