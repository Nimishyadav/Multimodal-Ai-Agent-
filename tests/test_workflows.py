import json
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from langchain_core.messages import AIMessage
from app.workflows.assistants import build_workflow
from app.integrations.web_search import SearchError
from app.history.store import HistoryStore
from app.agent.session import Conversation


class FakeSearch:
    def __init__(self, failure=False):
        self.queries = []
        self.failure = failure

    def search(self, query):
        self.queries.append(query)
        if self.failure:
            raise SearchError("Search unavailable.")
        return {"sources": [{"title": "Tourism evidence", "url": "https://example.org/source",
                            "snippet": "A museum and a public park are visitor attractions."}],
                "searched_at": "2026-10-14T00:00:00Z", "cached": False}


class FakeModel:
    def __init__(self, draft):
        self.draft = draft
        self.calls = 0

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.calls += 1
        return AIMessage(content=json.dumps(self.draft))


def report_draft():
    return {"summary": "A source describes visitor attractions.",
            "findings": [{"text": "The area includes a museum.", "source_ids": ["S1"]}],
            "limitations": ["Only snippets reviewed."]}


def trip_request():
    start = date.today() + timedelta(days=10)
    return {"mode": "trip", "origin": "Delhi", "destination": "Jaipur",
            "start": start.isoformat(), "end": (start + timedelta(days=1)).isoformat(),
            "travelers": 2, "budget_inr": 10000, "interests": "Museums"}


def trip_draft():
    return {"days": [{"day": i, "morning": "Suggested museum visit.",
                      "afternoon": "Suggested park visit.", "evening": "Rest at accommodation.",
                      "source_ids": ["S1"]} for i in (1, 2)],
            "limitations": ["Opening hours and costs need confirmation."]}


class WorkflowTests(unittest.TestCase):
    def test_research_calls_and_actual_sources(self):
        model, search = FakeModel(report_draft()), FakeSearch()
        result = build_workflow(model, search).invoke({"request": {"mode": "research", "topic": "Tourism"}})
        self.assertEqual(model.calls, 1)
        self.assertEqual(len(search.queries), 2)
        self.assertEqual(len(result["sources"]), 1)
        self.assertIn("[S1]", result["answer"])
        self.assertIn("https://example.org/source", result["answer"])

    def test_no_sources_no_model(self):
        model, search = FakeModel(report_draft()), FakeSearch(failure=True)
        result = build_workflow(model, search).invoke({"request": {"mode": "research", "topic": "Tourism"}})
        self.assertEqual(model.calls, 0)
        self.assertEqual(result["sources"], [])
        self.assertIn("No web evidence", result["answer"])

    def test_invalid_citation_rejected(self):
        draft = report_draft()
        draft["findings"][0]["source_ids"] = ["S999"]
        with self.assertRaises(ValueError):
            build_workflow(FakeModel(draft), FakeSearch()).invoke(
                {"request": {"mode": "research", "topic": "Tourism"}})

    def test_trip_budget_and_dates(self):
        result = build_workflow(FakeModel(trip_draft()), FakeSearch()).invoke({"request": trip_request()})
        self.assertIn("Daily group allowance:** INR 5,000.00", result["answer"])
        self.assertIn("Per-person daily allowance:** INR 2,500.00", result["answer"])
        self.assertIn("not market prices", result["answer"])

    def test_invalid_trip_before_search(self):
        request = trip_request()
        request["end"] = request["start"] = "2020-01-01"
        search = FakeSearch()
        with self.assertRaises(ValueError):
            build_workflow(FakeModel(trip_draft()), search).invoke({"request": request})
        self.assertEqual(search.queries, [])

    def test_wrong_number_of_days_rejected(self):
        draft = trip_draft()
        draft["days"].pop()
        with self.assertRaises(ValueError):
            build_workflow(FakeModel(draft), FakeSearch()).invoke({"request": trip_request()})

    def test_generated_url_rejected(self):
        draft = report_draft()
        draft["summary"] = "See https://invented.example.org"
        with self.assertRaises(ValueError):
            build_workflow(FakeModel(draft), FakeSearch()).invoke(
                {"request": {"mode": "research", "topic": "Tourism"}})

    def test_completed_workflow_persist_and_reopen(self):
        with tempfile.TemporaryDirectory() as folder:
            history = HistoryStore(Path(folder) / "history.sqlite3")
            model = FakeModel(report_draft())
            conversation = Conversation(model, history_store=history, user_id="tester")
            try:
                result = build_workflow(model, FakeSearch()).invoke(
                    {"request": {"mode": "research", "topic": "Tourism"}})
                conversation.record_workflow("Research tourism", result["answer"],
                                             result["activity"], result["sources"])
                chat_id = conversation.thread_id
                self.assertEqual(len(history.transcript("tester", chat_id)), 2)
            finally:
                conversation.close()
            restored = Conversation(model, history_store=history, user_id="tester", chat_id=chat_id)
            try:
                self.assertIn("Research Report", restored.graph.get_state(restored.config).values["messages"][-1].content)
            finally:
                restored.close()


if __name__ == "__main__":
    unittest.main()
