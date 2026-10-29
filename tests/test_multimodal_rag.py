import unittest
from unittest.mock import patch
from app.rag.documents import DocumentError, DocumentIndex
from app.rag.anything_rag import (
    AnythingRAGIndex,
    RagBackendError,
    create_document_index,
)
from app.rag.anything_rag_worker import pages_from_blocks, validate_media_paths
from pathlib import Path


class WorkerStub:
    def __init__(self, fail_index=False):
        self.calls = []
        self.fail_index = fail_index

    def run(self, action, name, payload=None, query=""):
        self.calls.append((action, name))
        if action == "parse":
            return {
                "pages": [
                    {"page": 1, "text": "Document title and intro."},
                    {"page": 2, "text": "The tax amount is 25."},
                ],
                "folder": "job",
            }
        if action == "index" and self.fail_index:
            raise RagBackendError("Quota unavailable")
        if action == "query":
            return {"context": "Retrieved relationship: tax relates to sales."}
        return {"folder": "job", "message": "ready"}


class MultimodalTests(unittest.TestCase):
    def test_normalization_preserves_page_and_modalities(self):
        pages = pages_from_blocks(
            [
                {"type": "text", "text": "Title", "page_idx": 0},
                {
                    "type": "table",
                    "table_body": "Tax 25",
                    "table_caption": ["Sales"],
                    "page_idx": 1,
                },
                {"type": "equation", "latex": "a+b", "page_idx": 1},
            ]
        )
        self.assertEqual([p["page"] for p in pages], [1, 2])
        self.assertIn("Tax 25", pages[1]["text"])
        self.assertIn("a+b", pages[1]["text"])

    def test_media_path_is_confined(self):
        with self.assertRaises(ValueError):
            validate_media_paths(
                [{"img_path": str(Path(__file__).resolve())}], Path("data/anotherjob")
            )

    def test_graph_query_augments_real_passages(self):
        worker = WorkerStub()
        index = AnythingRAGIndex(worker)
        index._install_pages(
            "sample.pdf",
            "hash",
            [
                {"page": 1, "text": "Document title and intro."},
                {"page": 2, "text": "Tax amount is 25."},
            ],
        )
        index._jobs["sample.pdf"] = "job"
        index.backend_status["sample.pdf"] = {"retrieval": "RAG-Anything"}
        result = index.search("tax")
        self.assertIn("multimodal_context", result)
        self.assertEqual(result["passages"][0]["page"], 2)
        index.search("tax")
        self.assertEqual(len(worker.calls), 1)

    def test_page_scope_never_calls_graph(self):
        worker = WorkerStub()
        index = AnythingRAGIndex(worker)
        index._install_pages(
            "sample.pdf",
            "hash",
            [{"page": 1, "text": "First page"}, {"page": 2, "text": "Second page"}],
        )
        index.backend_status["sample.pdf"] = {"retrieval": "RAG-Anything"}
        result = index.search("second page", "sample.pdf", "1")
        self.assertEqual(result["method"], "exact_page")
        self.assertEqual(result["passages"][0]["text"], "First page")
        self.assertEqual(worker.calls, [])

    def test_index_failure_keeps_mineru_evidence(self):
        index = AnythingRAGIndex(WorkerStub(fail_index=True))
        index.add_file("notes.docx", b"officefixture")
        self.assertEqual(index.backend_status["notes.docx"]["retrieval"], "parsed_only")
        self.assertIn("Quota", index.backend_status["notes.docx"]["message"])
        self.assertEqual(index.search("title")["status"], "ok")

    def test_factory_keeps_base_environment_optional(self):
        with patch.dict("os.environ", {"DOCUMENT_RAG_BACKEND": "standard"}):
            self.assertIs(type(create_document_index()), DocumentIndex)

    def test_clear_drops_jobs_and_scope(self):
        index = AnythingRAGIndex(WorkerStub())
        index._jobs = {"one": "job"}
        index.clear()
        self.assertFalse(index._jobs)
        self.assertFalse(index.backend_status)

    def test_invalid_page_metadata_is_rejected(self):
        with self.assertRaises(DocumentError):
            AnythingRAGIndex(WorkerStub())._install_pages(
                "x.pdf", "h", [{"page": 0, "text": "x"}]
            )


if __name__ == "__main__":
    unittest.main()
