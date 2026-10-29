import tempfile
import unittest
from pathlib import Path
from langchain_core.messages import AIMessage
from app.multi_agent.system import handle_multi_agent
from app.agent.session import Conversation
from app.history.store import HistoryStore
from app.rag.documents import DocumentIndex
from app.tools.document_search import create_document_tool
from app.spreadsheet.analysis import read_table,detect_header_row
from test_multi_agent import Model,call
from test_workflows import FakeSearch
from test_chat_router import PriceStub

class FileGroundingTests(unittest.TestCase):

    def test_compound_guard_requires_missing_document_handoff(self):
        frame=read_table(b"Tax\n10\n20\n","table.csv")
        model=Model(call("delegate",{"agent":"data","task":"Sum Excel tax"},"h1"),
                    call("inspect_table",{},"i1"),
                    call("calculate_table",{"plan_json":'{"operation":"sum","column":"Tax"}'},"c1"),
                    AIMessage(content="30"),
                    AIMessage(content="Done: Excel tax 30 and imaginary PDF example."),
                    call("delegate",{"agent":"documents","task":"Explain PDF tax example"},"h2"),
                    call("ask_user",{"question":"Please upload PDF."},"q1"),
                    AIMessage(content="Upload PDF"),
                    AIMessage(content="Excel completed; PDF missing."))
        with tempfile.TemporaryDirectory() as folder:
            conv=Conversation(model,history_store=HistoryStore(Path(folder)/"history.sqlite3"))
            try:
                answer,_=handle_multi_agent("calculate the tax column and what is python take help of pdf and tell me",conv,model,FakeSearch(),PriceStub(),frame=frame)
                self.assertIn("Orchestrator assigned documents specialist.",conv.activity)
                self.assertNotIn("imaginary",answer)
                self.assertIn("No PDF passages",answer)
            finally:conv.close()


    def test_empty_data_explanation_keeps_actual_calculation(self):
        frame=read_table(b"Tax\n10\n20\n","table.csv")
        model=Model(call("delegate",{"agent":"data","task":"Sum Excel tax"},"h1"),
                    call("inspect_table",{},"i1"),
                    call("calculate_table",{"plan_json":'{"operation":"sum","column":"Tax"}'},"c1"),
                    AIMessage(content=""),
                    AIMessage(content="Fake Excel range F2:F8"))
        with tempfile.TemporaryDirectory() as folder:
            conv=Conversation(model,history_store=HistoryStore(Path(folder)/"history.sqlite3"))
            try:
                answer,_=handle_multi_agent("Excel tax total",conv,model,FakeSearch(),PriceStub(),frame=frame)
                self.assertIn("30",answer)
                self.assertNotIn("F2:F8",answer)
                self.assertIn("Verified local",answer)
            finally:conv.close()

    def test_actual_files_reject_fabricated_final(self):
        pdf=Path(__file__).resolve().parent / "fixtures" / "Python Tutorial.pdf"
        excel=Path(__file__).resolve().parent / "fixtures" / "Supermarket-Sales-Sample-Data.xlsx"
        if not pdf.exists() or not excel.exists():
            self.skipTest("Optional fixtures unavailable")
        index=DocumentIndex()
        index.add(pdf.name,pdf.read_bytes())
        data=excel.read_bytes()
        frame=read_table(data,excel.name,header_row=detect_header_row(data,excel.name))
        handoffs=AIMessage(content="",tool_calls=[
            {"name":"delegate","args":{"agent":"documents","task":"Explain PDF tax example"},"id":"h1","type":"tool_call"},
            {"name":"delegate","args":{"agent":"data","task":"Sum Excel Tax (USD)"},"id":"h2","type":"tool_call"}])
        model=Model(handoffs,
                    call("search_documents",{"query":"tax price 12.5 100.50","document":pdf.name},"d1"),
                    AIMessage(content="Product A 50 dollars at 8 percent."),
                    call("inspect_table",{},"i1"),
                    call("calculate_table",{"plan_json":'{"operation":"sum","column":"Tax (USD)"}'},"c1"),
                    AIMessage(content="=SUM(F2:F8)"),
                    AIMessage(content="Product A at 8 percent, =SUM(F2:F8)"))
        with tempfile.TemporaryDirectory() as folder:
            conv=Conversation(model,history_store=HistoryStore(Path(folder)/"history.sqlite3"),
                              additional_tools=[create_document_tool(index)],document_index=index)
            try:
                answer,_=handle_multi_agent("PDF ka tax example samjhao aur Excel ka total tax calculate karo.",
                                           conv,model,FakeSearch(),PriceStub(),frame=frame)
                self.assertNotIn("Product A",answer)
                self.assertNotIn("SUM(F2:F8)",answer)
                self.assertIn("93.984",answer)
                self.assertIn("page 16",answer)
                self.assertIn("12.5625",answer)
                self.assertIn("113.0625",answer)
            finally:conv.close()

if __name__=="__main__":unittest.main()
