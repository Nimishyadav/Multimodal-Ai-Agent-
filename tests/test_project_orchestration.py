import tempfile
import unittest
from pathlib import Path
from langchain_core.messages import AIMessage
from app.agent.session import Conversation as RealConversation
from app.history.store import HistoryStore
from app.memory.store import MemoryStore
from app.multi_agent.system import handle_multi_agent
from app.spreadsheet.analysis import read_table
from test_multi_agent import Model, call
from test_workflows import FakeSearch
from test_chat_router import PriceStub

class ProjectOrchestrationTests(unittest.TestCase):
    def run_case(self,model,question,frame=None,memory_setup=None):
        with tempfile.TemporaryDirectory() as folder:
            memory=MemoryStore(Path(folder)/"memory.sqlite3")
            history=HistoryStore(Path(folder)/"history.sqlite3")
            if memory_setup:memory_setup(memory)
            conv=RealConversation(model,memory_store=memory,history_store=history,user_id="tester")
            try:
                answer,pending=handle_multi_agent(question,conv,model,FakeSearch(),PriceStub(),frame=frame)
                self.assertEqual(len(history.transcript("tester",conv.thread_id)),2)
                return answer,list(conv.activity)
            finally:conv.close()

    def test_data_agent_inspects_and_calculates(self):
        model=Model(call("delegate",{"agent":"data","task":"Calculate total Sales"},"h1"),
                    call("inspect_table",{},"i1"),
                    call("calculate_table",{"plan_json":'{"operation":"sum","column":"Sales"}'},"c1"),
                    AIMessage(content="Total Sales: 350."),
                    AIMessage(content="Total Sales: 350."))
        frame=read_table(b"Sales\n100\n200\n50\n","data.csv")
        answer,activity=self.run_case(model,"Calculate Excel Sales",frame)
        self.assertIn("350",answer)
        self.assertIn("Data Analyst Agent selected tool: calculate_table",activity)


    def test_compound_research_then_data(self):
        model=Model(call("delegate",{"agent":"research","task":"Find tourism evidence"},"r1"),
                    call("search_web",{"query":"tourism evidence"},"s1"),
                    AIMessage(content="Museum evidence [S1]."),
                    call("delegate",{"agent":"data","task":"Calculate Sales to support report"},"d1"),
                    call("inspect_table",{},"i1"),
                    call("calculate_table",{"plan_json":'{"operation":"sum","column":"Sales"}'},"c1"),
                    AIMessage(content="Sales total 350."),
                    AIMessage(content="Tourism evidence [S1]; Sales total 350."))
        frame=read_table(b"Sales\n100\n200\n50\n","sales.csv")
        answer,activity=self.run_case(model,"Research and analyze my sales table",frame)
        self.assertIn("350",answer)
        self.assertIn("Orchestrator assigned research specialist.",activity)
        self.assertIn("Orchestrator assigned data specialist.",activity)

    def test_memory_agent_retrieves(self):
        model=Model(call("delegate",{"agent":"memory","task":"What is my name?"},"h1"),
                    call("retrieve_memory",{"key":"name"},"m1"),
                    AIMessage(content="Your name is Nimish."),
                    AIMessage(content="Your name is Nimish."))
        answer,activity=self.run_case(model,"What is my name?",memory_setup=lambda m:m.save("tester","name","Nimish"))
        self.assertIn("Nimish",answer)
        self.assertIn("Memory Agent selected tool: retrieve_memory",activity)

    def test_missing_pdf_is_clarified(self):
        model=Model(call("delegate",{"agent":"documents","task":"Read my PDF"},"h1"),
                    call("ask_user",{"question":"Please upload and index your PDF."},"q1"),
                    AIMessage(content="Please upload and index your PDF."),
                    AIMessage(content="Please upload and index your PDF."))
        answer,activity=self.run_case(model,"Read my PDF")
        self.assertIn("upload",answer)
        self.assertIn("Document Agent selected tool: ask_user",activity)

if __name__=="__main__":unittest.main()
