import unittest
import json
from unittest.mock import patch
from langchain_core.messages import AIMessage
from app.workflows.chat_router import handle_chat
from app.spreadsheet.analysis import read_table
from test_workflows import FakeSearch, report_draft, trip_request, trip_draft

class ConversationStub:
    def __init__(self): self.records=[]
    def record_workflow(self,*args): self.records.append(args)

class SequenceModel:
    def __init__(self,*results): self.results=list(results); self.calls=0
    def invoke(self,messages):
        self.calls+=1
        return AIMessage(content=json.dumps(self.results.pop(0)))

class PriceStub:
    def offers(self,request): raise AssertionError("Unexpected price request")

class ChatRouterTests(unittest.TestCase):
    def test_goa_codes_automatic_and_wrong_code_corrected(self):
        model=SequenceModel({"origin":"Delhi","destination":"Goa","include_prices":True,"arrival_airport":"DEL"})
        answer,pending=handle_chat("Plan Delhi to Goa trip with flight prices",ConversationStub(),model,FakeSearch(),PriceStub())
        self.assertEqual(pending["departure_airport"],"DEL")
        self.assertEqual(pending["arrival_airport"],"GOI,GOX")
        self.assertNotIn("arrival airport code",answer)
        self.assertIn("GOI,GOX",answer)

    def test_bare_trip_asks_details_without_model(self):
        model=SequenceModel()
        conv=ConversationStub()
        answer,pending=handle_chat("plan the trip ?",conv,model,FakeSearch(),PriceStub())
        self.assertIn("starting city",answer)
        self.assertIn("destination",answer)
        self.assertEqual(model.calls,0)
        self.assertEqual(pending["mode"],"trip")
        self.assertEqual(len(conv.records),1)

    def test_normal_chat_falls_through(self):
        conv=ConversationStub()
        answer,pending=handle_chat("Hello",conv,SequenceModel(),FakeSearch(),PriceStub())
        self.assertIsNone(answer)
        self.assertEqual(conv.records,[])

    def test_missing_trip_then_followup(self):
        conv=ConversationStub()
        model=SequenceModel({"destination":"Jaipur"},trip_request(),trip_draft())
        search=FakeSearch()
        answer,pending=handle_chat("Plan a trip to Jaipur",conv,model,search,PriceStub())
        self.assertIn("start date",answer)
        self.assertEqual(search.queries,[])
        answer,pending=handle_chat("Here are my dates and budget",conv,model,search,PriceStub(),pending=pending)
        self.assertIn("Suggested Trip Plan",answer)
        self.assertIsNone(pending)
        self.assertEqual(len(conv.records),2)

    def test_excel_tax_without_model(self):
        conv=ConversationStub()
        frame=read_table(b"Item,Tax (USD)\nA,10\nB,20\n","tax.csv")
        answer,pending=handle_chat("calculate all tax",conv,SequenceModel(),FakeSearch(),PriceStub(),frame=frame)
        self.assertIn("30.00",answer)
        self.assertEqual(len(conv.records),1)

    def test_research_from_chat(self):
        conv=ConversationStub()
        model=SequenceModel({"topic":"Tourism"},report_draft())
        answer,pending=handle_chat("Research tourism",conv,model,FakeSearch(),PriceStub())
        self.assertIn("Research Report",answer)
        self.assertEqual(model.calls,2)

    def test_cancellation(self):
        answer,pending=handle_chat("cancel",ConversationStub(),SequenceModel(),FakeSearch(),PriceStub(),pending={"mode":"trip"})
        self.assertIn("Cancelled",answer)
        self.assertIsNone(pending)

    def test_missing_excel_upload(self):
        answer,pending=handle_chat("Analyze my Excel",ConversationStub(),SequenceModel(),FakeSearch(),PriceStub())
        self.assertIn("sidebar",answer)

if __name__=="__main__": unittest.main()
