import unittest
from langchain_core.messages import AIMessage
from app.multi_agent.system import handle_multi_agent,CallBudget,run_tool_agent
from test_workflows import FakeSearch
from test_chat_router import PriceStub

def call(name,args,id):
    return AIMessage(content="",tool_calls=[{"name":name,"args":args,"id":id,"type":"tool_call"}])

class Model:
    def __init__(self,*outputs):
        self.outputs=list(outputs)
        self.bound=[]
    def bind_tools(self,tools,**kwargs):
        self.bound.append([t.name for t in tools])
        return self
    def invoke(self,messages):
        return self.outputs.pop(0)

class GraphStub:
    def get_state(self,config):
        class State: values={"messages":[]}
        return State()

class Conversation:
    graph=GraphStub()
    config={}
    def __init__(self):self.records=[]
    def record_workflow(self,*args):self.records.append(args)

class MultiAgentTests(unittest.TestCase):
    def test_supervisor_research_action_observation(self):
        model=Model(call("delegate",{"agent":"research","task":"Research tourism"},"h1"),
                    call("search_web",{"query":"tourism evidence"},"s1"),
                    AIMessage(content="Museum finding [S1]."),
                    AIMessage(content="Research report: Museum finding [S1]."))
        conv=Conversation()
        answer,pending=handle_multi_agent("Research tourism",conv,model,FakeSearch(),PriceStub())
        self.assertIn("Retrieved sources",answer)
        self.assertIn("https://example.org/source",answer)
        self.assertIn("Orchestrator assigned research specialist.",conv.records[0][2])
        self.assertIn("Research Agent selected tool: search_web",conv.records[0][2])
        self.assertTrue(any("search_web" in names for names in model.bound))

    def test_travel_asks_user_no_price_search(self):
        model=Model(call("delegate",{"agent":"travel","task":"Plan the trip"},"h1"),
                    call("ask_user",{"question":"Please provide destination and dates."},"q1"),
                    AIMessage(content="Please provide destination and dates."),
                    AIMessage(content="Please provide destination and dates."))
        answer,pending=handle_multi_agent("Plan the trip",Conversation(),model,FakeSearch(),PriceStub())
        self.assertIn("destination",answer)
        self.assertTrue(pending)
        self.assertTrue(any("search_travel_prices" in names for names in model.bound))
        self.assertTrue(any("lookup_airports" in names for names in model.bound))

    def test_no_handoff_is_not_fake_orchestration(self):
        with self.assertRaises(ValueError):
            handle_multi_agent("Research",Conversation(),Model(AIMessage(content="Invented report")),FakeSearch(),PriceStub())


    def test_travel_selects_airport_and_price_tools(self):
        from test_workflows import trip_request
        request=trip_request()
        request.update(departure_airport="DEL",arrival_airport="GOI,GOX",destination="Goa")
        class Prices:
            def __init__(self):self.requests=[]
            def offers(self,r):
                self.requests.append(r)
                return {"start":r["start"],"end":r["end"],"travelers":r["travelers"],
                        "nights":1,"searched_at":"test","flights":[],"hotels":[],
                        "warnings":["No usable listings."]}
        prices=Prices()
        lookup=AIMessage(content="",tool_calls=[
            {"name":"lookup_airports","args":{"city":"Delhi"},"id":"a1","type":"tool_call"},
            {"name":"lookup_airports","args":{"city":"Goa"},"id":"a2","type":"tool_call"}])
        model=Model(call("delegate",{"agent":"travel","task":"Check Delhi to Goa prices"},"h1"),
                    lookup,call("search_travel_prices",{"request_json":__import__("json").dumps(request)},"p1"),
                    AIMessage(content="No confirmed price available."),
                    AIMessage(content="Travel Agent found no usable listings."))
        conv=Conversation()
        answer,pending=handle_multi_agent("Check prices",conv,model,FakeSearch(),prices)
        self.assertEqual(len(prices.requests),1)
        self.assertEqual(prices.requests[0]["arrival_airport"],"GOI,GOX")
        self.assertIn("Travel Agent selected tool: lookup_airports",conv.records[0][2])
        self.assertIn("Travel Agent selected tool: search_travel_prices",conv.records[0][2])
        self.assertIn("No booking or payment",answer)

    def test_shared_budget(self):
        budget=CallBudget(1)
        budget.consume()
        with self.assertRaises(ValueError):budget.consume()

if __name__=="__main__":unittest.main()
