import json
import unittest
from datetime import date, timedelta
from unittest.mock import patch
from app.integrations.travel_prices import TravelPriceClient, TravelSearchError, format_offers, public_link

class Response:
    def __init__(self, payload):
        self.raw=json.dumps(payload).encode()
    def __enter__(self): return self
    def __exit__(self,*args): return False
    def read(self,limit): return self.raw[:limit]

class TravelPricesTests(unittest.TestCase):
    def request(self):
        start=date.today()+timedelta(days=10)
        return {"departure_airport":"DEL","arrival_airport":"JAI","destination":"Jaipur",
                "start":start.isoformat(),"end":(start+timedelta(days=2)).isoformat(),"travelers":2}

    def test_city_autocomplete_returns_multiple_airports(self):
        client=TravelPriceClient("test-placeholder")
        with patch.object(client,"_search",return_value={"suggestions":[
            {"name":"Paris, France","type":"city","airports":[{"id":"CDG"},{"id":"ORY"}]}]}):
            self.assertEqual(client.resolve_airports("Paris, France"),"CDG,ORY")

    def test_goa_multi_airport_search(self):
        client=TravelPriceClient("test-placeholder")
        request=self.request()
        request["arrival_airport"]="GOI,GOX"
        request["destination"]="Goa"
        with patch.object(client,"_search",return_value={}) as search:
            result=client.offers(request)
        self.assertEqual(search.call_args_list[0].args[0]["arrival_id"],"GOI,GOX")
        self.assertEqual(result["arrival_airports"],"GOI,GOX")

    def test_listings_cache_and_budget_units(self):
        payloads=[Response({"best_flights":[{"price":2000,"flights":[{"airline":"Test Airline"}]}]}),
                  Response({"properties":[{"name":"Test Hotel","total_rate":{"extracted_lowest":3000},
                                           "rate_per_night":{"extracted_lowest":1500},
                                           "link":"https://hotel.example.org"}]})]
        with patch("app.integrations.travel_prices.urlopen",side_effect=payloads) as network:
            client=TravelPriceClient("test-placeholder")
            results=client.offers(self.request())
            client.offers(self.request())
            self.assertEqual(network.call_count,2)
        self.assertEqual(results["hotels"][0]["listed_inr"],3000)
        answer=format_offers(results,10000)
        self.assertIn("Combined estimate: INR 7,000.00",answer)
        self.assertIn("NOT a group quote",answer)

    def test_no_nightly_fallback(self):
        with patch("app.integrations.travel_prices.urlopen",side_effect=[
            Response({}),Response({"properties":[{"name":"Hotel","rate_per_night":{"extracted_lowest":1000}}]})]):
            result=TravelPriceClient("test-placeholder").offers(self.request())
        self.assertEqual(result["hotels"],[])

    def test_missing_key_never_calls_network(self):
        with patch("app.integrations.travel_prices.urlopen") as network:
            result=TravelPriceClient(None).offers(self.request())
            self.assertTrue(result["warnings"])
            network.assert_not_called()

    def test_bad_dates_and_codes_no_network(self):
        request=self.request()
        request["arrival_airport"]="BAD CODE"
        with patch("app.integrations.travel_prices.urlopen") as network:
            with self.assertRaises(TravelSearchError):
                TravelPriceClient("test-placeholder").offers(request)
            network.assert_not_called()

    def test_partial_failure_retains_hotels(self):
        with patch("app.integrations.travel_prices.urlopen",side_effect=[
            TimeoutError(),Response({"properties":[{"name":"Hotel","total_rate":{"extracted_lowest":1000}}]})]):
            result=TravelPriceClient("test-placeholder").offers(self.request())
        self.assertEqual(len(result["hotels"]),1)
        self.assertEqual(len(result["warnings"]),1)

    def test_credential_links_rejected(self):
        self.assertEqual(public_link("https://serpapi.com/search.json?api_key=SECRET"),"")
        self.assertEqual(public_link("https://example.org/?api_key=SECRET"),"")
        self.assertEqual(public_link("javascript:alert(1)"),"")

if __name__=="__main__": unittest.main()
