"""Look up airports and travel prices through SerpApi; no bookings are made."""
import json
import math
import re
import time
from datetime import date, datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import urlopen
from app.integrations.web_search import safe_source_url


class TravelSearchError(ValueError):
    """Safe error without request URLs, response bodies or credentials."""


def price(value):
    return float(value) if type(value) in (int, float) and math.isfinite(value) and value > 0 else None


def public_link(value):
    if not safe_source_url(value):
        return ""
    parsed = urlsplit(value)
    if "serpapi.com" in parsed.hostname.lower() or re.search(r"api.?key|token=", value, re.I):
        return ""
    return value


class TravelPriceClient:
    def __init__(self, api_key):
        self._api_key = api_key
        self._cache = {}

    def _search(self, parameters):
        if not self._api_key:
            raise TravelSearchError("Add SERPAPI_API_KEY in your local .env and restart Streamlit.")
        cache_key = json.dumps(parameters, sort_keys=True)
        cached = self._cache.get(cache_key)
        if cached and time.monotonic() - cached[0] < 600:
            return cached[1]
        # Do not log the full URL because it contains the SerpApi key.
        url = "https://serpapi.com/search.json?" + urlencode({**parameters, "api_key": self._api_key})
        try:
            with urlopen(url, timeout=25) as response:
                raw = response.read(3000001)
            if len(raw) > 3000000:
                raise TravelSearchError("Travel search response exceeded the size limit.")
            data = json.loads(raw)
        except HTTPError as error:
            status = error.code
            error.close()
            raise TravelSearchError({401: "Invalid SerpApi key.", 403: "SerpApi access denied.",
                                    429: "SerpApi quota/rate limit reached."}.get(
                                        status, "SerpApi could not complete this search.")) from None
        except (URLError, TimeoutError, OSError):
            raise TravelSearchError("Travel search timed out or could not connect.") from None
        except (ValueError, UnicodeError):
            raise TravelSearchError("Travel search returned invalid data.") from None
        if not isinstance(data, dict) or data.get("error"):
            raise TravelSearchError("No travel result returned. Check dates, airport codes and SerpApi access.")
        if len(self._cache) >= 20:
            self._cache.pop(next(iter(self._cache)))
        self._cache[cache_key] = (time.monotonic(), data)
        return data

    def resolve_airports(self, city):
        """Unknown cities use provider autocomplete, never LLM-invented IATA codes."""
        city = str(city).strip()
        if not city or len(city) > 150:
            raise TravelSearchError("Please provide the city and country.")
        data = self._search({"engine": "google_flights_autocomplete", "q": city,
                             "hl": "en", "gl": "in", "exclude_regions": "true"})
        city_name = city.split(",")[0].strip().lower()
        matches = [item for item in data.get("suggestions", [])
                   if isinstance(item, dict) and item.get("type") == "city"
                   and str(item.get("name", "")).lower().split(",")[0].strip() == city_name]
        if len(matches) != 1:
            raise TravelSearchError("Airport location is ambiguous or unavailable. Please specify the city and country, or an airport name.")
        airports = [item for item in matches[0].get("airports", [])
                    if isinstance(item, dict) and re.fullmatch("[A-Z]{3}", str(item.get("id", "")))]
        if not airports:
            raise TravelSearchError("No airport found for this city. Please provide a nearby city or airport name.")
        return ",".join(dict.fromkeys(item["id"] for item in airports[:3]))

    def offers(self, request):
        start, end = date.fromisoformat(request["start"]), date.fromisoformat(request["end"])
        if start < date.today() or end <= start:
            raise TravelSearchError("Price search requires a future/current check-in and a later check-out date.")
        source, destination = request["departure_airport"].strip().upper(), request["arrival_airport"].strip().upper()
        if not re.fullmatch(r"[A-Z]{3}(?:,[A-Z]{3}){0,2}", source) or not re.fullmatch(r"[A-Z]{3}(?:,[A-Z]{3}){0,2}", destination) or set(source.split(",")) & set(destination.split(",")):
            raise TravelSearchError("Enter different 3-letter airport codes, for example DEL and JAI.")
        adults = request["travelers"]
        if type(adults) is not int or not 1 <= adults <= 6:
            raise TravelSearchError("This first price-search version supports 1–6 adult travelers in one room.")
        results = {"flights": [], "hotels": [], "warnings": [],
                   "searched_at": datetime.now(timezone.utc).isoformat(),
                   "nights": (end-start).days, "travelers": adults,
                   "start": request["start"], "end": request["end"], "departure_airports": source, "arrival_airports": destination}
        for kind in ("flights", "hotels"):
            params = {"engine": "google_" + kind, "currency": "INR", "hl": "en", "gl": "in"}
            if kind == "flights":
                params.update(departure_id=source, arrival_id=destination, outbound_date=request["start"],
                              return_date=request["end"], type=1, adults=1)
            else:
                params.update(q=request["destination"] + " hotels", check_in_date=request["start"],
                              check_out_date=request["end"], adults=adults)
            try:
                data = self._search(params)
                if data.get("search_parameters", {}).get("currency", "INR") != "INR":
                    raise TravelSearchError("Provider returned a different currency; no budget comparison was made.")
                fetched = data.get("search_metadata", {}).get("created_at") or results["searched_at"]
                if kind == "flights":
                    for row in (data.get("best_flights", []) + data.get("other_flights", [])):
                        amount = price(row.get("price"))
                        legs = row.get("flights") or []
                        if amount is None or not legs:
                            continue
                        results["flights"].append({
                            "description": " / ".join(str(leg.get("airline", "Airline")) for leg in legs),
                            "listed_inr": amount, "basis": "Round-trip search fare for ONE adult; not a confirmed group quote",
                            "link": public_link(data.get("search_metadata", {}).get("google_flights_url", ""))
                                    or "https://www.google.com/travel/flights?" + urlencode({"q": f"{source} to {destination} {request['start']} return {request['end']}"}),
                            "retrieved_at": fetched})
                else:
                    rows = data.get("properties") or ([data] if data.get("name") else [])
                    for row in rows:
                        amount = price((row.get("total_rate") or {}).get("extracted_lowest"))
                        if amount is None:
                            continue
                        results["hotels"].append({
                            "description": str(row.get("name") or "Hotel"),
                            "listed_inr": amount, "basis": f"Reported total stay, {results['nights']} nights; one-room search for {adults} adults",
                            "link": public_link(row.get("link", "")) or "https://www.google.com/travel/search?" + urlencode({"q": str(row.get("name") or request["destination"]) + " hotels"}),
                            "retrieved_at": fetched})
                results[kind] = sorted(results[kind], key=lambda row: row["listed_inr"])[:3]
                if not results[kind]:
                    results["warnings"].append(f"No usable {kind} prices returned for these parameters.")
            except TravelSearchError as error:
                results["warnings"].append(str(error))
        return results


def format_offers(results, budget):
    lines = ["# Flight and Hotel Price Listings", "",
             f"**Dates:** {results['start']} to {results['end']} | **Adults:** {results['travelers']}",
             f"**Airports searched:** {results.get('departure_airports', '')} → {results.get('arrival_airports', '')}",
             f"**Search run UTC:** {results['searched_at']}",
             "Provider results can be cached. These are listings, not reserved availability or confirmed bookings.", ""]
    for kind in ("flights", "hotels"):
        lines.append("## " + kind.title())
        if not results[kind]:
            lines.append("No usable prices available.")
        for row in results[kind]:
            lines.extend([f"- **{row['description']}** — INR {row['listed_inr']:,.2f}",
                          "  " + row["basis"] + "; retrieved " + row["retrieved_at"]])
            if row["link"]:
                lines.append("  [Check provider/search page](" + row["link"] + ")")
    if results["flights"] and results["hotels"]:
        flight_estimate = results["flights"][0]["listed_inr"] * results["travelers"]
        hotel = results["hotels"][0]["listed_inr"]
        subtotal = flight_estimate + hotel
        lines.extend(["", "## Illustrative budget comparison",
                      f"- Flight estimate: one-adult listed fare × {results['travelers']} = INR {flight_estimate:,.2f}",
                      f"- One-room reported stay total: INR {hotel:,.2f}",
                      f"- Combined estimate: INR {subtotal:,.2f}",
                      f"- Your budget: INR {budget:,.2f}",
                      f"- Budget minus this estimate: INR {budget-subtotal:,.2f}",
                      "This is NOT a group quote. Group fares/room occupancy may differ. Food, local transport, "
                      "attractions, baggage and additional taxes/fees may be missing. No claim of trip affordability."])
    lines.extend(["", "Confirm pricing basis, taxes, room occupancy and availability with the seller before payment.",
                  "No booking or payment was performed."])
    lines.extend("- " + warning for warning in results["warnings"])
    return "\n".join(lines)
