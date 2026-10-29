"""Provide the earlier research and travel panel."""
from datetime import date, timedelta
import streamlit as st
from app.workflows.assistants import build_workflow
from app.integrations.travel_prices import TravelPriceClient, TravelSearchError, format_offers
from app.config.settings import configured_key


def render_assistant_panel(model, search_client, conversation, history_store, user_id):
    with st.expander("Research Report Assistant / Trip Planner", expanded=False):
        st.caption("LangGraph: coordinator → web evidence → specialist draft → validation. Up to 2 Tavily searches and 1 model call per run.")
        mode = st.radio("Assistant mode", ["Research Report", "Trip Planner"], horizontal=True)
        request = None
        price_request = None
        if mode == "Research Report":
            with st.form("research_report_form"):
                topic = st.text_input("Research topic", placeholder="Benefits and limitations of rooftop solar", max_chars=300)
                focus = st.text_area("Focus or questions (optional)", max_chars=500)
                submitted = st.form_submit_button("Generate research report")
            if submitted:
                request = {"mode": "research", "topic": topic, "focus": focus}
        else:
            with st.form("trip_planner_form"):
                origin = st.text_input("Starting city", max_chars=150)
                destination = st.text_input("Destination", max_chars=150)
                start = st.date_input("Start date", value=date.today() + timedelta(days=7), min_value=date.today())
                end = st.date_input("End date", value=date.today() + timedelta(days=9), min_value=date.today())
                travelers = st.number_input("Travelers", min_value=1, max_value=20, value=1, step=1)
                budget = st.number_input("Total group budget in INR (entire trip)", min_value=1, max_value=10000000, value=15000, step=100)
                interests = st.text_input("Interests or travel preferences", placeholder="History, vegetarian food, relaxed pace", max_chars=500)
                st.caption("1–7 days. Itinerary budget is an allowance. Search listed flight/hotel prices separately below.")
                departure_airport = st.text_input("Departure airport code (price search)", placeholder="DEL", max_chars=3)
                arrival_airport = st.text_input("Arrival airport code (price search)", placeholder="JAI", max_chars=3)
                st.caption("Price search: round-trip flight fare for one adult + one-room hotel search for all travelers. 1–6 adults, at least one night. Up to 2 SerpApi searches; cached results may be returned.")
                submitted = st.form_submit_button("Generate trip plan")
                prices_submitted = st.form_submit_button("Search flight and hotel prices")
            if submitted:
                request = {"mode": "trip", "origin": origin, "destination": destination,
                           "start": start.isoformat(), "end": end.isoformat(),
                           "travelers": int(travelers), "budget_inr": int(budget), "interests": interests}
            if prices_submitted:
                price_request = {"departure_airport": departure_airport, "arrival_airport": arrival_airport,
                                 "destination": destination.strip(), "start": start.isoformat(), "end": end.isoformat(),
                                 "travelers": int(travelers)}
        key = "assistant_download_" + user_id
        if price_request is not None:
            st.session_state.pop(key, None)
            try:
                if not price_request["destination"]:
                    raise TravelSearchError("Enter a destination city for hotel search.")
                client_key = "travel_price_client_" + user_id
                if client_key not in st.session_state:
                    st.session_state[client_key] = TravelPriceClient(configured_key("SERPAPI_API_KEY", "SERP_API_KEY"))
                with st.spinner("Searching flight and hotel listings…"):
                    offers = st.session_state[client_key].offers(price_request)
                if not offers["flights"] and not offers["hotels"]:
                    for warning in offers["warnings"]:
                        st.warning(warning)
                else:
                    answer = format_offers(offers, int(budget))
                    activity = ["SerpApi price listings retrieved; no LLM call, booking or payment."]
                    conversation.record_workflow(
                        f"Price search: {departure_airport} to {arrival_airport}, {start} to {end}",
                        answer, activity, [])
                    st.session_state.chat = history_store.transcript(user_id, conversation.thread_id)
                    st.session_state[key] = (answer, activity, conversation.thread_id)
                    st.rerun()
            except TravelSearchError as error:
                st.error(str(error))
            except Exception:
                st.error("Could not finish the price search. Check SerpApi access and retry.")
        if request is not None:
            st.session_state.pop(key, None)
            try:
                with st.spinner("Gathering evidence and building your result…"):
                    result = build_workflow(model, search_client).invoke({"request": request})
                if not result["sources"]:
                    st.warning(result["answer"])
                    for warning in result["warnings"]:
                        st.caption(warning)
                else:
                    question = ("Research report: " + request["topic"] if request["mode"] == "research"
                                else f"Trip plan: {request['origin']} to {request['destination']}, {request['start']} to {request['end']}")
                    conversation.record_workflow(question, result["answer"], result["activity"], result["sources"])
                    st.session_state.chat = history_store.transcript(user_id, conversation.thread_id)
                    st.session_state[key] = (result["answer"], result["activity"], conversation.thread_id)
                    st.rerun()
            except ValueError as error:
                st.error(str(error))
            except Exception:
                st.error("The assistant could not complete this run. Check provider access/quota or retry later. No incomplete report was saved.")
        saved = st.session_state.get(key)
        if saved and saved[2] == conversation.thread_id:
            st.success("Latest assistant result is saved in this conversation history.")
            st.download_button("Download latest assistant result (.md)", saved[0],
                               file_name="assistant_result.md", mime="text/markdown")
            with st.expander("Workflow activity"):
                for event in saved[1]:
                    st.write(event)
