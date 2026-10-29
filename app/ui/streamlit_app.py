"""Display the chat and keep each browser session separate."""
import sys
from pathlib import Path

# Allow this file to run directly through Streamlit.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st
from groq import APIConnectionError, APIStatusError

from app.agent.session import Conversation
from app.history.store import HistoryStore
from app.history.migration import upgrade_history_store
from app.rag.documents import DocumentIndex, DocumentError
from app.rag.anything_rag import create_document_index, AnythingRAGIndex
from app.rag.embeddings import cloud_mode
import os
from app.tools.document_search import create_document_tool
from app.config.settings import load_settings
from app.llm.client import create_model, create_data_model
from app.attachments.migration import AttachmentStore, upgrade_attachment_store
from app.attachments.persistence import snapshot_attachments, restore_attachments
from app.ui.runtime import load_presentation
presentation = load_presentation()
public_activity = presentation.public_activity
simple_activity = presentation.simple_activity
public_answer = presentation.public_answer
public_error = presentation.public_error
from app.integrations.web_reader import WebPageClient
from app.multi_agent.system import handle_multi_agent
from app.multi_agent.errors import AgentExecutionError
from app.integrations.travel_prices import TravelPriceClient
from app.config.settings import configured_key
from app.memory.store import MemoryStore, MemoryStoreError, validate_identifier
from app.integrations.web_search import SearchClient
from app.tools.web_search import create_search_tool

st.set_page_config(page_title="AI Agent", page_icon="✦", layout="wide")
st.html("""
<style>
.block-container {max-width: 980px; padding-top: 2rem;}
.hero {padding: 1.5rem 1.7rem; border: 1px solid #334155; border-radius: 20px;
background: linear-gradient(120deg, #172554, #0f172a); color: #f8fafc; margin-bottom: 1.3rem;}
.hero h1 {font-size: 2.1rem; margin: .2rem 0;}
.hero p {color: #cbd5e1; margin: .4rem 0;}
/* Native Streamlit upload button; change its visual icon while preserving click/accessibility. */
button[data-testid="stChatInputFileUploadButton"] svg {display: none;}
button[data-testid="stChatInputFileUploadButton"]::after {content: "+"; font-size: 1.7rem; line-height: 1;}
.badge {font-size: .75rem; letter-spacing: .15em; color: #93c5fd;}
</style>
""")
st.html("""
<div class="hero">
<h1>AI Agent</h1>
<p>Ask anything</p></div>
""")

if "conversation" not in st.session_state or st.session_state.get("app_version") != 26:
    try:
        settings = load_settings()
        model = create_model(settings)
        st.session_state.memory_store = MemoryStore()
        st.session_state.history_store = HistoryStore()
        st.session_state.document_index = create_document_index()
        st.session_state.document_tool = create_document_tool(st.session_state.document_index)
        st.session_state.upload_version = 0
        st.session_state.attachment_chats = {}
        st.session_state.bound_model = model
        st.session_state.data_model = create_data_model(model)
        st.session_state.search_enabled = bool(settings.tavily_api_key or configured_key("SERPAPI_API_KEY", "SERP_API_KEY"))
        st.session_state.search_client = SearchClient(settings.tavily_api_key, configured_key("SERPAPI_API_KEY", "SERP_API_KEY"))
        st.session_state.search_tool = create_search_tool(st.session_state.search_client)
        st.session_state.price_client = TravelPriceClient(configured_key("SERPAPI_API_KEY", "SERP_API_KEY"))
        st.session_state.profile_id = "local-user"
        st.session_state.conversation = Conversation(model, memory_store=st.session_state.memory_store, user_id=st.session_state.profile_id, additional_tools=[st.session_state.search_tool, st.session_state.document_tool], history_store=st.session_state.history_store, document_index=st.session_state.document_index)
        st.session_state.app_version = 26
        st.session_state.chat = []
    except MemoryStoreError as error:
        st.error(str(error))
        st.stop()
    except ValueError:
        st.error("Check your AI service configuration in the local .env, then restart.")
        st.stop()

conversation = st.session_state.conversation
# Keep the active chat and uploads while refreshing an older cached database object.
st.session_state.history_store = upgrade_history_store(st.session_state.history_store)
conversation.history_store = st.session_state.history_store
# Restore readable attachments for this profile and chat, including after a restart.
attachment_key = st.session_state.profile_id + ":" + conversation.thread_id
if attachment_key not in st.session_state.attachment_chats:
    saved_files = st.session_state.history_store.load_attachments(st.session_state.profile_id, conversation.thread_id)
    try:
        restored_files = restore_attachments(AttachmentStore(create_document_index()), saved_files)
    except (ValueError, KeyError, TypeError):
        st.error("Saved attachments could not be restored. Please reattach the files.")
        st.stop()
    st.session_state.attachment_chats[attachment_key] = restored_files
if 'web_clients' not in st.session_state:
    st.session_state.web_clients = {}
if attachment_key not in st.session_state.web_clients:
    st.session_state.web_clients[attachment_key] = WebPageClient()
attachments = upgrade_attachment_store(st.session_state.attachment_chats[attachment_key])
if os.getenv('DOCUMENT_RAG_BACKEND','standard').lower()=='raganything' and not isinstance(attachments.documents,AnythingRAGIndex):
    upgraded_index=create_document_index()
    upgraded_index._documents.update(attachments.documents._documents)
    for field in ('backend_status', '_jobs', '_rag_queries'):
        if hasattr(attachments.documents, field):
            setattr(upgraded_index, field, getattr(attachments.documents, field))
    attachments.documents=upgraded_index
st.session_state.attachment_chats[attachment_key] = attachments
if attachments.items and not st.session_state.history_store.load_attachments(st.session_state.profile_id, conversation.thread_id):
    st.session_state.history_store.save_attachments(st.session_state.profile_id, conversation.thread_id, snapshot_attachments(attachments))
st.session_state.document_index = attachments.documents
st.session_state.document_tool = create_document_tool(attachments.documents)
conversation.document_index = attachments.documents
conversation.tools = [t for t in conversation.tools if t.name != "search_documents"] + [st.session_state.document_tool]
with st.sidebar:
    st.title("AI Agent Lab")
    st.caption("Document questions use RAG-Anything. External facts use internet search.")
    st.divider()
    st.caption("Web search: enabled" if st.session_state.search_enabled else "Web search: missing Tavily/SerpApi key")
    if isinstance(attachments.documents,AnythingRAGIndex):
        st.caption('Multimodal document knowledge: enabled')
        cloud_approved = st.checkbox(
            "Allow document uploads to MinerU Cloud and text embeddings to Google",
            value=os.getenv('RAG_CLOUD_UPLOAD_APPROVED','false').lower()=='true', key='mineru_cloud_consent_v2_' + attachment_key,
            help="MinerU receives the complete file; Google receives extracted text for embeddings. Cloud mode does not silently start local AI models. Keep unchecked if these uploads are not approved.")
        attachments.documents.client.cloud_upload_approved = cloud_approved
        if cloud_approved and not configured_key('MINERU_API_TOKEN'):
            st.warning('Add MINERU_API_TOKEN to local .env before using cloud parsing.')

        if cloud_mode() or os.getenv('RAG_EMBEDDING_PROVIDER','gemini')=='gemini':
            st.caption('Cloud document processing: parsing and embedding inference use APIs. Old indexes need a one-time rebuild for the selected search mode.')
        states=getattr(attachments.documents,'backend_status',{})
        if states:
            with st.expander('Document processing status'):
                for filename,status in states.items():
                    st.write(filename)
                    st.caption(status.get('parser','')+' / '+status.get('retrieval',''))
                    st.caption(status.get('message',''))
                if attachments.documents._jobs:
                    if st.button('Rebuild document indexes', disabled=not cloud_approved):
                        with st.spinner('Building document search index…'):
                            attachments.documents.retry_incomplete_indexes(rebuild_all=True)
                            st.session_state.history_store.save_attachments(st.session_state.profile_id, conversation.thread_id, snapshot_attachments(attachments))
                        st.rerun()
                if any(state.get('retrieval')=='parsed_only' for state in states.values()):
                    if st.button('Retry incomplete knowledge indexes'):
                        with st.spinner('Retrying knowledge indexing…'):
                            attachments.documents.retry_incomplete_indexes()
                        st.rerun()
    st.write("**Session**")
    st.caption(conversation.thread_id[:12])
    st.info("Chats are saved automatically. Relevant previous chats are recalled across conversations in the same profile. Readable attachments are saved locally with this chat and restored when you reopen it. Older chats may need their files attached once again.")
    if st.button("New chat", type="primary", width="stretch"):
        conversation.clear()
        st.session_state["saved_chat_" + st.session_state.profile_id] = None
        st.session_state.chat = []
        st.rerun()
    st.divider()
    if st.button("Clear screen"):
        st.session_state.chat = []
        st.rerun()
    st.caption("Clear screen hides the display only; saved history remains.")
    st.subheader("Previous chats")
    try:
        history = st.session_state.history_store
        chats = history.list_chats(st.session_state.profile_id)
        labels = {chat["id"]: chat["title"] + " · " + chat["updated_at"] for chat in chats}
        selected_chat = st.selectbox(
            "Saved conversation", [chat["id"] for chat in chats],
            format_func=lambda value: labels[value],
            index=None, key="saved_chat_" + st.session_state.profile_id,
            placeholder="Choose a previous conversation to open it",
        )
        open_selected = st.button("Open selected chat", disabled=selected_chat is None)
        if selected_chat and (selected_chat != conversation.thread_id or open_selected):
            restored = Conversation(
                st.session_state.bound_model, memory_store=st.session_state.memory_store,
                user_id=st.session_state.profile_id,
                additional_tools=[st.session_state.search_tool, st.session_state.document_tool],
                history_store=history, chat_id=selected_chat, document_index=st.session_state.document_index,
            )
            restored_chat = history.transcript(st.session_state.profile_id, selected_chat)
            conversation.close()
            st.session_state.conversation = restored
            st.session_state.chat = restored_chat
            st.rerun()
        with st.form("rename_chat"):
            title = st.text_input("Chat title", value=history.get_chat(st.session_state.profile_id, conversation.thread_id)["title"], max_chars=100)
            if st.form_submit_button("Rename chat"):
                history.rename(st.session_state.profile_id, conversation.thread_id, title)
                st.rerun()
        st.download_button("Export current chat (.md)",
                           history.export_markdown(st.session_state.profile_id, conversation.thread_id),
                           file_name="conversation.md", mime="text/markdown")
        confirmed = st.checkbox("Confirm deletion of the selected saved chat")
        if st.button("Delete selected chat", disabled=not confirmed or selected_chat is None):
            history.delete(st.session_state.profile_id, selected_chat)
            st.session_state.attachment_chats.pop(st.session_state.profile_id + ":" + selected_chat, None)
            conversation.checkpointer.delete_thread(selected_chat)
            if selected_chat == conversation.thread_id:
                conversation.clear()
                st.session_state.chat = []
            st.rerun()
    except (MemoryStoreError, ValueError):
        st.error("Saved chat could not be loaded. Please try again.")

    st.divider()
    st.subheader("Attached files")
    st.caption("Use the attachment button inside the chat. Up to 10 files/chat (5 text documents), 10 MB/file.")
    st.caption("PDF, Word DOCX, PowerPoint PPTX, spreadsheets, text/code and common images. First worksheet / image frame only.")
    for item in attachments.manifest():
        st.write(item['name'] + " · " + item['kind'])
        st.caption(item.get('coverage', item.get('reader', '')))
    if st.button("Clear attachments"):
        attachments.clear()
        st.session_state.history_store.save_attachments(st.session_state.profile_id, conversation.thread_id, snapshot_attachments(attachments))
        st.rerun()
    st.caption("Documents extract locally; selected text goes to the chat model. Table calculations run locally; relevant metadata/results go to the AI service. Image questions send a resized image to the configured vision service.")

    st.subheader("Saved preferences")
    st.caption("Local profiles are not authenticated accounts. Saved preferences are sent to the selected LLM provider when you chat.")
    with st.form("profile_form"):
        profile = st.text_input("Profile ID", value=st.session_state.profile_id, max_chars=64)
        if st.form_submit_button("Use profile"):
            try:
                profile = validate_identifier(profile, "Profile ID")
                if profile != st.session_state.profile_id:
                    conversation.close()
                    # Load the selected profile's attachments on the next rerun.
                    st.session_state.upload_version += 1
                    st.session_state.profile_id = profile
                    st.session_state.conversation = Conversation(
                        st.session_state.bound_model,
                        memory_store=st.session_state.memory_store, user_id=profile,
                        additional_tools=[st.session_state.search_tool, st.session_state.document_tool],
                        history_store=st.session_state.history_store,
                        document_index=st.session_state.document_index,
                    )
                    st.session_state.chat = []
                    st.rerun()
            except MemoryStoreError as error:
                st.error(str(error))

    with st.form("save_preference", clear_on_submit=True):
        key = st.text_input("Preference key", placeholder="preferred_language", max_chars=64)
        value = st.text_input("Preference value", placeholder="Python", max_chars=500)
        st.caption("Save useful preferences only. Do not store passwords, API keys or sensitive information.")
        if st.form_submit_button("Save preference"):
            try:
                st.session_state.memory_store.save(st.session_state.profile_id, key, value)
                st.success("Preference saved.")
            except MemoryStoreError as error:
                st.error(str(error))

    try:
        preferences = st.session_state.memory_store.list_preferences(st.session_state.profile_id)
        if preferences:
            st.dataframe(
                [{"Key": p.key, "Value": p.value} for p in preferences],
                hide_index=True, width="stretch",
            )
            selected = st.selectbox("Preference to delete", [p.key for p in preferences])
            if st.button("Delete selected preference"):
                st.session_state.memory_store.delete(st.session_state.profile_id, selected)
                # Remove saved context that could bring back the deleted preference.
                conversation.clear()
                st.session_state.chat = []
                st.rerun()
        else:
            st.caption("No saved preferences for this profile.")
    except MemoryStoreError as error:
        st.error(str(error))

    with st.expander("How this works"):
        st.write("Question → Orchestrator → specialist agent → selected tools → observations → answer.")
        st.write("SQLite saves chat history and preferences. Document retrieval uses the configured local or multimodal knowledge backend.")
        st.write("Activity shows tool actions, never hidden reasoning.")

table_frame = next(iter(attachments.tables.values())) if len(attachments.tables) == 1 else None
pending_key = "chat_task_" + st.session_state.profile_id + "_" + conversation.thread_id

if not st.session_state.chat:
    col1, col2 = st.columns(2)
    col1.info("Try maths\n\nCalculate (125 * 37) + 450 / 9.")
    col2.info("Try memory\n\nRemember that I prefer Python. Then ask: What language do I prefer?")

for entry in st.session_state.chat:
    with st.chat_message(entry["role"]):
        if entry.get("created_at"):
            st.caption(entry["created_at"] + " UTC")
        st.markdown(public_answer(entry["content"]))
        if entry.get("sources"):
            with st.expander("Sources used by the agent", expanded=False):
                for source in entry["sources"]:
                    if source.get("type") in {"pdf", "document"}:
                        st.write(f"{source['document']} — {'page' if source.get('type') == 'pdf' else source.get('location', 'section')} {source['page']}")
                        st.caption(source.get("excerpt", ""))
                    else:
                        st.link_button(source["title"], source["url"])
                        st.caption("Cached result" if source.get("cached", False) else "Fresh search result")
        if entry.get("activity"):
            if any('selected tool: search_documents' in item for item in entry['activity']) and not any(item.startswith('Document retrieval:') for item in entry['activity']):
                st.caption('Older chat: retrieval details were not recorded.')
            with st.expander("Agents and tools used", expanded=True):
                for event in simple_activity(entry["activity"]):
                    st.write(event)

submission = st.chat_input(
    "Ask anything or attach files…", max_chars=4000,
    accept_file="multiple", file_type=None, max_upload_size=10,
)
if submission:
    question = submission if isinstance(submission, str) else submission.text
    uploads = [] if isinstance(submission, str) else submission.files
    errors = []
    accepted = []
    upload_activity = []
    for upload in uploads:
        try:
            with st.spinner("Reading " + upload.name + "…"):
                attachments.add(upload.name, upload.getvalue())
            st.session_state.history_store.save_attachments(st.session_state.profile_id, conversation.thread_id, snapshot_attachments(attachments))
            accepted.append(upload.name)
            upload_activity.append('Upload accepted: '+upload.name)
            entry=attachments.items.get(upload.name,{})
            if entry.get('kind')=='document':
                summary=next((item for item in attachments.documents.documents() if item['document']==upload.name),{})
                if summary.get('chunks'):
                    st.caption(f"{upload.name}: indexed {summary['chunks']} text chunks. Only relevant excerpts are sent to the model.")
            backend=getattr(attachments.documents,'backend_status',{}).get(upload.name)
            if backend:
                upload_activity.append('Upload processing: '+upload.name+' — '+backend.get('parser','')+' / '+backend.get('retrieval',''))
                st.caption(upload.name+': '+backend.get('parser','')+' / '+backend.get('retrieval',''))
                if backend.get('retrieval') != 'RAG-Anything':
                    st.warning(upload.name + ': ' + backend.get('message','Multimodal indexing unavailable.') + ' Readable local text remains available. Reattach this file to retry cloud processing.')
        except DocumentError as error:
            errors.append(upload.name + ": " + str(error))
    for error in errors:
        st.error(error)
    if uploads and not accepted:
        st.info("No new file was read. Fix the attachment and submit again; the model has not analyzed it.")
        st.stop()
    question = (question or "").strip()
    if not question:
        kinds = {attachments.items[name]['kind'] for name in accepted}
        tasks = []
        if 'document' in kinds: tasks.append('Summarize the attached document')
        if 'image' in kinds: tasks.append('Describe the attached image')
        if 'table' in kinds: tasks.append('Inspect the attached spreadsheet and count its rows')
        question = '; '.join(tasks) or 'What can you help with?'
    if errors:
        question += "\n\nFiles NOT read: " + "; ".join(errors)
    if accepted:
        question += "\n\nAttached files: " + ", ".join(accepted)
    table_frame = next(iter(attachments.tables.values())) if len(attachments.tables) == 1 else None
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Agent is working…"):
                answer, next_pending = handle_multi_agent(
                    question, conversation, st.session_state.bound_model,
                    st.session_state.search_client, st.session_state.price_client,
                    frame=table_frame, pending=st.session_state.get(pending_key), data_model=st.session_state.data_model, attachments=attachments, economy_mode=True, web_client=st.session_state.web_clients[attachment_key], initial_activity=upload_activity,
                )
                if answer is None:
                    answer = conversation.ask(question)
                st.session_state[pending_key] = next_pending
            st.markdown(public_answer(answer))
            with st.expander("Agents and tools used", expanded=True):
                for event in simple_activity(conversation.activity):
                    st.write(event)
            st.session_state.chat = st.session_state.history_store.transcript(st.session_state.profile_id, conversation.thread_id)
            st.rerun()
        except MemoryStoreError as error:
            st.error(str(error))
        except AgentExecutionError as error:
            st.error(public_error(error))
        except (APIStatusError, APIConnectionError, ValueError) as error:
            st.error(public_error(error))
        except Exception:
            st.error(public_error(sys.exception()))