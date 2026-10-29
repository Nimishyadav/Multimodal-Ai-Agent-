"""Summarize tool actions without displaying prompts, credentials or raw responses."""
import json


def tool_result_activity(agent, tool_name, result, document_details=False):
    content = getattr(result, 'content', '')
    if not isinstance(content, str):content = ''
    try:data = json.loads(content)
    except (ValueError, TypeError):data = {}
    if not isinstance(data, dict):data = {}
    failed = getattr(result, 'status', None) == 'error' or data.get('status') in {'error','failed'} or content.startswith(('Tool error:', 'Unknown tool.', 'Tool failed.', 'Tool could not complete'))
    state = 'failed' if failed else 'waiting for user details' if data.get('status') == 'needs_user' else 'returned output'
    events = [f'{agent} tool result: {tool_name} — {state}.']
    if failed:return events
    if tool_name == 'search_documents' and document_details:
        from app.rag.retrieval_status import document_retrieval_activity
        events.append(document_retrieval_activity(data))
    elif tool_name == 'inspect_table' and 'tables' in data:
        events.append(f"Table inspection: {len(data['tables'])} uploaded table(s) inspected.")
    elif tool_name == 'calculate_table' and 'result_preview' in data:
        events.append(f"Table calculation: Pandas returned {data.get('result_rows',len(data['result_preview']))} result row(s).")
    elif tool_name in {'search_web','web_search'} and 'sources' in data:
        events.append(f"Web search: {len(data['sources'])} source(s) returned.")
    elif tool_name == 'read_webpage' and data.get('status') == 'ok':
        events.append('Webpage reader: public page text retrieved.')
    elif tool_name == 'analyze_image':
        events.append('Image analysis: no verified result available.' if 'unavailable' in content.lower() else 'Image analysis: vision tool returned an observation.')
    return events
