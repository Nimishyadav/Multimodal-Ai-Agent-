"""Prepare readable answers, errors and tool activity for the chat."""
import re


def public_activity(events):
    return [event for event in events if not re.search(r'Model provider used:|unavailable; trying configured backup|groq|gemini|openrouter|gpt-oss',event,re.I)]


def simple_activity(events):
    """Keep actual tool actions visible, but remove internal routing chatter."""
    visible = public_activity(events)
    has_retrieval = any(event.startswith('Document retrieval:') for event in visible)
    result = []
    actions = {
        'search_documents': 'Document search', 'inspect_table': 'Spreadsheet inspection',
        'calculate_table': 'Spreadsheet calculation', 'calculator': 'Calculator',
        'web_search': 'Web search', 'search_web': 'Web search',
        'read_webpage': 'Webpage reader', 'analyze_image': 'Image reader',
        'read_chat_history': 'Chat history', 'save_memory': 'Save memory',
        'retrieve_memory': 'Recall memory', 'delete_memory': 'Delete memory',
        'ask_user': 'Ask for details',
    }
    for event in visible:
        line = None
        if event.startswith('Upload accepted:'):
            line = 'File added: ' + event.split(':', 1)[1].strip()
        elif event.startswith('Upload processing:') and not has_retrieval:
            if 'local_fallback' in event or 'BM25' in event:
                line = 'File read using local text extraction.'
            elif 'RAG-Anything' in event:
                line = 'File indexed with MinerU and RAG-Anything.'
        elif event.startswith('Document retrieval:'):
            if 'no source evidence' in event:
                line = 'No readable document text found.'
            elif 'exact PDF page' in event:
                match = re.search(r'exact PDF page (\d+)', event)
                line = 'Read PDF page ' + match.group(1) + '.' if match else 'Read the requested PDF page.'
            elif 'FAISS vector search' in event:
                line = 'Used FAISS to search document content.'
            elif 'hybrid graph/vector context' in event:
                line = 'Used RAG-Anything to find relevant content.'
            elif 'fallback' in event:
                line = 'Used local text search; RAG-Anything retrieval unavailable.'
            else:
                line = 'Used local document search.'
        elif 'selected tool:' in event:
            agent, name = event.split(' selected tool:', 1)
            name = name.strip().split()[0]
            if name != 'delegate' and name != 'ask_user':
                line = agent + ': ' + actions.get(name, name.replace('_', ' ')) + '.'
        elif 'tool result:' in event:
            if 'waiting for user details' in event:
                line = 'More details needed from you.'
            elif event.endswith('failed.'):
                name = event.split('tool result:', 1)[1].split('—', 1)[0].strip()
                line = actions.get(name, name.replace('_', ' ')) + ' could not finish.'
        elif event == 'Orchestrator Agent completed its response.':
            line = 'Response ready.'
        if line and line not in result:
            result.append(line)
    return result


def public_answer(text):
    text = re.sub(r'All configured compatible model APIs are unavailable or cooling down\.[^\n]*', 'Image or AI analysis is temporarily unavailable. Your attachments remain in this chat; please try again later.',text)
    return '\n'.join(line for line in text.splitlines()
                     if not re.match(r'\s*(?:Model provider used:|Vision provider:)',line,re.I)).strip()


def public_error(error):
    from main import friendly_error
    message=friendly_error(error)
    if re.search(r'groq|gemini|openrouter|model APIs|model provider|model unavailable|GOOGLE_API_KEY|GROQ_API_KEY',message,re.I):
        return 'The AI service is temporarily unavailable. Please try again later or check your local configuration.'
    return message
