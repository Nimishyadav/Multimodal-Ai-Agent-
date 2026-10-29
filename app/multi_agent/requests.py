"""Read page, output-length and file requirements from a question."""
import re
from app.rag.intents import is_document_overview


def user_text(question):
    return re.split(r"\n\n(?:Attached files|Files NOT read):", question, maxsplit=1)[0].strip()


def strict_document_scope(question):
    text=user_text(question).lower()
    return bool(re.search(r'\b(?:only|solely|strictly|sirf)\b.*\b(?:pdf|document|file|uploaded)\b|\b(?:according to|as per)\s+(?:(?:the|this|uploaded|attached)\s+)?(?:pdf|document|file)',text))


def external_requested(question):
    text=user_text(question).lower()
    return bool(re.search(r'\b(?:internet|online|latest|current|today|news|web search|search the web)\b|(?:document|pdf|file)\s+(?:se\s+)?(?:bahar|outside)|\boutside\b.*\b(?:document|pdf|file)\b|\bignore\b.*\b(?:document|pdf|file)\b',text))


def external_factual_question(question):
    text=user_text(question).lower()
    if strict_document_scope(question):return False
    if re.search(r'\d\s*[+*/%\-]\s*\d',text):return False
    if re.search(r'\b(?:my (?:name|age|address|phone|email|password|preferences)|about me|remember|preference|previous chat|our chat|calculate|calculation)\b',text):return False
    if re.search(r'\b(?:write|draft|translate|rewrite|create|generate)\b',text) and not external_requested(question):return False
    return external_requested(question) or bool(re.match(r'(?:please\s+)?(?:what|who|where|when|why|how|explain|tell me|give me information)\b',text))


def wants_one_line(question):
    return bool(re.search(r"\b(?:one|single|ek|1)\s+(?:line|sentence)\b", user_text(question),re.I))


def wants_calculation(question):
    return bool(re.search(r"\b(?:calculate|calculation|sum|total|average|mean|minimum|maximum|min|max|count|add|subtract|multiply|divide)\b|\b(?:kitna|jodo|jod)\b",user_text(question),re.I))


def file_roles(question, documents, tables, images):
    text=user_text(question)
    roles=[]
    if is_document_overview(text) or (documents and re.search(r'(?:document|report|presentation)',text,re.I)):
        roles.append('documents')
    if not roles and documents and not tables and not images and re.search(r'(?:explain|summari[sz]e|describe)\s+(?:it|this|that)',text,re.I):
        roles.append('documents')
    all_files=bool(re.search(r"\b(?:both|all|dono|saari|saare|sab)\b.*\b(?:files?|documents?|uploads?)\b|\b(?:files?|documents?)\b.*\b(?:both|all|dono)\b",text,re.I))
    if re.search(r"\b(?:pdf|docx|pptx|slides)\b|\bword\s+(?:file|document)\b|\b(?:uploaded|attached)\s+document\b",text,re.I):roles.append('documents')
    if re.search(r"\b(?:excel|csv|xlsx|xls|spreadsheet|tsv|table)\b",text,re.I):roles.append('data')
    if images and re.search(r"\b(?:image|photo|picture|screenshot|chart)\b",text,re.I):roles.append('images')
    if len(images)==1 and re.search(r'\b(?:colou?r|rang|dimensions|resolution)\b',text,re.I):roles.append('images')
    if all_files:
        if documents:roles.append('documents')
        if tables:roles.append('data')
        if images:roles.append('images')
    if tables and wants_calculation(text) and re.search(r"tax|column|sales|rows|table|excel",text,re.I):roles.append('data')
    if not roles and re.search(r"first\s*page|page\s*\d+",text,re.I) and documents:roles.append('documents')
    if external_requested(question) and not strict_document_scope(question):
        if re.search(r'\b(?:ignore|outside|unrelated)\b.*\b(?:pdf|document|file)\b|(?:document|pdf|file)\s+(?:se\s+)?(?:bahar|outside)',text,re.I):
            roles=[role for role in roles if role!='documents']
    return list(dict.fromkeys(roles))


def compact_results(completed):
    return [{**item,'result':item['result'][:800]} for item in completed[-3:]]


def requested_lines(question):
    match=re.search(r"\b(\d+|one|two|three|ek|do|teen)\s+lines?\b",user_text(question),re.I)
    if not match:return 1 if wants_one_line(question) else None
    value=match.group(1).lower()
    return int(value) if value.isdecimal() else {'one':1,'two':2,'three':3,'ek':1,'do':2,'teen':3}[value]


def page_title(passages):
    """Extract a visible title prefix; otherwise label the result as an excerpt."""
    text=' '.join(passages[0]['excerpt'].split())
    match=re.match(r'(.{1,140}?)\s+(?:Release|Version)\b',text,re.I)
    if match:return match.group(1).strip()
    return 'Page begins: “'+text[:120].rstrip()+'”'


def document_targets(question, names):
    """Keep retrieval on the file requested now, rather than a previous turn's file."""
    text = user_text(question).casefold()
    explicit = [name for name in names if name.casefold() in text]
    if explicit:
        return explicit
    if re.search(r"\bpdf\b", text):
        return [name for name in names if name.lower().endswith('.pdf')]
    if re.search(r"\b(?:word|docx)\b", text):
        return [name for name in names if name.lower().endswith('.docx')]
    footer = re.search(r"\n\nAttached files: (.*)", question)
    if footer:
        attached = [name for name in names if name in footer.group(1)]
        if attached:
            return attached
    return []
