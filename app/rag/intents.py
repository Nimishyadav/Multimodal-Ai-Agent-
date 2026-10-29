"""Recognize general document overviews without replacing topic-specific searches."""
import re
import json


def is_document_overview(text):
    text = re.split(r'\n\n(?:Attached files|Files NOT read):', text, maxsplit=1)[0]
    text = re.sub(r'[^\w\s]', '', text.lower()).strip()
    text = ' '.join(text.split())
    if text in {'summary','summarize','summarise','overview','tldr'}:return True
    document = r'(?:(?:the|this|my|uploaded|attached|entire|whole|full)\s+)*(?:documents?|pdf|files?|docx|report|presentation)'
    patterns = [
        rf'(?:please\s+)?(?:explain|describe|summari[sz]e)\s+{document}(?:\s+(?:in\s+)?(?:short|brief|briefly|detail|simple words|simply))?',
        rf'(?:please\s+)?(?:give|provide)\s+(?:me\s+)?(?:a\s+)?(?:short\s+|brief\s+)?(?:summary|overview)\s+of\s+{document}',
        rf'what\s+(?:is|are)\s+{document}\s+about',
        rf'(?:short\s+|brief\s+)?summary\s+of\s+{document}',
    ]
    return any(re.fullmatch(pattern, text) for pattern in patterns)


def brief_source_overview(passages, max_words=90):
    """Return a clearly labeled extractive fallback when generated prose is unavailable."""
    sentences=[];remaining=max_words
    for passage in passages[:3]:
        text=' '.join(passage.get('excerpt',passage.get('text','')).split())
        if not text:continue
        try:
            table = json.loads(text)
        except (ValueError, TypeError):
            table = None
        if table is None and text.startswith('{"columns":'):
            # Saved evidence can be truncated after the preview; the header is still readable.
            match = re.match(r'\{"columns":\s*(\[.*?\]),\s*"rows":\s*(\d+)', text)
            if match:
                table = {'columns': json.loads(match.group(1)), 'rows': int(match.group(2))}
        if isinstance(table, dict) and isinstance(table.get('columns'), list):
            columns = ', '.join(str(column) for column in table['columns'])
            rows = table.get('rows', 'an unknown number of')
            return f"This spreadsheet contains {rows} rows with these columns: {columns}. The preview does not verify any totals."
        sentence=re.split(r'(?<=[.!?])\s+',text,maxsplit=1)[0]
        words=sentence.split()[:min(35,remaining)]
        if not words:break
        sentence=' '.join(words)
        if sentence not in sentences:sentences.append(sentence);remaining-=len(words)
    if not sentences:return 'No readable source text is available. Please attach a readable document.'
    return 'From sampled document excerpts: '+' '.join(sentences)
