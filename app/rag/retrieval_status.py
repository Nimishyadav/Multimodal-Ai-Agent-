"""Describe the retrieval that actually ran, without exposing provider details."""


def document_retrieval_activity(result):
    info = result.get('retrieval_info', {})
    method = result.get('method', 'source search')
    if result.get('status') in {'error', 'no_documents'}:
        return 'Document retrieval: no source evidence available.'
    if method == 'exact_page':
        page = result.get('requested_page', '')
        return f'Document retrieval: exact PDF page {page}; hybrid search skipped to keep the requested page scope.'
    if info.get('hybrid_used') and result.get('multimodal_context'):
        return 'Document retrieval: Anything RAG hybrid graph/vector context + local source passages. Pipeline: Document Agent → search_documents → RAG-Anything → graph/vector retrieval → answer evidence.'
    if info.get('backend') == 'anything_rag':
        reasons = {
            'index_not_ready': 'Anything RAG index is not ready for the selected file',
            'document_not_found': 'the requested filename does not match an uploaded file',
            'selected_document_not_indexed': 'other files are indexed, but the selected file has no ready RAG-Anything index',
            'model_unavailable': 'using local evidence while AI processing is unavailable',
            'query_failed': 'Anything RAG query was unavailable or exceeded its time budget',
            'query_cooldown': 'a recent hybrid query failed; local evidence avoids repeating the wait',
            'no_hybrid_context': 'Anything RAG returned no usable context',
        }
        reason = reasons.get(info.get('reason'), 'hybrid context was not used')
        return f'Document retrieval: local {method} fallback; {reason}.'
    return f'Document retrieval: local {method}; standard backend selected.'
