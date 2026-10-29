"""Save readable attachment content with its chat, without reparsing on reload."""
from dataclasses import asdict
from io import StringIO
import pandas as pd
from app.rag.documents import Chunk


def snapshot_attachments(store):
    index = store.documents
    return {
        'version': 1,
        'items': store.items,
        'documents': {name: {'digest': digest, 'chunks': [asdict(c) for c in chunks]}
                      for name, (digest, chunks) in index._documents.items()},
        'tables': {name: frame.to_json(orient='table', date_format='iso', double_precision=15)
                   for name, frame in store.tables.items()},
        'images': store.images,
        'image_notes': store.image_notes,
        'rag_jobs': getattr(index, '_jobs', {}),
        'backend_status': getattr(index, 'backend_status', {}),
        'rag_namespace': getattr(getattr(index, 'client', None), 'directory', None).name if getattr(getattr(index, 'client', None), 'directory', None) else None,
    }


def restore_attachments(store, snapshot):
    if not snapshot:
        return store
    if snapshot.get('version') != 1:
        raise ValueError('Unsupported saved attachment version.')
    # Build the restored values first so a damaged snapshot cannot partly overwrite a chat.
    documents = {name: (entry['digest'], [Chunk(**c) for c in entry['chunks']])
                 for name, entry in snapshot['documents'].items()}
    tables = {name: pd.read_json(StringIO(data), orient='table')
              for name, data in snapshot['tables'].items()}
    store.documents._documents = documents
    store.items = snapshot['items']
    store.tables = tables
    store.images = snapshot['images']
    store.image_notes = snapshot.get('image_notes', {})
    if hasattr(store.documents, '_jobs'):
        store.documents._jobs = snapshot.get('rag_jobs', {})
        store.documents.backend_status = snapshot.get('backend_status', {})
        namespace = snapshot.get('rag_namespace')
        if namespace:
            from app.rag.anything_rag import RagWorkerClient
            store.documents.client = RagWorkerClient(namespace=namespace)
    return store
