"""Refresh cached history objects after Streamlit reloads the application."""
import importlib
from app.history import store as store_module


def upgrade_history_store(store):
    required = ('load_attachments', 'save_attachments')
    if all(callable(getattr(store, name, None)) for name in required):
        return store
    # Streamlit can retain both an old instance and its imported module.
    if not all(hasattr(store_module.HistoryStore, name) for name in required):
        importlib.reload(store_module)
    return store_module.HistoryStore(path=store.path)
