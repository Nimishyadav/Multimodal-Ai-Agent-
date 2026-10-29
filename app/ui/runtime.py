"""Refresh stale presentation helpers after a Streamlit reload."""
import importlib
from app.ui import presentation as presentation_module


def load_presentation():
    module=presentation_module
    required=('public_activity','simple_activity','public_answer','public_error')
    if not all(callable(getattr(module,name,None)) for name in required):
        importlib.invalidate_caches()
        module=importlib.reload(module)
    return module
