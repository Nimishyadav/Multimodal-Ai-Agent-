"""Keep existing uploaded files usable after a Streamlit code reload."""
import importlib
from app.attachments import store as store_module

# A running Streamlit session may still hold an older AttachmentStore class.
if not hasattr(store_module.AttachmentStore, 'cached_image_answer'):
    importlib.reload(store_module)
AttachmentStore = store_module.AttachmentStore


def upgrade_attachment_store(store):
    required = ('image_answers','image_notes','vision_state')
    if (isinstance(store, AttachmentStore) and getattr(store,'schema_version',0)==2
            and all(hasattr(store,field) for field in required)):
        return store
    upgraded = AttachmentStore(store.documents)
    for field in ['items','tables','images','image_answers','image_notes','vision_state']:
        if hasattr(store,field):
            setattr(upgraded,field,getattr(store,field))
    return upgraded
