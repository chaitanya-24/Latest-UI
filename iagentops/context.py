
import contextvars
from typing import Optional

_CONTEXT_CV = contextvars.ContextVar("iagentops_context", default={})

def set_context(conversation_id: Optional[str] = None, data_source_id: Optional[str] = None):
    """
    Set context attributes (conversation_id, data_source_id) for the current execution context.
    These values will be automatically attached to any subsequent spans created by the SDK.
    
    Args:
        conversation_id: Unique identifier for the conversation/thread
        data_source_id: Unique identifier for the data source/document being processed
    """
    current_ctx = _CONTEXT_CV.get().copy()
    if conversation_id is not None:
        current_ctx["conversation_id"] = conversation_id
    if data_source_id is not None:
        current_ctx["data_source_id"] = data_source_id
    _CONTEXT_CV.set(current_ctx)

def get_context():
    return _CONTEXT_CV.get()
