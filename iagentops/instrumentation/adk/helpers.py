import threading
import json
from datetime import datetime

# --- TIKTOKEN SETUP ---
try:
    import tiktoken
    _TIKTOKEN_AVAILABLE = True
except Exception:
    tiktoken = None
    _TIKTOKEN_AVAILABLE = False

_ENCODER_CACHE: dict = {}
_ENCODER_LOCK = threading.Lock()

def _get_encoder(model: str | None):
    if not _TIKTOKEN_AVAILABLE:
        return None
    key = (model or "cl100k_base").lower()
    with _ENCODER_LOCK:
        if key in _ENCODER_CACHE:
            return _ENCODER_CACHE[key]
        try:
            if model:
                enc = tiktoken.encoding_for_model(model)
            else:
                enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            try:
                enc = tiktoken.get_encoding("cl100k_base")
            except Exception:
                enc = None
        _ENCODER_CACHE[key] = enc
        return enc

def _safe_encode(text: str, model: str | None = None) -> int:
    if not text:
        return 0
    enc = _get_encoder(model)
    if enc is not None:
        try:
            return len(enc.encode(text))
        except Exception:
            pass
    return max(1, len(text) // 4)

# --- ADK SPECIFIC EXTRACTION ---

def _extract_text_from_adk_object(obj):
    """
    Helper to extract text from Google Content/Part objects.
    Can handle: Content, Part, or objects with 'text' attribute.
    """
    if not obj:
        return None
        
    try:
        # 1. Check if it's a Content object (has 'parts' list)
        if hasattr(obj, "parts"):
            parts = getattr(obj, "parts", [])
            # Iterate parts to find text
            if isinstance(parts, list):
                text_parts = []
                for part in parts:
                    txt = getattr(part, "text", None)
                    if txt:
                        text_parts.append(txt)
                if text_parts:
                    return "\n".join(text_parts)
                    
        # 2. Check if it's a Part object (has 'text' attribute directly)
        if hasattr(obj, "text") and isinstance(getattr(obj, "text"), str):
            return getattr(obj, "text")
            
        # 3. Check if it's a Dict (common in some ADK versions)
        if isinstance(obj, dict):
            if "text" in obj: return obj["text"]
            if "query" in obj: return obj["query"]
            if "parts" in obj:
                # Recursive simplistic check for dict-based parts
                p = obj["parts"]
                if isinstance(p, list) and len(p) > 0 and isinstance(p[0], dict):
                    return p[0].get("text", "")
                    
    except Exception:
        pass
    return None

def extract_input_message(args, kwargs):
    """
    Extract user input message from args/kwargs.
    Updated to look INSIDE objects (like InvocationContext) for 'user_content'.
    """
    # 1. Check kwargs for specific ADK keys
    for k in ["user_content", "content", "query", "input", "message"]:
        if k in kwargs:
            val = kwargs[k]
            if isinstance(val, str): return val
            extracted = _extract_text_from_adk_object(val)
            if extracted: return extracted

    # 2. Check positional args
    if args:
        for arg in args:
            if isinstance(arg, str):
                return arg
            
            # A. Check if the arg itself is a Content object
            extracted = _extract_text_from_adk_object(arg)
            if extracted:
                return extracted
            
            # B. Check if arg is a Dict with 'query'
            if isinstance(arg, dict) and "query" in arg:
                return arg["query"]
            
            # C. [CRITICAL FIX] Check if arg is a Context object containing 'user_content'
            # This handles the InvocationContext(user_content=...) case
            if hasattr(arg, "user_content"):
                user_content = getattr(arg, "user_content")
                extracted = _extract_text_from_adk_object(user_content)
                if extracted:
                    return extracted
            
            # D. Check if arg has 'query' attribute
            if hasattr(arg, "query"):
                q = getattr(arg, "query")
                if isinstance(q, str): return q

    return ""

def extract_adk_completion(result):
    """Extracts plain text completion from Google ADK result (List of Events)."""
    if isinstance(result, str):
        return result
        
    text_parts = []
    
    # Handle List of Events (Standard ADK output)
    if isinstance(result, list):
        for item in result:
            content = getattr(item, "content", None)
            if content:
                extracted = _extract_text_from_adk_object(content)
                if extracted:
                    text_parts.append(extracted)
    
    # Handle Single Event or Object
    elif hasattr(result, "content"):
        extracted = _extract_text_from_adk_object(result.content)
        if extracted:
            text_parts.append(extracted)

    if text_parts:
        return "\n".join(text_parts)
    
    return str(result)

def extract_tokens(args, result, model: str | None = None):
    input_tokens = 0
    output_tokens = 0
    found_server_tokens = False

    try:
        # --- PRIORITY 1: Google ADK / Vertex (List of Events) ---
        if isinstance(result, list):
            for item in result:
                meta = getattr(item, "usage_metadata", None)
                if meta:
                    found_server_tokens = True
                    p_t = getattr(meta, "prompt_token_count", 0)
                    c_t = getattr(meta, "candidates_token_count", 0)
                    
                    if p_t: input_tokens = max(input_tokens, int(p_t))
                    if c_t: output_tokens = max(output_tokens, int(c_t))

        # --- PRIORITY 2: Fallback (Calculate manually) ---
        if not found_server_tokens and (input_tokens == 0 and output_tokens == 0):
            # Input
            input_text = extract_input_message(args, {})
            input_tokens = _safe_encode(input_text, model)
            # Output
            output_text = extract_adk_completion(result)
            output_tokens = _safe_encode(output_text, model)

    except Exception:
        pass

    return int(input_tokens), int(output_tokens)

# --- PARAMETER FINDER ---

def _get_param(instance, kwargs, param_name, possible_keys=None):
    if possible_keys is None: possible_keys = [param_name]

    # 1. Direct kwargs
    for key in possible_keys:
        if key in kwargs: return kwargs[key]

    # 2. Nested kwargs
    for dict_key in ['generation_options', 'invocation_params', 'model_kwargs', 'options', 'generate_content_config']:
        d = kwargs.get(dict_key)
        if isinstance(d, dict):
            for key in possible_keys:
                if key in d: return d[key]
        elif d is not None:
             for key in possible_keys:
                if hasattr(d, key): return getattr(d, key)

    # 3. Instance Attributes
    for key in possible_keys:
        if hasattr(instance, key): return getattr(instance, key)

    # 4. GenerateContentConfig (ADK Specific)
    if hasattr(instance, "generate_content_config"):
        gcf = getattr(instance, "generate_content_config")
        if isinstance(gcf, dict):
            for key in possible_keys:
                if key in gcf: return gcf[key]
        else:
             for key in possible_keys:
                if hasattr(gcf, key): return getattr(gcf, key)

    # 5. Nested Model/LLM
    model = getattr(instance, "model", None)
    if model:
        if hasattr(model, "config") and isinstance(model.config, dict):
             for key in possible_keys:
                if key in model.config: return model.config[key]
        if hasattr(model, "generate_content_config"):
            gcf = getattr(model, "generate_content_config")
            for key in possible_keys:
                if hasattr(gcf, key): return getattr(gcf, key)

    return None

def temperature(instance, kwargs):
    return _get_param(instance, kwargs, 'temperature')

def top_p(instance, kwargs):
    return _get_param(instance, kwargs, 'top_p')

def _find_max_tokens(instance, kwargs):
    return _get_param(instance, kwargs, 'max_tokens', ['max_tokens', 'max_output_tokens', 'candidate_count'])

def find_model_version(instance, kwargs):
    return _get_param(instance, kwargs, 'api_version', ['openai_api_version', 'api_version', 'version'])