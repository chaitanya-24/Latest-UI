# def extract_input_message(args, kwargs):
#     """Extract user input message for telemetry from args/kwargs."""
#     # Check for dict with 'user_input' in args
#     if args:
#         for arg in args:
#             if isinstance(arg, dict) and "user_input" in arg:
#                 return arg["user_input"]
#     # Check for dict with 'user_input' in kwargs
#     for v in kwargs.values():
#         if isinstance(v, dict) and "user_input" in v:
#             return v["user_input"]
#     # Fallback: first string arg
#     if args:
#         for arg in args:
#             if isinstance(arg, str):
#                 return arg
#     return ""


def extract_input_message(args, kwargs):
    """Extract user input message for telemetry from args/kwargs (CrewAI/LangChain/LangGraph/General)."""
    # 1. Check dict args for 'message', 'input', 'user_input'
    if args:
        for arg in args:
            if isinstance(arg, dict):
                for key in ['message', 'input', 'user_input']:
                    if key in arg:
                        return arg[key]
    # 2. First string arg
    for arg in args:
        if isinstance(arg, str):
            return arg
    # 3. Check dict kwargs for 'message', 'input', 'user_input'
    for v in kwargs.values():
        if isinstance(v, dict):
            for key in ['message', 'input', 'user_input']:
                if key in v:
                    return v[key]
    # 4. Any string in kwargs
    for v in kwargs.values():
        if isinstance(v, str):
            return v
    return ""



def detect_provider(instance, provider_attr=None):
    """Detect provider name from instance, provider_attr, or class/module heuristics."""
    # Try direct attribute
    if provider_attr:
        provider = getattr(instance, provider_attr, None)
        if provider and provider != "unknown":
            return provider.lower()
    # Try common provider attribute
    provider = getattr(instance, "provider", None)
    if provider and provider != "unknown":
        return provider.lower()
    # Try LLM object
    llm = getattr(instance, "llm", None)
    if llm:
        provider = getattr(llm, "provider", None)
        if provider and provider != "unknown":
            return provider.lower()
        # Try model string on llm
        model_str = getattr(llm, "model", None) or getattr(llm, "model_name", None)
        if isinstance(model_str, str):
            if model_str.startswith("azure/"):
                return "azure"
            elif model_str.startswith("openai/"):
                return "openai"
            elif model_str.startswith("anthropic/"):
                return "anthropic"
            elif model_str.startswith("google/") or model_str.startswith("gemini/"):
                return "google"
        class_name = llm.__class__.__name__.lower()
        if "openai" in class_name:
            return "openai"
        elif "anthropic" in class_name or "claude" in class_name:
            return "anthropic"
        elif "azure" in class_name:
            return "azure"
        elif "google" in class_name or "gemini" in class_name:
            return "google"
    # Try model string on instance
    model_str = getattr(instance, "model", None) or getattr(instance, "model_name", None)
    if isinstance(model_str, str):
        if model_str.startswith("azure/"):
            return "azure"
        elif model_str.startswith("openai/"):
            return "openai"
        elif model_str.startswith("anthropic/"):
            return "anthropic"
        elif model_str.startswith("google/") or model_str.startswith("gemini/"):
            return "google"
    # Try instance class name
    class_name = instance.__class__.__name__.lower()
    if "openai" in class_name:
        return "openai"
    elif "anthropic" in class_name or "claude" in class_name:
        return "anthropic"
    elif "azure" in class_name:
        return "azure"
    elif "google" in class_name or "gemini" in class_name:
        return "google"
    # Try module name
    module = getattr(instance.__class__, "__module__", "")
    if "openai" in module:
        return "openai"
    elif "anthropic" in module or "claude" in module:
        return "anthropic"
    elif "azure" in module:
        return "azure"
    elif "google" in module or "gemini" in module:
        return "google"
    return "unknown"
# Token counting helpers and small utilities

import threading

try:
    import tiktoken
    _TIKTOKEN_AVAILABLE = True
except Exception:
    tiktoken = None
    _TIKTOKEN_AVAILABLE = False

_ENCODER_CACHE: dict = {}
_ENCODER_LOCK = threading.Lock()


def _get_encoder(model: str | None):
    """Return a cached tiktoken encoder for model or None if unavailable."""
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
    """Return token count for text using cached tiktoken encoder when available, otherwise a heuristic."""
    if not text:
        return 0
    enc = _get_encoder(model)
    if enc is not None:
        try:
            return len(enc.encode(text))
        except Exception:
            pass
    # Heuristic fallback: characters / 4
    return max(1, len(text) // 4)

import importlib

def _safe_wrap(package, obj_path, wrapper):
    try:
        mod = importlib.import_module(package)
    except Exception:
        return False
    parent = mod
    for part in obj_path.split('.'):
        if not hasattr(parent, part):
            return False
        parent = getattr(parent, part)
    from wrapt import wrap_function_wrapper
    wrap_function_wrapper(package, obj_path, wrapper)
    return True


def count_chat_tokens(messages, model: str | None = None) -> int:
    """Count tokens for chat-style messages (list of dicts) using tiktoken when possible.

    Uses OpenAI-style framing defaults when model looks like a gpt model.
    """
    if not messages:
        return 0

    # Decide framing rules conservatively
    model_key = (model or "").lower()
    # Default framing values (based on common OpenAI guidance)
    tokens_per_message = 3
    tokens_per_name = 1
    if "gpt" in model_key or model_key.startswith("gpt-"):
        tokens_per_message = 4
        tokens_per_name = -1

    enc = _get_encoder(model)
    total = 0
    for msg in messages:
        total += tokens_per_message
        if isinstance(msg, dict):
            # common keys: role, name, content
            for k, v in msg.items():
                if v is None:
                    continue
                text = str(v)
                if enc is not None:
                    try:
                        total += len(enc.encode(text))
                    except Exception:
                        total += max(1, len(text) // 4)
                else:
                    total += max(1, len(text) // 4)
                if k == "name":
                    total += tokens_per_name
        else:
            # fallback: encode as string
            text = str(msg)
            if enc is not None:
                try:
                    total += len(enc.encode(text))
                except Exception:
                    total += max(1, len(text) // 4)
            else:
                total += max(1, len(text) // 4)

    total += 3  # reply priming tokens (model dependent)
    return int(total)


def _extract_text_from_arg(arg):
    """Normalize common prompt/message types to plain text for token counting."""
    # Strings
    if isinstance(arg, str):
        return arg

    # LangChain/Chat messages (list of dicts with 'content' or 'message' structures)
    if isinstance(arg, list):
        parts = []
        for item in arg:
            if isinstance(item, dict):
                # common keys: 'content', 'text', 'message'
                text = item.get("content") or item.get("text") or item.get("message")
                if text:
                    parts.append(str(text))
            else:
                parts.append(str(item))
        return "\n".join(parts)

    # Single message dict
    if isinstance(arg, dict):
        text = arg.get("content") or arg.get("text") or arg.get("message")
        if text:
            return str(text)
        # Fallback to string representation
        return str(arg)

    # Other types -> string representation
    try:
        return str(arg)
    except Exception:
        return ""


def extract_tokens(args, result, model: str | None = None):
    """Compute (input_tokens, output_tokens).

    - args: the positional args passed to the LLM call (instrumentors usually pass args tuple)
    - result: the returned value from the LLM call
    - model: optional model name to improve token estimation with tiktoken

    Returns a tuple of integers (input_tokens, output_tokens)
    """
    input_tokens = 0
    try:
        # Input: try to extract user input message
        input_text = extract_input_message(args, {})
        input_tokens = _safe_encode(input_text, model)

        # Output: handle several possible result shapes
        if result is None:
            output_tokens = 0
        else:
            # If result is a string
            if isinstance(result, str):
                output_tokens = _safe_encode(result, model)
            # If result has 'usage' with token counts (server-reported), prefer it
            elif isinstance(result, dict) and "usage" in result:
                try:
                    output_tokens = int(result["usage"].get("completion_tokens", 0))
                except Exception:
                    output_tokens = 0
            # If result has 'content' or 'raw' or 'output' fields
            elif hasattr(result, "content"):
                output_tokens = _safe_encode(getattr(result, "content", ""), model)
            elif hasattr(result, "raw"):
                output_tokens = _safe_encode(getattr(result, "raw", ""), model)
            elif isinstance(result, dict) and "output" in result:
                output_tokens = _safe_encode(result["output"], model)
            elif isinstance(result, dict) and "result" in result:
                output_tokens = _safe_encode(result["result"], model)
            else:
                try:
                    output_tokens = _safe_encode(str(result), model)
                except Exception:
                    output_tokens = 0
    except Exception:
        input_tokens = 0
        output_tokens = 0
    return int(input_tokens), int(output_tokens)


# def _find_max_tokens(instance, kwargs):
#         """A robust method to find the max_tokens parameter."""
        
#         # List of possible keys for the max_tokens parameter
#         possible_keys = ['max_tokens', 'max_output_tokens', 'max_tokens_to_sample']

#         # 1. Check in the direct keyword arguments of the call
#         for key in possible_keys:
#             if key in kwargs:
#                 return kwargs[key]
        
#         # 2. Check in nested dictionary arguments of the call (e.g., 'generation_options')
#         for arg_name in ['generation_options', 'invocation_params', 'model_kwargs', 'options']:
#             nested_dict = kwargs.get(arg_name)
#             if isinstance(nested_dict, dict):
#                 for key in possible_keys:
#                     if key in nested_dict:
#                         return nested_dict[key]

#         # 3. Check for the value as a direct attribute on the model instance
#         for key in possible_keys:
#             value = getattr(instance, key, None)
#             if value is not None:
#                 return value

#         # 4. Check for the value in nested dictionaries on the model instance (e.g., 'model_kwargs')
#         for attr_name in ['model_kwargs', 'client_kwargs', 'generation_kwargs']:
#             nested_dict = getattr(instance, attr_name, None)
#             if isinstance(nested_dict, dict):
#                 for key in possible_keys:
#                     if key in nested_dict:
#                         return nested_dict[key]
        
#         return None # Return None if not found anywhere


def find_model_version(instance, kwargs):
        """A robust method to find the max_tokens parameter."""
        
        # List of possible keys for the max_tokens parameter
        possible_keys = ['openai_api_version','api_version']

        # 1. Check in the direct keyword arguments of the call
        for key in possible_keys:
            if key in kwargs:
                return kwargs[key]
        
        # 2. Check in nested dictionary arguments of the call (e.g., 'generation_options')
        for arg_name in ['generation_options', 'invocation_params', 'model_kwargs', 'options']:
            nested_dict = kwargs.get(arg_name)
            if isinstance(nested_dict, dict):
                for key in possible_keys:
                    if key in nested_dict:
                        return nested_dict[key]

        # 3. Check for the value as a direct attribute on the model instance
        for key in possible_keys:
            value = getattr(instance, key, None)
            if value is not None:
                return value

        # 4. Check for the value in nested dictionaries on the model instance (e.g., 'model_kwargs')
        for attr_name in ['model_kwargs', 'client_kwargs', 'generation_kwargs']:
            nested_dict = getattr(instance, attr_name, None)
            if isinstance(nested_dict, dict):
                for key in possible_keys:
                    if key in nested_dict:
                        return nested_dict[key]
        
        return None # Return None if not found anywhere



# def temperature(instance, kwargs):
#         """A robust method to find the max_tokens parameter."""
        
#         # List of possible keys for the max_tokens parameter
#         possible_keys = ['temperature']

#         # 1. Check in the direct keyword arguments of the call
#         for key in possible_keys:
#             if key in kwargs:
#                 return kwargs[key]
        
#         # 2. Check in nested dictionary arguments of the call (e.g., 'generation_options')
#         for arg_name in ['generation_options', 'invocation_params', 'model_kwargs', 'options']:
#             nested_dict = kwargs.get(arg_name)
#             if isinstance(nested_dict, dict):
#                 for key in possible_keys:
#                     if key in nested_dict:
#                         return nested_dict[key]

#         # 3. Check for the value as a direct attribute on the model instance
#         for key in possible_keys:
#             value = getattr(instance, key, None)
#             if value is not None:
#                 return value

#         # 4. Check for the value in nested dictionaries on the model instance (e.g., 'model_kwargs')
#         for attr_name in ['model_kwargs', 'client_kwargs', 'generation_kwargs']:
#             nested_dict = getattr(instance, attr_name, None)
#             if isinstance(nested_dict, dict):
#                 for key in possible_keys:
#                     if key in nested_dict:
#                         return nested_dict[key]
        
#         return None # Return None if not found anywhere



# def top_p(instance, kwargs):
#         """A robust method to find the max_tokens parameter."""
        
#         # List of possible keys for the max_tokens parameter
#         possible_keys = ['top_p']

#         # 1. Check in the direct keyword arguments of the call
#         for key in possible_keys:
#             if key in kwargs:
#                 return kwargs[key]
        
#         # 2. Check in nested dictionary arguments of the call (e.g., 'generation_options')
#         for arg_name in ['generation_options', 'invocation_params', 'model_kwargs', 'options']:
#             nested_dict = kwargs.get(arg_name)
#             if isinstance(nested_dict, dict):
#                 for key in possible_keys:
#                     if key in nested_dict:
#                         return nested_dict[key]

#         # 3. Check for the value as a direct attribute on the model instance
#         for key in possible_keys:
#             value = getattr(instance, key, None)
#             if value is not None:
#                 return value

#         # 4. Check for the value in nested dictionaries on the model instance (e.g., 'model_kwargs')
#         for attr_name in ['model_kwargs', 'client_kwargs', 'generation_kwargs']:
#             nested_dict = getattr(instance, attr_name, None)
#             if isinstance(nested_dict, dict):
#                 for key in possible_keys:
#                     if key in nested_dict:
#                         return nested_dict[key]
        
#         return None # Return None if not found anywhere


from datetime import datetime

from iagentops.context import _CONTEXT_CV

def get_active_context(kwargs):
    """
    Merge explicit kwargs with the global ContextVar context.
    Kwargs take precedence.
    """
    ctx = _CONTEXT_CV.get().copy()
    
    # Allow passing these via kwargs as well for override/legacy support
    if "conversation_id" in kwargs:
        ctx["conversation_id"] = kwargs["conversation_id"]
    if "data_source_id" in kwargs:
        ctx["data_source_id"] = kwargs["data_source_id"]
        
    return ctx

def emit_agent_telemetry(span, instance, args, kwargs, result=None, model=None, duration=None, agent_id=None):
    """
    Add agent attributes and prompt/completion events to the span.
    """
    # --- Robust model extraction ---
    if not model or model == "unknown":
        # Try helpers
        model = (
            getattr(instance, "model", None)
            or getattr(instance, "model_name", None)
            or (getattr(getattr(instance, "llm", None), "model", None) if hasattr(instance, "llm") else None)
            or (getattr(getattr(instance, "llm", None), "model_name", None) if hasattr(instance, "llm") else None)
        ) or "unknown"
        # Try nested .llm.llm
        try:
            deep_llm = getattr(getattr(instance, "llm", None), "llm", None)
            if deep_llm:
                model = getattr(deep_llm, "model", None) or getattr(deep_llm, "model_name", None) or model
        except Exception:
            pass

    # --- Agent info ---
    # Prioritize explicit agent_id if set (e.g. from iagentops.init)
    if agent_id:
        final_agent_id = agent_id
    else:
        # Fallback to instance attributes
        final_agent_id = getattr(instance, "id", None) or getattr(instance, "agent_id", None)
        
    agent_name = getattr(instance, "name", None) or getattr(instance, "agent_name", None) or "unknown"
    agent_desc = getattr(instance, "description", None) or ""
    agent_role = getattr(instance, "role", None)

    # Improved Agent ID logic: 
    # If using instance ID and it looks like a UUID (length 36) but we have a role/name, prefer role/name for stability
    # BUT if explicit agent_id was passed, we trust it matches user intent.
    _agent_id_str = str(final_agent_id) if final_agent_id is not None else "unknown"
    
    # Only apply heuristic fallback if we are using the instance ID (not explicit global ID)
    # and it looks unstable.
    if not agent_id and agent_role:
        _agent_id_str = agent_role.lower().replace(" ", "_")
    elif not agent_id and agent_name and agent_name != "unknown":
         if not final_agent_id or len(str(final_agent_id)) > 30: # Heuristic for random UUID
            _agent_id_str = agent_name.lower().replace(" ", "_")

    agent_tools = getattr(instance, "tools", [])
    if isinstance(agent_tools, list):
        agent_tools = ", ".join([getattr(t, "name", str(t)) for t in agent_tools])

    # --- Prompt / completion ---
    prompt = extract_input_message(args, kwargs)
    completion = None
    if result is not None:
        if isinstance(result, dict) and "completion" in result:
            completion = result["completion"]
        elif hasattr(result, "completion"):
            completion = result.completion
        elif isinstance(result, str):
            completion = result
        else:
            completion = str(result)

    # --- Token usage ---
    input_tokens, output_tokens = extract_tokens(args, result, model)
    token_usage = input_tokens + output_tokens

    # --- Duration ---
    duration = duration or 0

    # --- Set agent and request attributes ---
    span.set_attribute("gen_ai.operation.name", "invoke_agent")
    # span.set_attribute("gen_ai.system", "crewai") # Don't hardcode system here, let caller do it
    
    # Use the computed stable agent ID
    span.set_attribute("agent.id", _agent_id_str)
    
    span.set_attribute("gen_ai.agent.name", agent_name)
    span.set_attribute("gen_ai.agent.description", agent_desc)
    span.set_attribute("gen_ai.agent.tools", agent_tools)
    span.set_attribute("gen_ai.request.model", model)
    span.set_attribute("gen_ai.client.token.usage", token_usage)
    span.set_attribute("gen_ai.client.operation.duration", duration)
    
    # --- Context Propagation ---
    ctx = get_active_context(kwargs)
    span.set_attribute("gen_ai.conversation.id", ctx.get("conversation_id", "unknown"))
    span.set_attribute("gen_ai.data_source.id", ctx.get("data_source_id", "unknown"))
    span.set_attribute("gen_ai.agent.name", agent_name)
    span.set_attribute("gen_ai.agent.description", agent_desc)
    span.set_attribute("gen_ai.agent.tools", agent_tools)
    span.set_attribute("gen_ai.request.model", model)
    span.set_attribute("gen_ai.client.token.usage", token_usage)
    span.set_attribute("gen_ai.client.operation.duration", duration)
    # Do not override deployment.environment or service.name here; they are set on the Resource/span elsewhere

    # --- Emit events ---
    now = int(datetime.utcnow().timestamp() * 1e9)
    if prompt:
        span.add_event("gen_ai.content.prompt", {"gen_ai.prompt": prompt}, timestamp=now)
    if completion:
        span.add_event("gen_ai.content.completion", {"gen_ai.completion": completion}, timestamp=now)


# def frequency_penalty(instance, kwargs):
#     if 'frequency_penalty' in kwargs: return kwargs['frequency_penalty']
#     for k in ['generation_options','invocation_params','model_kwargs','options']:
#         d = kwargs.get(k); 
#         if isinstance(d, dict) and 'frequency_penalty' in d: return d['frequency_penalty']
#     v = getattr(instance, 'frequency_penalty', None)
#     if v is not None: return v
#     for k in ['model_kwargs','client_kwargs','generation_kwargs']:
#         d = getattr(instance, k, None)
#         if isinstance(d, dict) and 'frequency_penalty' in d: return d['frequency_penalty']
#     return None

# def presence_penalty(instance, kwargs):
#     if 'presence_penalty' in kwargs: return kwargs['presence_penalty']
#     for k in ['generation_options','invocation_params','model_kwargs','options']:
#         d = kwargs.get(k); 
#         if isinstance(d, dict) and 'presence_penalty' in d: return d['presence_penalty']
#     v = getattr(instance, 'presence_penalty', None)
#     if v is not None: return v
#     for k in ['model_kwargs','client_kwargs','generation_kwargs']:
#         d = getattr(instance, k, None)
#         if isinstance(d, dict) and 'presence_penalty' in d: return d['presence_penalty']
#     return None

# def stop_sequences(instance, kwargs):
#     for key in ['stop','stop_sequences']:
#         if key in kwargs: return kwargs[key]
#     for k in ['generation_options','invocation_params','model_kwargs','options']:
#         d = kwargs.get(k)
#         if isinstance(d, dict):
#             for key in ['stop','stop_sequences']:
#                 if key in d: return d[key]
#     for attr in ['stop','stop_sequences']:
#         v = getattr(instance, attr, None)
#         if v is not None: return v
#     for k in ['model_kwargs','client_kwargs','generation_kwargs']:
#         d = getattr(instance, k, None)
#         if isinstance(d, dict):
#             for key in ['stop','stop_sequences']:
#                 if key in d: return d[key]
#     return None

# def seed(instance, kwargs):
#     if 'seed' in kwargs: return kwargs['seed']
#     for k in ['generation_options','invocation_params','model_kwargs','options']:
#         d = kwargs.get(k); 
#         if isinstance(d, dict) and 'seed' in d: return d['seed']
#     v = getattr(instance, 'seed', None)
#     if v is not None: return v
#     for k in ['model_kwargs','client_kwargs','generation_kwargs']:
#         d = getattr(instance, k, None)
#         if isinstance(d, dict) and 'seed' in d: return d['seed']
#     return None


def _get_nested_attr(obj, attr):
    """Helper to get attribute from nested objects (e.g., llm.llm)"""
    if not obj:
        return None
    if hasattr(obj, attr):
        return getattr(obj, attr)
    if hasattr(obj, 'llm'):
        return _get_nested_attr(obj.llm, attr)
    return None

def _get_param(instance, kwargs, param_name, possible_keys=None):
    """Generic parameter getter that checks multiple locations"""
    if possible_keys is None:
        possible_keys = [param_name]
    
    # 1. Check direct kwargs
    for key in possible_keys:
        if key in kwargs:
            return kwargs[key]
    
    # 2. Check nested dicts in kwargs
    for dict_key in ['generation_options', 'invocation_params', 'model_kwargs', 'options']:
        d = kwargs.get(dict_key, {})
        if not isinstance(d, dict):
            continue
        for key in possible_keys:
            if key in d:
                return d[key]
    
    # 3. Check instance and nested LLMs
    for key in possible_keys:
        value = _get_nested_attr(instance, key)
        if value is not None:
            return value
    
    # 4. Check model_kwargs, etc. on instance
    for dict_key in ['model_kwargs', 'client_kwargs', 'generation_kwargs']:
        d = getattr(instance, dict_key, {})
        if not isinstance(d, dict):
            continue
        for key in possible_keys:
            if key in d:
                return d[key]
    
    return None


def _extract_param(instance, param_name, default=None):
    """
    Extract a parameter from an instance, checking multiple possible locations
    in the same way model extraction works.
    """
    checked = set()
    
    def read(obj):
        if not obj or id(obj) in checked:
            return None
        checked.add(id(obj))
        
        # 1. Direct attribute
        value = getattr(obj, param_name, None)
        if value is not None:
            return value
            
        # 2. Pydantic/dict-like access
        for getter in ("model_dump", "dict", "get"):
            try:
                if hasattr(obj, getter):
                    d = getattr(obj, getter)()
                    if isinstance(d, dict) and param_name in d:
                        return d[param_name]
            except Exception:
                pass
                
        # 3. Nested LLM objects
        llm = getattr(obj, "llm", None)
        if llm:
            nested_value = read(llm)
            if nested_value is not None:
                return nested_value
                
        # 4. Common containers (agent, parent, crew)
        for sub in ("agent", "parent", "crew"):
            sub_obj = getattr(obj, sub, None)
            if sub_obj:
                nested_value = read(sub_obj)
                if nested_value is not None:
                    return nested_value
                    
        # 5. Crew agents
        agents = getattr(obj, "agents", None)
        if isinstance(agents, (list, tuple)):
            for agent in agents:
                nested_value = read(agent)
                if nested_value is not None:
                    return nested_value
                    
        return None
        
    return read(instance) or default


def temperature(instance, kwargs):
    """Get temperature from instance or kwargs"""
    # Try kwargs first
    if 'temperature' in kwargs:
        return kwargs['temperature']
    # Then try instance with nested lookup
    return _extract_param(instance, 'temperature')

def top_p(instance, kwargs):
    if 'top_p' in kwargs:
        return kwargs['top_p']
    return _extract_param(instance, 'top_p')

def frequency_penalty(instance, kwargs):
    if 'frequency_penalty' in kwargs:
        return kwargs['frequency_penalty']
    return _extract_param(instance, 'frequency_penalty')

def presence_penalty(instance, kwargs):
    if 'presence_penalty' in kwargs:
        return kwargs['presence_penalty']
    return _extract_param(instance, 'presence_penalty')

def seed(instance, kwargs):
    if 'seed' in kwargs:
        return kwargs['seed']
    return _extract_param(instance, 'seed')

def stop_sequences(instance, kwargs):
    if 'stop' in kwargs:
        return kwargs['stop']
    if 'stop_sequences' in kwargs:
        return kwargs['stop_sequences']
    return _extract_param(instance, 'stop') or _extract_param(instance, 'stop_sequences')

def _find_max_tokens(instance, kwargs):
    if 'max_tokens' in kwargs:
        return kwargs['max_tokens']
    if 'max_output_tokens' in kwargs:
        return kwargs['max_output_tokens']
    return _extract_param(instance, 'max_tokens') or _extract_param(instance, 'max_output_tokens')


# # Update all parameter getters to use _get_param
# def temperature(instance, kwargs):
#     return _get_param(instance, kwargs, 'temperature')

# def top_p(instance, kwargs):
#     return _get_param(instance, kwargs, 'top_p')

# def frequency_penalty(instance, kwargs):
#     return _get_param(instance, kwargs, 'frequency_penalty')

# def presence_penalty(instance, kwargs):
#     return _get_param(instance, kwargs, 'presence_penalty')

# def seed(instance, kwargs):
#     return _get_param(instance, kwargs, 'seed')

# def stop_sequences(instance, kwargs):
#     return _get_param(instance, kwargs, 'stop_sequences', ['stop', 'stop_sequences'])

# def _find_max_tokens(instance, kwargs):
#     return _get_param(instance, kwargs, 'max_tokens', ['max_tokens', 'max_output_tokens', 'max_tokens_to_sample'])